# frozen_string_literal: true

module NanoGPT
  # GPT Language Model
  class GPT < Torch::NN::Module
    attr_reader :config

    def initialize(config)
      super()
      raise ArgumentError, "vocab_size must be set" unless config.vocab_size
      raise ArgumentError, "block_size must be set" unless config.block_size

      @config = config

      # Token and position embeddings
      @wte = Torch::NN::Embedding.new(config.vocab_size, config.n_embd)
      @wpe = Torch::NN::Embedding.new(config.block_size, config.n_embd)
      @drop = Torch::NN::Dropout.new(p: config.dropout)

      # Transformer blocks
      @h = Torch::NN::ModuleList.new(
        config.n_layer.times.map { Layers::Block.new(config) }
      )

      # Final layer norm
      @ln_f = Layers::LayerNorm.new(config.n_embd, bias: config.bias)

      # Note: We use weight tying - lm_head shares weights with wte
      # Instead of a separate Linear layer, we use wte.weight directly in forward

      # Initialize weights
      apply(method(:_init_weights))

      # Special scaled init for residual projections (per GPT-2 paper)
      named_parameters.each do |name, param|
        if name.end_with?("c_proj.weight")
          Torch::NN::Init.normal!(param, mean: 0.0, std: 0.02 / Math.sqrt(2 * config.n_layer))
        end
      end

      puts format("number of parameters: %.2fM", num_params / 1e6)
    end

    def num_params(non_embedding: true)
      n_params = parameters.sum(&:numel)
      n_params -= @wpe.weight.numel if non_embedding
      n_params
    end

    def forward(idx, targets: nil)
      b, t = idx.shape
      raise ArgumentError, "Sequence length #{t} exceeds block_size #{@config.block_size}" if t > @config.block_size

      device = idx.device

      # Position indices
      pos = Torch.arange(0, t, dtype: :long, device: device)

      # Embeddings
      tok_emb = @wte.call(idx)  # (B, T, n_embd)
      pos_emb = @wpe.call(pos)  # (T, n_embd)
      x = @drop.call(tok_emb + pos_emb)

      # Transformer blocks
      @h.each { |block| x = block.call(x) }

      # Final layer norm
      x = @ln_f.call(x)

      if targets
        # Training: compute logits for all positions using tied weights
        logits = Torch::NN::Functional.linear(x, @wte.weight, nil)
        loss = Torch::NN::Functional.cross_entropy(
          logits.view(-1, logits.size(-1)),
          targets.view(-1),
          ignore_index: -1
        )
      else
        # Inference: only compute logits for last position (optimization)
        # Use narrow to get last position: x[:, -1:, :]
        x_last = x.narrow(1, x.size(1) - 1, 1)
        logits = Torch::NN::Functional.linear(x_last, @wte.weight, nil)
        loss = nil
      end

      [logits, loss]
    end

    def generate(idx, max_new_tokens, temperature: 1.0, top_k: nil)
      Torch.no_grad do
        max_new_tokens.times do
          # Crop context if exceeds block_size
          idx_cond = if idx.size(1) <= @config.block_size
                       idx
                     else
                       idx.narrow(1, idx.size(1) - @config.block_size, @config.block_size)
                     end

          # Forward pass
          logits, _loss = forward(idx_cond)

          # Get logits for last position and scale by temperature
          # logits shape is (B, 1, vocab_size), squeeze to (B, vocab_size)
          logits = logits.squeeze(1) / temperature

          # Optional top-k filtering
          if top_k
            k = [top_k, logits.size(-1)].min
            v, _indices = logits.topk(k)
            # Get the k-th largest value as threshold
            threshold = v.narrow(1, k - 1, 1)
            # Mask out values below threshold
            logits = logits.masked_fill(logits.lt(threshold), -Float::INFINITY)
          end

          # Sample from probability distribution
          probs = Torch::NN::Functional.softmax(logits, dim: -1)
          idx_next = Torch.multinomial(probs, num_samples: 1)

          # Append to sequence
          idx = Torch.cat([idx, idx_next], dim: 1)
        end
      end

      idx
    end

    def crop_block_size(block_size)
      raise ArgumentError, "Cannot crop to larger block_size" if block_size > @config.block_size

      @config.block_size = block_size

      # Create new embedding with cropped weights
      new_wpe = Torch::NN::Embedding.new(block_size, @config.n_embd)
      Torch.no_grad do
        new_wpe.weight.copy!(@wpe.weight[0...block_size])
      end
      @wpe = new_wpe

      # Update attention masks in all blocks
      @h.each do |block|
        attn = block.instance_variable_get(:@attn)
        next unless attn.instance_variable_defined?(:@mask)

        mask = attn.instance_variable_get(:@mask)
        attn.instance_variable_set(:@mask, mask[nil, nil, 0...block_size, 0...block_size])
      end
    end

    def configure_optimizers(weight_decay:, learning_rate:, betas:, device_type:)
      # Separate parameters into decay and no-decay groups
      # All 2D+ params (weights) get weight decay, 1D params (biases, layernorm) don't
      decay_params = []
      nodecay_params = []

      parameters.each do |param|
        next unless param.requires_grad

        if param.dim >= 2
          decay_params << param
        else
          nodecay_params << param
        end
      end

      num_decay = decay_params.sum(&:numel)
      num_nodecay = nodecay_params.sum(&:numel)
      puts "num decayed parameter tensors: #{decay_params.size}, with #{num_decay} parameters"
      puts "num non-decayed parameter tensors: #{nodecay_params.size}, with #{num_nodecay} parameters"

      # Create optimizer with parameter groups (using symbol keys for torch.rb)
      Torch::Optim::AdamW.new(
        [
          { params: decay_params, weight_decay: weight_decay },
          { params: nodecay_params, weight_decay: 0.0 }
        ],
        lr: learning_rate,
        betas: betas
      )
    end

    private

    def _init_weights(mod)
      case mod
      when Torch::NN::Linear
        Torch::NN::Init.normal!(mod.weight, mean: 0.0, std: 0.02)
        # Check if bias exists (it won't when bias: false)
        if mod.instance_variable_defined?(:@bias) && mod.instance_variable_get(:@bias)
          Torch::NN::Init.zeros!(mod.instance_variable_get(:@bias))
        end
      when Torch::NN::Embedding
        Torch::NN::Init.normal!(mod.weight, mean: 0.0, std: 0.02)
      end
    end
  end
end
