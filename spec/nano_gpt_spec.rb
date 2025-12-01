# frozen_string_literal: true

require "spec_helper"

RSpec.describe NanoGPT do
  describe NanoGPT::GPTConfig do
    it "creates config with default values" do
      config = NanoGPT::GPTConfig.new

      expect(config.block_size).to eq(1024)
      expect(config.vocab_size).to eq(50304)
      expect(config.n_layer).to eq(12)
      expect(config.n_head).to eq(12)
      expect(config.n_embd).to eq(768)
      expect(config.dropout).to eq(0.0)
      expect(config.bias).to eq(true)
    end

    it "creates config with custom values" do
      config = NanoGPT::GPTConfig.new(
        block_size: 256,
        vocab_size: 65,
        n_layer: 6,
        n_head: 6,
        n_embd: 384,
        dropout: 0.2,
        bias: false
      )

      expect(config.block_size).to eq(256)
      expect(config.vocab_size).to eq(65)
      expect(config.n_layer).to eq(6)
      expect(config.n_head).to eq(6)
      expect(config.n_embd).to eq(384)
      expect(config.dropout).to eq(0.2)
      expect(config.bias).to eq(false)
    end

    it "converts to hash" do
      config = NanoGPT::GPTConfig.new(vocab_size: 100, n_layer: 2)
      hash = config.to_h

      expect(hash[:vocab_size]).to eq(100)
      expect(hash[:n_layer]).to eq(2)
    end

    it "calculates head_size" do
      config = NanoGPT::GPTConfig.new(n_embd: 384, n_head: 6)
      expect(config.head_size).to eq(64)
    end
  end

  describe NanoGPT::Layers::LayerNorm do
    it "normalizes input tensor" do
      ln = NanoGPT::Layers::LayerNorm.new(64, bias: true)
      x = Torch.randn(2, 8, 64)

      output = ln.call(x)

      expect(output.shape).to eq([2, 8, 64])
      # Check output is normalized (mean ~0, std ~1 along last dim)
      mean = output.mean(-1)
      expect(mean.abs.max.item).to be < 1e-5
    end

    it "works without bias" do
      ln = NanoGPT::Layers::LayerNorm.new(32, bias: false)
      x = Torch.randn(2, 4, 32)

      output = ln.call(x)

      expect(output.shape).to eq([2, 4, 32])
      expect(ln.bias).to be_nil
    end
  end

  describe NanoGPT::Layers::MLP do
    let(:config) do
      NanoGPT::GPTConfig.new(n_embd: 64, dropout: 0.0, bias: true)
    end

    it "applies feed-forward transformation" do
      mlp = NanoGPT::Layers::MLP.new(config)
      x = Torch.randn(2, 8, 64)

      output = mlp.call(x)

      expect(output.shape).to eq([2, 8, 64])
    end

    it "expands and contracts dimensions (4x expansion)" do
      mlp = NanoGPT::Layers::MLP.new(config)

      # Check internal layer dimensions
      expect(mlp.instance_variable_get(:@c_fc).weight.shape).to eq([256, 64])
      expect(mlp.instance_variable_get(:@c_proj).weight.shape).to eq([64, 256])
    end
  end

  describe NanoGPT::Layers::CausalSelfAttention do
    let(:config) do
      NanoGPT::GPTConfig.new(
        block_size: 32,
        n_embd: 64,
        n_head: 4,
        dropout: 0.0,
        bias: true
      )
    end

    it "applies self-attention" do
      attn = NanoGPT::Layers::CausalSelfAttention.new(config)
      x = Torch.randn(2, 16, 64)

      output = attn.call(x)

      expect(output.shape).to eq([2, 16, 64])
    end

    it "raises error if n_embd not divisible by n_head" do
      bad_config = NanoGPT::GPTConfig.new(n_embd: 63, n_head: 4)

      expect { NanoGPT::Layers::CausalSelfAttention.new(bad_config) }
        .to raise_error(ArgumentError, /divisible/)
    end
  end

  describe NanoGPT::Layers::Block do
    let(:config) do
      NanoGPT::GPTConfig.new(
        block_size: 32,
        n_embd: 64,
        n_head: 4,
        dropout: 0.0,
        bias: true
      )
    end

    it "applies transformer block" do
      block = NanoGPT::Layers::Block.new(config)
      x = Torch.randn(2, 16, 64)

      output = block.call(x)

      expect(output.shape).to eq([2, 16, 64])
    end
  end

  describe NanoGPT::GPT do
    # Use small config for fast tests
    let(:config) do
      NanoGPT::GPTConfig.new(
        block_size: 32,
        vocab_size: 100,
        n_layer: 2,
        n_head: 4,
        n_embd: 64,
        dropout: 0.0,
        bias: true
      )
    end

    it "creates model and reports parameters" do
      expect { NanoGPT::GPT.new(config) }.to output(/number of parameters/).to_stdout
    end

    it "computes forward pass without targets" do
      model = NanoGPT::GPT.new(config)
      idx = Torch.randint(0, config.vocab_size, [2, 16], dtype: :long)

      logits, loss = model.call(idx)

      # Without targets, only last position logits returned
      expect(logits.shape).to eq([2, 1, config.vocab_size])
      expect(loss).to be_nil
    end

    it "computes forward pass with targets" do
      model = NanoGPT::GPT.new(config)
      idx = Torch.randint(0, config.vocab_size, [2, 16], dtype: :long)
      targets = Torch.randint(0, config.vocab_size, [2, 16], dtype: :long)

      logits, loss = model.call(idx, targets: targets)

      expect(logits.shape).to eq([2, 16, config.vocab_size])
      expect(loss).to be_a(Torch::Tensor)
      expect(loss.item).to be > 0
    end

    it "generates tokens" do
      model = NanoGPT::GPT.new(config)
      model.eval
      idx = Torch.zeros(1, 1, dtype: :long)

      output = model.generate(idx, 10, temperature: 1.0, top_k: 10)

      expect(output.shape).to eq([1, 11]) # 1 initial + 10 generated
    end

    it "raises error for sequence exceeding block_size" do
      model = NanoGPT::GPT.new(config)
      idx = Torch.randint(0, config.vocab_size, [1, config.block_size + 1], dtype: :long)

      expect { model.call(idx) }.to raise_error(ArgumentError, /exceeds block_size/)
    end

    it "counts parameters correctly" do
      model = NanoGPT::GPT.new(config)

      num_params = model.num_params
      num_params_with_emb = model.num_params(non_embedding: false)

      expect(num_params).to be > 0
      expect(num_params_with_emb).to be > num_params
    end

    it "crops block size" do
      model = NanoGPT::GPT.new(config)
      original_block_size = model.config.block_size

      model.crop_block_size(16)

      expect(model.config.block_size).to eq(16)
      expect(model.config.block_size).to be < original_block_size
    end

    it "configures optimizer with parameter groups" do
      model = NanoGPT::GPT.new(config)

      optimizer = model.configure_optimizers(
        weight_decay: 0.1,
        learning_rate: 1e-4,
        betas: [0.9, 0.95],
        device_type: "cpu"
      )

      expect(optimizer).to be_a(Torch::Optim::AdamW)

      # Verify parameter groups are set up correctly
      # Group 0: decay params (2D+), Group 1: no-decay params (1D)
      param_groups = optimizer.param_groups
      expect(param_groups.size).to eq(2)
      expect(param_groups[0][:weight_decay]).to eq(0.1)
      expect(param_groups[1][:weight_decay]).to eq(0.0)
    end
  end

  describe "training step" do
    let(:config) do
      NanoGPT::GPTConfig.new(
        block_size: 16,
        vocab_size: 50,
        n_layer: 1,
        n_head: 2,
        n_embd: 32,
        dropout: 0.0,
        bias: true
      )
    end

    it "performs backward pass and optimizer step" do
      model = NanoGPT::GPT.new(config)
      optimizer = model.configure_optimizers(
        weight_decay: 0.1,
        learning_rate: 1e-3,
        betas: [0.9, 0.95],
        device_type: "cpu"
      )

      # Create batch
      x = Torch.randint(0, config.vocab_size, [4, config.block_size], dtype: :long)
      y = Torch.randint(0, config.vocab_size, [4, config.block_size], dtype: :long)

      # Forward
      _logits, loss = model.call(x, targets: y)
      initial_loss = loss.item

      # Backward
      optimizer.zero_grad
      loss.backward
      optimizer.step

      # Check loss changed after step
      _logits, new_loss = model.call(x, targets: y)

      expect(new_loss.item).not_to eq(initial_loss)
    end
  end
end
