# frozen_string_literal: true

require "fileutils"

module NanoGPT
  # Training loop for GPT models
  class Trainer
    attr_reader :model, :optimizer, :config, :iter_num, :best_val_loss

    def initialize(model:, data_loader:, config: {})
      @model = model
      @data_loader = data_loader
      @config = default_config.merge(config)

      @iter_num = 0
      @best_val_loss = Float::INFINITY

      setup_optimizer
      setup_lr_scheduler
    end

    def default_config
      {
        out_dir: "out",
        eval_interval: 250,
        log_interval: 10,
        eval_iters: 200,
        eval_only: false,
        always_save_checkpoint: false,

        # Optimizer
        learning_rate: 1e-3,
        weight_decay: 1e-1,
        beta1: 0.9,
        beta2: 0.99,
        grad_clip: 1.0,

        # LR scheduler
        decay_lr: true,
        warmup_iters: 100,
        lr_decay_iters: 5000,
        min_lr: 1e-4,

        # Training
        max_iters: 5000,
        gradient_accumulation_steps: 1,

        device: "cpu"
      }
    end

    def train
      puts "Starting training..."
      puts "Tokens per iteration: #{tokens_per_iter}"

      @model.train
      x, y = @data_loader.get_batch(:train)
      t0 = Time.now

      while @iter_num <= @config[:max_iters]
        # Set learning rate for this iteration
        lr = @config[:decay_lr] ? @lr_scheduler.step(@optimizer, @iter_num) : @config[:learning_rate]

        # Evaluate and checkpoint
        if @iter_num % @config[:eval_interval] == 0
          losses = estimate_loss
          puts "step #{@iter_num}: train loss #{losses[:train].round(4)}, val loss #{losses[:val].round(4)}"

          if losses[:val] < @best_val_loss || @config[:always_save_checkpoint]
            @best_val_loss = [losses[:val], @best_val_loss].min
            save_checkpoint if @iter_num > 0
          end
        end

        break if @iter_num == 0 && @config[:eval_only]

        # Forward/backward with gradient accumulation
        @optimizer.zero_grad

        accumulated_loss = 0.0
        @config[:gradient_accumulation_steps].times do |micro_step|
          logits, loss = @model.call(x, targets: y)
          loss = loss / @config[:gradient_accumulation_steps]
          accumulated_loss += loss.item
          loss.backward

          # Prefetch next batch
          x, y = @data_loader.get_batch(:train)
        end

        # Gradient clipping (manual implementation since torch.rb lacks clip_grad_norm_)
        if @config[:grad_clip] > 0.0
          clip_grad_norm(@model.parameters, @config[:grad_clip])
        end

        # Optimizer step
        @optimizer.step

        # Logging
        t1 = Time.now
        dt = t1 - t0
        t0 = t1

        if @iter_num % @config[:log_interval] == 0
          puts "iter #{@iter_num}: loss #{accumulated_loss.round(4)}, time #{(dt * 1000).round(2)}ms, lr #{lr.round(6)}"
        end

        @iter_num += 1
      end

      puts "Training complete!"
    end

    def estimate_loss
      @model.eval
      out = {}

      [:train, :val].each do |split|
        losses = []
        @config[:eval_iters].times do
          x, y = @data_loader.get_batch(split)
          Torch.no_grad do
            _logits, loss = @model.call(x, targets: y)
            losses << loss.item
          end
        end
        out[split] = losses.sum / losses.size
      end

      @model.train
      out
    end

    def save_checkpoint
      FileUtils.mkdir_p(@config[:out_dir])
      path = File.join(@config[:out_dir], "ckpt.pt")

      # Note: torch.rb doesn't support optimizer.state_dict yet
      # We save model state and training metadata
      # Convert symbol keys to strings for Torch.save compatibility
      checkpoint = {
        "model" => @model.state_dict,
        "model_args" => stringify_keys(@model.config.to_h),
        "iter_num" => @iter_num,
        "best_val_loss" => @best_val_loss,
        "config" => stringify_keys(@config)
      }

      Torch.save(checkpoint, path)
      puts "Saved checkpoint to #{path}"
    end

    def load_checkpoint(path)
      checkpoint = Torch.load(path)

      @model.load_state_dict(checkpoint["model"])
      @iter_num = checkpoint["iter_num"]
      @best_val_loss = checkpoint["best_val_loss"]

      # Reinitialize optimizer (since we can't restore optimizer state in torch.rb)
      setup_optimizer

      puts "Loaded checkpoint from #{path} (iter #{@iter_num})"
      checkpoint
    end

    private

    # Convert symbol keys to strings recursively (for Torch.save)
    def stringify_keys(hash)
      hash.transform_keys(&:to_s).transform_values do |v|
        v.is_a?(Hash) ? stringify_keys(v) : v
      end
    end

    # Manual gradient clipping (torch.rb doesn't have clip_grad_norm_)
    def clip_grad_norm(parameters, max_norm)
      total_norm = 0.0
      parameters.each do |p|
        next unless p.grad

        param_norm = p.grad.data.norm(2).item
        total_norm += param_norm ** 2
      end
      total_norm = Math.sqrt(total_norm)

      clip_coef = max_norm / (total_norm + 1e-6)
      if clip_coef < 1
        parameters.each do |p|
          next unless p.grad

          p.grad.data.mul!(clip_coef)
        end
      end

      total_norm
    end

    def setup_optimizer
      @optimizer = @model.configure_optimizers(
        weight_decay: @config[:weight_decay],
        learning_rate: @config[:learning_rate],
        betas: [@config[:beta1], @config[:beta2]],
        device_type: @config[:device]
      )
    end

    def setup_lr_scheduler
      @lr_scheduler = LRScheduler.new(
        learning_rate: @config[:learning_rate],
        min_lr: @config[:min_lr],
        warmup_iters: @config[:warmup_iters],
        lr_decay_iters: @config[:lr_decay_iters]
      )
    end

    def tokens_per_iter
      @config[:gradient_accumulation_steps] * @data_loader.batch_size * @data_loader.block_size
    end
  end
end
