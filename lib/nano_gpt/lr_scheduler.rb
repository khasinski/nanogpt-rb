# frozen_string_literal: true

module NanoGPT
  # Cosine learning rate scheduler with linear warmup
  class LRScheduler
    attr_reader :learning_rate, :min_lr, :warmup_iters, :lr_decay_iters

    def initialize(learning_rate:, min_lr:, warmup_iters:, lr_decay_iters:)
      @learning_rate = learning_rate
      @min_lr = min_lr
      @warmup_iters = warmup_iters
      @lr_decay_iters = lr_decay_iters
    end

    # Get learning rate for given iteration
    def get_lr(iter)
      # 1) Linear warmup for warmup_iters steps
      if iter < @warmup_iters
        return @learning_rate * (iter + 1).to_f / (@warmup_iters + 1)
      end

      # 2) If iter > lr_decay_iters, return min learning rate
      if iter > @lr_decay_iters
        return @min_lr
      end

      # 3) In between, use cosine decay down to min learning rate
      decay_ratio = (iter - @warmup_iters).to_f / (@lr_decay_iters - @warmup_iters)
      coeff = 0.5 * (1.0 + Math.cos(Math::PI * decay_ratio))
      @min_lr + coeff * (@learning_rate - @min_lr)
    end

    # Apply learning rate to optimizer
    def step(optimizer, iter)
      lr = get_lr(iter)
      optimizer.param_groups.each do |group|
        group[:lr] = lr
      end
      lr
    end
  end
end
