# frozen_string_literal: true

module NanoGPT
  module Layers
    # Transformer block: LayerNorm -> Attention -> LayerNorm -> MLP
    class Block < Torch::NN::Module
      def initialize(config)
        super()
        @ln_1 = LayerNorm.new(config.n_embd, bias: config.bias)
        @attn = CausalSelfAttention.new(config)
        @ln_2 = LayerNorm.new(config.n_embd, bias: config.bias)
        @mlp = MLP.new(config)
      end

      def forward(x)
        x = x + @attn.call(@ln_1.call(x))
        x = x + @mlp.call(@ln_2.call(x))
        # Trigger GC to free intermediate tensors (critical for torch.rb memory management)
        # Ruby's GC doesn't run frequently enough during forward pass, causing memory accumulation
        GC.start(full_mark: false, immediate_sweep: true)
        x
      end
    end
  end
end
