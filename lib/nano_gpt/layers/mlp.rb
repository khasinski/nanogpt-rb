# frozen_string_literal: true

module NanoGPT
  module Layers
    # Feed-forward network with GELU activation
    class MLP < Torch::NN::Module
      def initialize(config)
        super()
        @c_fc = Torch::NN::Linear.new(config.n_embd, 4 * config.n_embd, bias: config.bias)
        @gelu = Torch::NN::GELU.new
        @c_proj = Torch::NN::Linear.new(4 * config.n_embd, config.n_embd, bias: config.bias)
        @dropout = Torch::NN::Dropout.new(p: config.dropout)
      end

      def forward(x)
        x = @c_fc.call(x)
        x = @gelu.call(x)
        x = @c_proj.call(x)
        @dropout.call(x)
      end
    end
  end
end
