# frozen_string_literal: true

module NanoGPT
  module Layers
    # LayerNorm with optional bias (PyTorch doesn't support bias=false directly)
    class LayerNorm < Torch::NN::Module
      attr_reader :weight, :bias

      def initialize(ndim, bias: true)
        super()
        @ndim = ndim
        @weight = Torch::NN::Parameter.new(Torch.ones(ndim))
        @bias = bias ? Torch::NN::Parameter.new(Torch.zeros(ndim)) : nil
      end

      def forward(input)
        Torch::NN::Functional.layer_norm(input, [@ndim], weight: @weight, bias: @bias, eps: 1e-5)
      end
    end
  end
end
