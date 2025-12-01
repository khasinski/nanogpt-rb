# frozen_string_literal: true

module NanoGPT
  module Layers
    # Multi-head causal self-attention
    class CausalSelfAttention < Torch::NN::Module
      def initialize(config)
        super()
        raise ArgumentError, "n_embd must be divisible by n_head" unless (config.n_embd % config.n_head).zero?

        @n_head = config.n_head
        @n_embd = config.n_embd
        @head_size = config.n_embd / config.n_head
        @dropout_p = config.dropout

        # Key, query, value projections for all heads, combined
        @c_attn = Torch::NN::Linear.new(config.n_embd, 3 * config.n_embd, bias: config.bias)
        # Output projection
        @c_proj = Torch::NN::Linear.new(config.n_embd, config.n_embd, bias: config.bias)
        # Regularization
        @attn_dropout = Torch::NN::Dropout.new(p: config.dropout)
        @resid_dropout = Torch::NN::Dropout.new(p: config.dropout)

        # torch.rb doesn't fully support is_causal in scaled_dot_product_attention
        # so we always use manual attention with causal mask
        @flash = false

        # Causal mask for manual attention
        mask = Torch.tril(Torch.ones(config.block_size, config.block_size))
        register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
      end

      def forward(x)
        b, t, c = x.shape

        # Calculate Q, K, V
        qkv = @c_attn.call(x)
        q, k, v = qkv.split(@n_embd, 2)

        # Reshape: (B, T, C) -> (B, nh, T, hs)
        q = q.view(b, t, @n_head, @head_size).transpose(1, 2)
        k = k.view(b, t, @n_head, @head_size).transpose(1, 2)
        v = v.view(b, t, @n_head, @head_size).transpose(1, 2)

        # Manual attention implementation with causal mask
        # Use in-place operations to reduce memory usage (important for torch.rb)
        scale = 1.0 / Math.sqrt(@head_size)
        att = q.matmul(k.transpose(-2, -1))
        att.mul!(scale)  # In-place scale

        # Apply causal mask - slice mask to current sequence length
        # Using narrow to slice: (1, 1, block_size, block_size) -> (1, 1, t, t)
        mask_slice = @mask.narrow(2, 0, t).narrow(3, 0, t)
        att.masked_fill!(mask_slice.eq(0), -Float::INFINITY)  # In-place mask
        att = Torch::NN::Functional.softmax(att, dim: -1)
        att = @attn_dropout.call(att)
        y = att.matmul(v)

        # Reassemble heads: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous.view(b, t, c)

        # Output projection
        @resid_dropout.call(@c_proj.call(y))
      end
    end
  end
end
