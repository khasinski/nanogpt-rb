# frozen_string_literal: true

module NanoGPT
  # Configuration for GPT model architecture
  class GPTConfig
    attr_accessor :block_size, :vocab_size, :n_layer, :n_head, :n_embd, :dropout, :bias

    def initialize(
      block_size: 1024,
      vocab_size: 50304,
      n_layer: 12,
      n_head: 12,
      n_embd: 768,
      dropout: 0.0,
      bias: true
    )
      @block_size = block_size
      @vocab_size = vocab_size
      @n_layer = n_layer
      @n_head = n_head
      @n_embd = n_embd
      @dropout = dropout
      @bias = bias
    end

    def to_h
      {
        block_size: @block_size,
        vocab_size: @vocab_size,
        n_layer: @n_layer,
        n_head: @n_head,
        n_embd: @n_embd,
        dropout: @dropout,
        bias: @bias
      }
    end

    def head_size
      @n_embd / @n_head
    end
  end
end
