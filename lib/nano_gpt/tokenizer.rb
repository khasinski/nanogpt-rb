# frozen_string_literal: true

require "json"

module NanoGPT
  # Base tokenizer interface
  class Tokenizer
    attr_reader :vocab_size

    def encode(text)
      raise NotImplementedError
    end

    def decode(ids)
      raise NotImplementedError
    end

    # Auto-detect and load the appropriate tokenizer
    # If meta.json exists, use character-level; otherwise use GPT-2 BPE
    def self.for_dataset(dataset_dir)
      meta_path = File.join(dataset_dir, "meta.json")
      if File.exist?(meta_path)
        CharTokenizer.from_file(meta_path)
      else
        GPT2Tokenizer.new
      end
    end
  end

  # Character-level tokenizer
  class CharTokenizer < Tokenizer
    attr_reader :stoi, :itos

    def initialize(stoi: nil, itos: nil)
      super()
      @stoi = stoi || {}
      @itos = itos || {}
      @vocab_size = @stoi.size
    end

    # Build vocabulary from text
    def self.from_text(text)
      chars = text.chars.uniq.sort
      stoi = chars.each_with_index.to_h
      itos = chars.each_with_index.map { |c, i| [i, c] }.to_h
      new(stoi: stoi, itos: itos)
    end

    # Load from meta.json file
    def self.from_file(path)
      meta = JSON.parse(File.read(path))
      # Convert string keys to integers for itos
      itos = meta["itos"].transform_keys(&:to_i)
      new(stoi: meta["stoi"], itos: itos)
    end

    # Encode string to list of integers
    def encode(text)
      text.chars.map { |c| @stoi[c] }
    end

    # Decode list of integers to string
    def decode(ids)
      ids.map { |i| @itos[i] }.join
    end

    # Save to meta.json file
    def save(path)
      meta = {
        "vocab_size" => @vocab_size,
        "stoi" => @stoi,
        "itos" => @itos.transform_keys(&:to_s)
      }
      File.write(path, JSON.pretty_generate(meta))
    end
  end

  # GPT-2 BPE tokenizer using tiktoken
  class GPT2Tokenizer < Tokenizer
    GPT2_VOCAB_SIZE = 50257
    EOT_TOKEN = "<|endoftext|>"

    def initialize
      super()
      require "tiktoken_ruby"
      # GPT-2 uses the r50k_base encoding
      @enc = Tiktoken.get_encoding(:r50k_base)
      @vocab_size = GPT2_VOCAB_SIZE
    end

    # Encode string to list of integers
    def encode(text)
      @enc.encode(text)
    end

    # Decode list of integers to string
    def decode(ids)
      @enc.decode(ids)
    end

    # Get the end-of-text token ID
    def eot_token
      @enc.encode(EOT_TOKEN).first
    end
  end
end
