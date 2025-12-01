# frozen_string_literal: true

require "json"

module NanoGPT
  # Character-level tokenizer
  class Tokenizer
    attr_reader :vocab_size, :stoi, :itos

    def initialize(stoi: nil, itos: nil)
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
end
