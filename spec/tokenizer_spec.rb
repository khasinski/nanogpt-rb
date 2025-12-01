# frozen_string_literal: true

require "spec_helper"

RSpec.describe "Tokenizers" do
  describe NanoGPT::CharTokenizer do
    let(:text) { "hello world" }
    let(:tokenizer) { NanoGPT::CharTokenizer.from_text(text) }

    it "builds vocabulary from text" do
      expect(tokenizer.vocab_size).to eq(8) # ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
    end

    it "encodes and decodes text" do
      encoded = tokenizer.encode(text)
      decoded = tokenizer.decode(encoded)

      expect(decoded).to eq(text)
    end

    it "saves and loads from file" do
      require "tempfile"

      Tempfile.create(["meta", ".json"]) do |f|
        tokenizer.save(f.path)
        loaded = NanoGPT::CharTokenizer.from_file(f.path)

        expect(loaded.vocab_size).to eq(tokenizer.vocab_size)
        expect(loaded.encode(text)).to eq(tokenizer.encode(text))
      end
    end
  end

  describe NanoGPT::GPT2Tokenizer do
    let(:tokenizer) { NanoGPT::GPT2Tokenizer.new }

    it "has GPT-2 vocabulary size" do
      expect(tokenizer.vocab_size).to eq(50257)
    end

    it "encodes and decodes text" do
      text = "Hello, world!"
      encoded = tokenizer.encode(text)
      decoded = tokenizer.decode(encoded)

      expect(decoded).to eq(text)
      expect(encoded).to be_an(Array)
      expect(encoded.first).to be_an(Integer)
    end

    it "provides end-of-text token" do
      eot = tokenizer.eot_token
      expect(eot).to be_an(Integer)
    end
  end

  describe NanoGPT::Tokenizer do
    it "auto-detects character-level tokenizer when meta.json exists" do
      tokenizer = NanoGPT::Tokenizer.for_dataset("data/shakespeare_char")

      expect(tokenizer).to be_a(NanoGPT::CharTokenizer)
      expect(tokenizer.vocab_size).to eq(65)
    end

    it "falls back to GPT-2 tokenizer when no meta.json" do
      require "tmpdir"

      Dir.mktmpdir do |dir|
        tokenizer = NanoGPT::Tokenizer.for_dataset(dir)
        expect(tokenizer).to be_a(NanoGPT::GPT2Tokenizer)
      end
    end
  end
end
