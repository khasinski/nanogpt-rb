# frozen_string_literal: true

require "spec_helper"
require "tempfile"

RSpec.describe NanoGPT::Tokenizer do
  describe ".from_text" do
    it "builds vocabulary from text" do
      text = "hello world"
      tokenizer = NanoGPT::Tokenizer.from_text(text)

      expect(tokenizer.vocab_size).to eq(8) # ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'
      expect(tokenizer.stoi.keys.sort).to eq([" ", "d", "e", "h", "l", "o", "r", "w"])
    end

    it "creates sorted vocabulary" do
      text = "cba"
      tokenizer = NanoGPT::Tokenizer.from_text(text)

      expect(tokenizer.stoi["a"]).to eq(0)
      expect(tokenizer.stoi["b"]).to eq(1)
      expect(tokenizer.stoi["c"]).to eq(2)
    end
  end

  describe "#encode" do
    it "converts string to integer list" do
      tokenizer = NanoGPT::Tokenizer.from_text("abc")

      expect(tokenizer.encode("abc")).to eq([0, 1, 2])
      expect(tokenizer.encode("cab")).to eq([2, 0, 1])
      expect(tokenizer.encode("aaa")).to eq([0, 0, 0])
    end
  end

  describe "#decode" do
    it "converts integer list to string" do
      tokenizer = NanoGPT::Tokenizer.from_text("abc")

      expect(tokenizer.decode([0, 1, 2])).to eq("abc")
      expect(tokenizer.decode([2, 0, 1])).to eq("cab")
      expect(tokenizer.decode([0, 0, 0])).to eq("aaa")
    end
  end

  describe "encode/decode roundtrip" do
    it "preserves original text" do
      text = "Hello, World! 123"
      tokenizer = NanoGPT::Tokenizer.from_text(text)

      encoded = tokenizer.encode(text)
      decoded = tokenizer.decode(encoded)

      expect(decoded).to eq(text)
    end
  end

  describe "#save and .from_file" do
    it "saves and loads tokenizer" do
      tokenizer = NanoGPT::Tokenizer.from_text("hello")

      Tempfile.create(["meta", ".json"]) do |f|
        tokenizer.save(f.path)

        loaded = NanoGPT::Tokenizer.from_file(f.path)

        expect(loaded.vocab_size).to eq(tokenizer.vocab_size)
        expect(loaded.stoi).to eq(tokenizer.stoi)
        expect(loaded.encode("hello")).to eq(tokenizer.encode("hello"))
      end
    end
  end
end
