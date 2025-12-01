# frozen_string_literal: true

require "spec_helper"
require "tmpdir"

RSpec.describe NanoGPT::TrainConfig do
  describe "DEFAULTS" do
    it "has expected default values" do
      expect(NanoGPT::TrainConfig::DEFAULTS[:out_dir]).to eq("out-shakespeare-char")
      expect(NanoGPT::TrainConfig::DEFAULTS[:dataset]).to eq("shakespeare_char")
      expect(NanoGPT::TrainConfig::DEFAULTS[:batch_size]).to eq(64)
      expect(NanoGPT::TrainConfig::DEFAULTS[:block_size]).to eq(256)
      expect(NanoGPT::TrainConfig::DEFAULTS[:n_layer]).to eq(6)
      expect(NanoGPT::TrainConfig::DEFAULTS[:n_head]).to eq(6)
      expect(NanoGPT::TrainConfig::DEFAULTS[:n_embd]).to eq(384)
      expect(NanoGPT::TrainConfig::DEFAULTS[:learning_rate]).to eq(1e-3)
      expect(NanoGPT::TrainConfig::DEFAULTS[:max_iters]).to eq(5000)
      expect(NanoGPT::TrainConfig::DEFAULTS[:device]).to eq("auto")
    end
  end

  describe "#initialize" do
    it "uses default values" do
      config = NanoGPT::TrainConfig.new
      expect(config[:batch_size]).to eq(64)
      expect(config[:learning_rate]).to eq(1e-3)
    end

    it "merges custom values with defaults" do
      config = NanoGPT::TrainConfig.new(batch_size: 32, learning_rate: 5e-4)
      expect(config[:batch_size]).to eq(32)
      expect(config[:learning_rate]).to eq(5e-4)
      expect(config[:block_size]).to eq(256) # default preserved
    end
  end

  describe "#[] and #[]=" do
    it "accesses values with symbol or string keys" do
      config = NanoGPT::TrainConfig.new
      expect(config[:batch_size]).to eq(64)
      expect(config["batch_size"]).to eq(64)
    end

    it "sets values" do
      config = NanoGPT::TrainConfig.new
      config[:batch_size] = 128
      expect(config[:batch_size]).to eq(128)
    end
  end

  describe "#to_h" do
    it "returns a hash copy of values" do
      config = NanoGPT::TrainConfig.new
      hash = config.to_h
      expect(hash).to be_a(Hash)
      expect(hash[:batch_size]).to eq(64)

      # Verify it's a copy
      hash[:batch_size] = 999
      expect(config[:batch_size]).to eq(64)
    end
  end

  describe ".load" do
    it "parses command-line integer overrides" do
      args = ["--batch_size=32", "--max_iters=100"]
      config = NanoGPT::TrainConfig.load(args)
      expect(config[:batch_size]).to eq(32)
      expect(config[:max_iters]).to eq(100)
    end

    it "parses command-line float overrides" do
      args = ["--learning_rate=0.0005", "--dropout=0.3"]
      config = NanoGPT::TrainConfig.load(args)
      expect(config[:learning_rate]).to eq(0.0005)
      expect(config[:dropout]).to eq(0.3)
    end

    it "parses command-line boolean overrides" do
      args = ["--bias=true", "--decay_lr=false"]
      config = NanoGPT::TrainConfig.load(args)
      expect(config[:bias]).to eq(true)
      expect(config[:decay_lr]).to eq(false)
    end

    it "parses command-line string overrides" do
      args = ["--dataset=openwebtext", "--out_dir=out-custom"]
      config = NanoGPT::TrainConfig.load(args)
      expect(config[:dataset]).to eq("openwebtext")
      expect(config[:out_dir]).to eq("out-custom")
    end

    it "warns about unknown keys" do
      args = ["--unknown_key=value"]
      expect { NanoGPT::TrainConfig.load(args) }.to output(/Warning: Unknown config key: unknown_key/).to_stdout
    end

    it "ignores malformed arguments" do
      args = ["--no-equals", "positional", "--valid=123"]
      config = NanoGPT::TrainConfig.load(args)
      # Only valid arg should be applied
      expect(config[:batch_size]).to eq(64) # default unchanged
    end
  end

  describe "#load_json" do
    let(:tmp_dir) { Dir.mktmpdir }
    let(:config_path) { File.join(tmp_dir, "config.json") }

    after { FileUtils.rm_rf(tmp_dir) }

    it "loads values from JSON file" do
      File.write(config_path, '{"batch_size": 128, "learning_rate": 0.0001}')

      config = NanoGPT::TrainConfig.new
      config.load_json(config_path)

      expect(config[:batch_size]).to eq(128)
      expect(config[:learning_rate]).to eq(0.0001)
    end

    it "preserves defaults for keys not in JSON" do
      File.write(config_path, '{"batch_size": 128}')

      config = NanoGPT::TrainConfig.new
      config.load_json(config_path)

      expect(config[:batch_size]).to eq(128)
      expect(config[:learning_rate]).to eq(1e-3) # default
    end

    it "raises error for missing file" do
      config = NanoGPT::TrainConfig.new
      expect { config.load_json("/nonexistent/path.json") }.to raise_error(/Config file not found/)
    end

    it "warns about unknown keys in JSON" do
      File.write(config_path, '{"batch_size": 128, "unknown_json_key": 42}')

      config = NanoGPT::TrainConfig.new
      expect { config.load_json(config_path) }.to output(/Warning: Unknown config key in JSON: unknown_json_key/).to_stdout
    end
  end

  describe ".load with --config" do
    let(:tmp_dir) { Dir.mktmpdir }
    let(:config_path) { File.join(tmp_dir, "config.json") }

    after { FileUtils.rm_rf(tmp_dir) }

    it "loads JSON config and applies CLI overrides" do
      File.write(config_path, '{"batch_size": 128, "learning_rate": 0.0001}')

      args = ["--config=#{config_path}", "--batch_size=256"]
      config = NanoGPT::TrainConfig.load(args)

      # CLI override wins
      expect(config[:batch_size]).to eq(256)
      # JSON value preserved
      expect(config[:learning_rate]).to eq(0.0001)
    end

    it "applies CLI overrides after JSON" do
      File.write(config_path, '{"max_iters": 1000}')

      args = ["--max_iters=500", "--config=#{config_path}"]
      config = NanoGPT::TrainConfig.load(args)

      # CLI args are processed after JSON regardless of order
      expect(config[:max_iters]).to eq(500)
    end
  end

  describe "#save_json" do
    let(:tmp_dir) { Dir.mktmpdir }
    let(:config_path) { File.join(tmp_dir, "saved.json") }

    after { FileUtils.rm_rf(tmp_dir) }

    it "saves config to JSON file" do
      config = NanoGPT::TrainConfig.new(batch_size: 256)
      config.save_json(config_path)

      expect(File.exist?(config_path)).to be true
      loaded = JSON.parse(File.read(config_path))
      expect(loaded["batch_size"]).to eq(256)
    end
  end
end

RSpec.describe NanoGPT::SampleConfig do
  describe "DEFAULTS" do
    it "has expected default values" do
      expect(NanoGPT::SampleConfig::DEFAULTS[:out_dir]).to eq("out-shakespeare-char")
      expect(NanoGPT::SampleConfig::DEFAULTS[:dataset]).to eq("shakespeare_char")
      expect(NanoGPT::SampleConfig::DEFAULTS[:start]).to eq("\n")
      expect(NanoGPT::SampleConfig::DEFAULTS[:num_samples]).to eq(5)
      expect(NanoGPT::SampleConfig::DEFAULTS[:max_new_tokens]).to eq(500)
      expect(NanoGPT::SampleConfig::DEFAULTS[:temperature]).to eq(0.8)
      expect(NanoGPT::SampleConfig::DEFAULTS[:top_k]).to eq(200)
      expect(NanoGPT::SampleConfig::DEFAULTS[:seed]).to eq(1337)
      expect(NanoGPT::SampleConfig::DEFAULTS[:device]).to eq("auto")
    end
  end

  describe "#initialize" do
    it "uses default values" do
      config = NanoGPT::SampleConfig.new
      expect(config[:num_samples]).to eq(5)
      expect(config[:temperature]).to eq(0.8)
    end

    it "merges custom values with defaults" do
      config = NanoGPT::SampleConfig.new(num_samples: 10, temperature: 0.5)
      expect(config[:num_samples]).to eq(10)
      expect(config[:temperature]).to eq(0.5)
      expect(config[:max_new_tokens]).to eq(500) # default preserved
    end
  end

  describe ".load" do
    it "parses command-line overrides" do
      args = ["--num_samples=3", "--temperature=0.7", "--max_new_tokens=200"]
      config = NanoGPT::SampleConfig.load(args)
      expect(config[:num_samples]).to eq(3)
      expect(config[:temperature]).to eq(0.7)
      expect(config[:max_new_tokens]).to eq(200)
    end

    it "parses string overrides" do
      args = ["--start=Hello world", "--out_dir=custom-out"]
      config = NanoGPT::SampleConfig.load(args)
      expect(config[:start]).to eq("Hello world")
      expect(config[:out_dir]).to eq("custom-out")
    end
  end

  describe "#load_json" do
    let(:tmp_dir) { Dir.mktmpdir }
    let(:config_path) { File.join(tmp_dir, "sample_config.json") }

    after { FileUtils.rm_rf(tmp_dir) }

    it "loads values from JSON file" do
      File.write(config_path, '{"num_samples": 10, "temperature": 0.5}')

      config = NanoGPT::SampleConfig.new
      config.load_json(config_path)

      expect(config[:num_samples]).to eq(10)
      expect(config[:temperature]).to eq(0.5)
    end
  end
end
