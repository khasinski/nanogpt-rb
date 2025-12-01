# frozen_string_literal: true

require "json"

module NanoGPT
  # Configuration system for training and sampling
  # Supports JSON config files with command-line overrides
  #
  # Priority (highest to lowest):
  #   1. Command-line arguments (--key=value)
  #   2. JSON config file (--config=path.json)
  #   3. Default values
  #
  # Usage:
  #   config = TrainConfig.load(ARGV)
  #   config[:learning_rate]  # => 0.001
  #
  class TrainConfig
    # Defaults match bin/train exactly
    DEFAULTS = {
      # I/O
      out_dir: "out-shakespeare-char",
      eval_interval: 250,
      log_interval: 10,
      eval_iters: 200,
      eval_only: false,
      always_save_checkpoint: false,
      init_from: "scratch",  # 'scratch' or 'resume'

      # Data
      dataset: "shakespeare_char",
      batch_size: 64,
      block_size: 256,
      gradient_accumulation_steps: 1,

      # Model
      n_layer: 6,
      n_head: 6,
      n_embd: 384,
      dropout: 0.2,
      bias: false,

      # Optimizer
      learning_rate: 1e-3,
      weight_decay: 1e-1,
      beta1: 0.9,
      beta2: 0.99,
      grad_clip: 1.0,

      # LR scheduler
      decay_lr: true,
      warmup_iters: 100,
      lr_decay_iters: 5000,
      min_lr: 1e-4,

      # Training
      max_iters: 5000,

      # System
      device: "auto"
    }.freeze

    attr_reader :values

    def initialize(values = {})
      @values = DEFAULTS.merge(values)
    end

    def [](key)
      @values[key.to_sym]
    end

    def []=(key, value)
      @values[key.to_sym] = value
    end

    def to_h
      @values.dup
    end

    # Load config from command-line args
    # Supports:
    #   --config=path/to/config.json  (load JSON file)
    #   --key=value                   (override specific values)
    def self.load(args)
      config = new

      # First pass: find and load JSON config file
      config_file = nil
      args.each do |arg|
        if arg.start_with?("--config=")
          config_file = arg.split("=", 2).last
          break
        end
      end

      if config_file
        config.load_json(config_file)
      end

      # Second pass: apply command-line overrides
      args.each do |arg|
        next unless arg.start_with?("--") && arg.include?("=")
        next if arg.start_with?("--config=")

        key, val = arg[2..].split("=", 2)
        key = key.to_sym

        unless config.values.key?(key)
          puts "Warning: Unknown config key: #{key}"
          next
        end

        config[key] = parse_value(val, config[key])
        puts "Override: #{key} = #{config[key]}"
      end

      config
    end

    # Load values from JSON file
    def load_json(path)
      unless File.exist?(path)
        raise "Config file not found: #{path}"
      end

      json = JSON.parse(File.read(path))
      puts "Loaded config from #{path}"

      json.each do |key, val|
        key = key.to_sym
        unless @values.key?(key)
          puts "Warning: Unknown config key in JSON: #{key}"
          next
        end
        @values[key] = val
      end

      self
    end

    # Save current config to JSON file
    def save_json(path)
      File.write(path, JSON.pretty_generate(@values))
      puts "Saved config to #{path}"
    end

    private

    def self.parse_value(val, existing)
      case existing
      when Integer then val.to_i
      when Float then val.to_f
      when TrueClass, FalseClass then val.downcase == "true"
      else val
      end
    end
  end

  # Configuration for sampling/generation
  class SampleConfig
    # Defaults match bin/sample exactly
    DEFAULTS = {
      out_dir: "out-shakespeare-char",
      dataset: "shakespeare_char",
      start: "\n",
      num_samples: 5,
      max_new_tokens: 500,
      temperature: 0.8,
      top_k: 200,
      seed: 1337,
      device: "auto"
    }.freeze

    attr_reader :values

    def initialize(values = {})
      @values = DEFAULTS.merge(values)
    end

    def [](key)
      @values[key.to_sym]
    end

    def []=(key, value)
      @values[key.to_sym] = value
    end

    def to_h
      @values.dup
    end

    def self.load(args)
      config = new

      # First pass: find and load JSON config file
      config_file = nil
      args.each do |arg|
        if arg.start_with?("--config=")
          config_file = arg.split("=", 2).last
          break
        end
      end

      if config_file
        config.load_json(config_file)
      end

      # Second pass: apply command-line overrides
      args.each do |arg|
        next unless arg.start_with?("--") && arg.include?("=")
        next if arg.start_with?("--config=")

        key, val = arg[2..].split("=", 2)
        key = key.to_sym

        unless config.values.key?(key)
          puts "Warning: Unknown config key: #{key}"
          next
        end

        config[key] = parse_value(val, config[key])
      end

      config
    end

    def load_json(path)
      unless File.exist?(path)
        raise "Config file not found: #{path}"
      end

      json = JSON.parse(File.read(path))
      puts "Loaded config from #{path}"

      json.each do |key, val|
        key = key.to_sym
        unless @values.key?(key)
          puts "Warning: Unknown config key in JSON: #{key}"
          next
        end
        @values[key] = val
      end

      self
    end

    private

    def self.parse_value(val, existing)
      case existing
      when Integer then val.to_i
      when Float then val.to_f
      when TrueClass, FalseClass then val.downcase == "true"
      else val
      end
    end
  end
end
