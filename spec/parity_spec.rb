# frozen_string_literal: true

# Parity test to verify Ruby implementation matches Python exactly
# Run: bundle exec rspec spec/parity_spec.rb

require "spec_helper"
require "tmpdir"
require "open3"

RSpec.describe "Python/Ruby Parity" do
  let(:weights_dir) { Dir.mktmpdir }
  let(:config) do
    {
      block_size: 32,
      vocab_size: 65,
      n_layer: 2,
      n_head: 2,
      n_embd: 64,
      dropout: 0.0,
      bias: false
    }
  end

  after { FileUtils.rm_rf(weights_dir) }

  # Helper to load numpy files
  def load_npy(path)
    data = File.binread(path)
    magic = data[0..5].bytes
    raise "Not a numpy file" unless magic == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]

    major = data[6].ord
    header_len = major == 1 ? data[8..9].unpack1("S<") : data[8..11].unpack1("L<")
    header_start = major == 1 ? 10 : 12
    header = data[header_start...(header_start + header_len)]

    shape = header.match(/shape':\s*\(([^)]*)\)/)[1].split(",").map(&:strip).reject(&:empty?).map(&:to_i)
    shape = [1] if shape.empty?
    dtype_str = header.match(/descr':\s*'([^']*)'/)[1]

    raw = data[(header_start + header_len)..]
    case dtype_str
    when "<f4" then Torch.tensor(raw.unpack("e*"), dtype: :float32).reshape(shape)
    when "<i8" then Torch.tensor(raw.unpack("q<*"), dtype: :long).reshape(shape)
    else raise "Unknown dtype: #{dtype_str}"
    end
  end

  def create_python_baseline
    python_script = <<~PYTHON
      import torch
      import numpy as np
      import sys
      import os
      sys.path.insert(0, '#{File.expand_path("../python", __dir__)}')
      from model import GPT, GPTConfig

      torch.manual_seed(1337)
      config = GPTConfig(block_size=32, vocab_size=65, n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=False)
      model = GPT(config)

      # Save weights
      os.makedirs('#{weights_dir}', exist_ok=True)
      state = model.state_dict()
      for key, tensor in state.items():
          np.save(f'#{weights_dir}/{key.replace(".", "_")}.npy', tensor.numpy())

      # Save test input
      torch.manual_seed(42)
      x = torch.randint(0, 65, (2, 16))
      y = torch.randint(0, 65, (2, 16))
      np.save('#{weights_dir}/x.npy', x.numpy())
      np.save('#{weights_dir}/y.npy', y.numpy())

      # Run forward, backward, step
      model.train()
      model.zero_grad()
      logits, loss = model(x, y)
      loss.backward()
      optimizer = model.configure_optimizers(0.1, 1e-3, (0.9, 0.99), 'cpu')
      optimizer.step()

      # Save results
      np.save('#{weights_dir}/py_loss.npy', np.array([loss.item()], dtype=np.float32))
      np.save('#{weights_dir}/py_logits_sum.npy', np.array([logits.sum().item()], dtype=np.float32))
      np.save('#{weights_dir}/py_grad_sum.npy', np.array([model.transformer.wte.weight.grad.sum().item()], dtype=np.float32))
      np.save('#{weights_dir}/py_weight_sum.npy', np.array([model.transformer.wte.weight.sum().item()], dtype=np.float32))
    PYTHON

    stdout, stderr, status = Open3.capture3("python", "-c", python_script)
    raise "Python failed: #{stderr}" unless status.success?
  end

  def load_weights_into_model(model)
    key_mapping = {
      "transformer_wte_weight" => ["@wte", "weight"],
      "transformer_wpe_weight" => ["@wpe", "weight"],
      "transformer_h_0_ln_1_weight" => ["@h", 0, "@ln_1", "weight"],
      "transformer_h_0_attn_c_attn_weight" => ["@h", 0, "@attn", "@c_attn", "weight"],
      "transformer_h_0_attn_c_proj_weight" => ["@h", 0, "@attn", "@c_proj", "weight"],
      "transformer_h_0_ln_2_weight" => ["@h", 0, "@ln_2", "weight"],
      "transformer_h_0_mlp_c_fc_weight" => ["@h", 0, "@mlp", "@c_fc", "weight"],
      "transformer_h_0_mlp_c_proj_weight" => ["@h", 0, "@mlp", "@c_proj", "weight"],
      "transformer_h_1_ln_1_weight" => ["@h", 1, "@ln_1", "weight"],
      "transformer_h_1_attn_c_attn_weight" => ["@h", 1, "@attn", "@c_attn", "weight"],
      "transformer_h_1_attn_c_proj_weight" => ["@h", 1, "@attn", "@c_proj", "weight"],
      "transformer_h_1_ln_2_weight" => ["@h", 1, "@ln_2", "weight"],
      "transformer_h_1_mlp_c_fc_weight" => ["@h", 1, "@mlp", "@c_fc", "weight"],
      "transformer_h_1_mlp_c_proj_weight" => ["@h", 1, "@mlp", "@c_proj", "weight"],
      "transformer_ln_f_weight" => ["@ln_f", "weight"]
    }

    key_mapping.each do |file_key, path|
      tensor = load_npy("#{weights_dir}/#{file_key}.npy")
      obj = model
      path[0...-1].each { |s| obj = s.is_a?(Integer) ? obj[s] : obj.instance_variable_get(s) }
      Torch.no_grad { obj.weight.copy!(tensor) }
    end
  end

  let(:python_model_path) { File.expand_path("../python/model.py", __dir__) }

  it "produces identical results to Python" do
    # Skip if Python/nanoGPT not available
    skip "Python nanoGPT not found" unless File.exist?(python_model_path)

    # Create Python baseline
    create_python_baseline

    # Load Python results
    py_loss = load_npy("#{weights_dir}/py_loss.npy").item
    py_logits_sum = load_npy("#{weights_dir}/py_logits_sum.npy").item
    py_grad_sum = load_npy("#{weights_dir}/py_grad_sum.npy").item
    py_weight_sum = load_npy("#{weights_dir}/py_weight_sum.npy").item

    # Create Ruby model with same weights
    model_config = NanoGPT::GPTConfig.new(**config)
    model = NanoGPT::GPT.new(model_config)
    load_weights_into_model(model)

    # Load test input
    x = load_npy("#{weights_dir}/x.npy")
    y = load_npy("#{weights_dir}/y.npy")

    # Run forward, backward, step
    model.train
    model.parameters.each { |p| p.grad&.zero! }
    logits, loss = model.call(x, targets: y)
    loss.backward

    optimizer = model.configure_optimizers(
      weight_decay: 0.1, learning_rate: 1e-3, betas: [0.9, 0.99], device_type: "cpu"
    )
    optimizer.step

    wte = model.instance_variable_get(:@wte)
    rb_loss = loss.item
    rb_logits_sum = logits.sum.item
    rb_grad_sum = wte.weight.grad.sum.item
    rb_weight_sum = wte.weight.sum.item

    # Compare with tolerance for floating point differences
    expect(rb_loss).to be_within(1e-6).of(py_loss)
    expect(rb_logits_sum).to be_within(1e-4).of(py_logits_sum)
    expect(rb_grad_sum).to be_within(1e-6).of(py_grad_sum)
    expect(rb_weight_sum).to be_within(1e-5).of(py_weight_sum)
  end
end
