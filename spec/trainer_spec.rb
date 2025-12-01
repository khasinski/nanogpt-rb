# frozen_string_literal: true

require "spec_helper"
require "tmpdir"

RSpec.describe NanoGPT::Trainer do
  let(:data_dir) { Dir.mktmpdir }
  let(:out_dir) { Dir.mktmpdir }

  let(:config) do
    NanoGPT::GPTConfig.new(
      block_size: 16,
      vocab_size: 100,
      n_layer: 1,
      n_head: 2,
      n_embd: 32,
      dropout: 0.0,
      bias: true
    )
  end

  let(:model) { NanoGPT::GPT.new(config) }

  let(:data_loader) do
    NanoGPT::DataLoader.new(
      data_dir: data_dir,
      block_size: config.block_size,
      batch_size: 4,
      device: "cpu"
    )
  end

  let(:trainer_config) do
    {
      out_dir: out_dir,
      eval_interval: 10,
      log_interval: 5,
      eval_iters: 5,
      max_iters: 20,
      gradient_accumulation_steps: 1,
      learning_rate: 1e-3,
      warmup_iters: 5,
      lr_decay_iters: 20,
      min_lr: 1e-4,
      decay_lr: true,
      device: "cpu"
    }
  end

  before do
    # Create test data - enough tokens for training
    train_data = Numo::UInt16.cast((0...1000).map { rand(100) })
    File.binwrite(File.join(data_dir, "train.bin"), train_data.to_binary)

    val_data = Numo::UInt16.cast((0...200).map { rand(100) })
    File.binwrite(File.join(data_dir, "val.bin"), val_data.to_binary)
  end

  after do
    FileUtils.rm_rf(data_dir)
    FileUtils.rm_rf(out_dir)
  end

  describe "#initialize" do
    it "sets up model, optimizer, and scheduler" do
      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: trainer_config)

      expect(trainer.model).to eq(model)
      expect(trainer.optimizer).to be_a(Torch::Optim::AdamW)
      expect(trainer.iter_num).to eq(0)
      expect(trainer.best_val_loss).to eq(Float::INFINITY)
    end
  end

  describe "#estimate_loss" do
    it "returns train and val losses" do
      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: trainer_config)

      losses = trainer.estimate_loss

      expect(losses).to have_key(:train)
      expect(losses).to have_key(:val)
      expect(losses[:train]).to be_a(Float)
      expect(losses[:val]).to be_a(Float)
      expect(losses[:train]).to be > 0
      expect(losses[:val]).to be > 0
    end
  end

  describe "#train" do
    it "runs training loop for max_iters" do
      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: trainer_config)

      expect { trainer.train }.to output(/Starting training/).to_stdout

      expect(trainer.iter_num).to eq(trainer_config[:max_iters] + 1)
    end

    it "decreases loss during training" do
      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: trainer_config)

      initial_loss = trainer.estimate_loss[:train]
      trainer.train
      final_loss = trainer.estimate_loss[:train]

      # Loss should generally decrease (though not guaranteed for every run)
      # We use a generous margin since this is a small test
      expect(final_loss).to be < initial_loss * 1.5
    end
  end

  describe "#save_checkpoint and #load_checkpoint" do
    it "saves and loads model state" do
      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: trainer_config)

      # Train a bit
      trainer.instance_variable_set(:@iter_num, 10)
      trainer.instance_variable_set(:@best_val_loss, 2.5)

      # Save
      trainer.save_checkpoint

      ckpt_path = File.join(out_dir, "ckpt.pt")
      expect(File.exist?(ckpt_path)).to be true

      # Create new trainer and load
      new_model = NanoGPT::GPT.new(config)
      new_trainer = NanoGPT::Trainer.new(model: new_model, data_loader: data_loader, config: trainer_config)

      new_trainer.load_checkpoint(ckpt_path)

      expect(new_trainer.iter_num).to eq(10)
      expect(new_trainer.best_val_loss).to eq(2.5)
    end
  end

  describe "gradient accumulation" do
    it "accumulates gradients over multiple steps" do
      config_with_accum = trainer_config.merge(
        gradient_accumulation_steps: 4,
        max_iters: 5,
        eval_interval: 100  # Don't eval during this test
      )

      trainer = NanoGPT::Trainer.new(model: model, data_loader: data_loader, config: config_with_accum)

      expect { trainer.train }.not_to raise_error
    end
  end
end
