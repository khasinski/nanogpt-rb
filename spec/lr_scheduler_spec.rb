# frozen_string_literal: true

require "spec_helper"

RSpec.describe NanoGPT::LRScheduler do
  let(:scheduler) do
    NanoGPT::LRScheduler.new(
      learning_rate: 1e-3,
      min_lr: 1e-4,
      warmup_iters: 100,
      lr_decay_iters: 1000
    )
  end

  describe "#get_lr" do
    context "during warmup phase" do
      it "returns linearly increasing learning rate" do
        # At iter 0: lr * 1/101
        lr_0 = scheduler.get_lr(0)
        expect(lr_0).to be_within(1e-8).of(1e-3 * 1.0 / 101)

        # At iter 50: lr * 51/101
        lr_50 = scheduler.get_lr(50)
        expect(lr_50).to be_within(1e-8).of(1e-3 * 51.0 / 101)

        # At iter 99: lr * 100/101
        lr_99 = scheduler.get_lr(99)
        expect(lr_99).to be_within(1e-8).of(1e-3 * 100.0 / 101)
      end

      it "increases monotonically during warmup" do
        lrs = (0...100).map { |i| scheduler.get_lr(i) }
        expect(lrs).to eq(lrs.sort)
      end
    end

    context "at warmup boundary" do
      it "reaches approximately max learning rate" do
        lr = scheduler.get_lr(100)
        # At iter 100, warmup is done, cosine starts at max
        expect(lr).to be_within(1e-5).of(1e-3)
      end
    end

    context "during cosine decay phase" do
      it "decays from max to min" do
        lr_start = scheduler.get_lr(100)  # Start of decay
        lr_mid = scheduler.get_lr(550)    # Middle of decay
        lr_end = scheduler.get_lr(1000)   # End of decay

        expect(lr_start).to be > lr_mid
        expect(lr_mid).to be > lr_end
        expect(lr_end).to be_within(1e-8).of(1e-4)
      end

      it "follows cosine curve" do
        # At midpoint, should be halfway between max and min
        mid_iter = 100 + (1000 - 100) / 2  # 550
        lr_mid = scheduler.get_lr(mid_iter)
        expected_mid = (1e-3 + 1e-4) / 2  # 5.5e-4

        expect(lr_mid).to be_within(1e-5).of(expected_mid)
      end
    end

    context "after decay phase" do
      it "returns minimum learning rate" do
        expect(scheduler.get_lr(1001)).to eq(1e-4)
        expect(scheduler.get_lr(2000)).to eq(1e-4)
        expect(scheduler.get_lr(10000)).to eq(1e-4)
      end
    end
  end

  describe "#step" do
    it "updates optimizer learning rate" do
      # Create a simple mock optimizer with param_groups
      config = NanoGPT::GPTConfig.new(
        block_size: 16, vocab_size: 50, n_layer: 1, n_head: 2, n_embd: 32
      )
      model = NanoGPT::GPT.new(config)
      optimizer = model.configure_optimizers(
        weight_decay: 0.1, learning_rate: 1e-3, betas: [0.9, 0.99], device_type: "cpu"
      )

      # Step and check LR is applied
      new_lr = scheduler.step(optimizer, 500)

      optimizer.param_groups.each do |group|
        expect(group[:lr]).to eq(new_lr)
      end
    end
  end
end
