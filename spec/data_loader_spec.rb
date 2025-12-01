# frozen_string_literal: true

require "spec_helper"
require "tmpdir"

RSpec.describe NanoGPT::DataLoader do
  let(:data_dir) { Dir.mktmpdir }
  let(:block_size) { 8 }
  let(:batch_size) { 4 }

  before do
    # Create test data files
    # Train: 100 tokens (0-99 repeated)
    train_data = Numo::UInt16.cast((0...100).to_a)
    File.binwrite(File.join(data_dir, "train.bin"), train_data.to_binary)

    # Val: 50 tokens (100-149)
    val_data = Numo::UInt16.cast((100...150).to_a)
    File.binwrite(File.join(data_dir, "val.bin"), val_data.to_binary)
  end

  after do
    FileUtils.rm_rf(data_dir)
  end

  let(:loader) do
    NanoGPT::DataLoader.new(
      data_dir: data_dir,
      block_size: block_size,
      batch_size: batch_size,
      device: "cpu"
    )
  end

  describe "#initialize" do
    it "loads train and val data" do
      expect(loader.train_size).to eq(100)
      expect(loader.val_size).to eq(50)
    end
  end

  describe "#get_batch" do
    context "for training split" do
      it "returns tensors of correct shape" do
        x, y = loader.get_batch(:train)

        expect(x.shape).to eq([batch_size, block_size])
        expect(y.shape).to eq([batch_size, block_size])
      end

      it "returns long dtype tensors" do
        x, y = loader.get_batch(:train)

        expect(x.dtype).to eq(:int64)
        expect(y.dtype).to eq(:int64)
      end

      it "returns y as x shifted by 1" do
        x, y = loader.get_batch(:train)

        # For each batch item, y should be x shifted by 1
        batch_size.times do |b|
          x_vals = x[b, 0...-1].to_a
          y_vals = y[b, 0...-1].to_a

          # y[i] should equal the token after x[i] in the sequence
          # Since y is shifted by 1 from x's start position
          x_first = x[b, 0].item
          y_first = y[b, 0].item
          expect(y_first).to eq(x_first + 1)
        end
      end

      it "returns different batches on successive calls" do
        x1, _y1 = loader.get_batch(:train)
        x2, _y2 = loader.get_batch(:train)

        # Very unlikely to be exactly equal with random sampling
        # (could happen but probability is extremely low)
        expect(x1.to_a).not_to eq(x2.to_a)
      end
    end

    context "for validation split" do
      it "returns tensors of correct shape" do
        x, y = loader.get_batch(:val)

        expect(x.shape).to eq([batch_size, block_size])
        expect(y.shape).to eq([batch_size, block_size])
      end

      it "samples from validation data range" do
        x, _y = loader.get_batch(:val)

        # All values should be in range 100-149 (our val data)
        x.to_a.flatten.each do |val|
          expect(val).to be >= 100
          expect(val).to be < 150
        end
      end
    end
  end
end
