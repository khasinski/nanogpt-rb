# frozen_string_literal: true

module NanoGPT
  # Loads batches from binary token files
  # Memory-efficient: reads from file each batch (like Python's memmap recreation)
  class DataLoader
    attr_reader :block_size, :batch_size

    BYTES_PER_TOKEN = 2  # uint16

    def initialize(data_dir:, block_size:, batch_size:, device: "cpu")
      @data_dir = data_dir
      @block_size = block_size
      @batch_size = batch_size
      @device = device

      # Store file paths and sizes (NOT the data itself)
      @train_path = File.join(data_dir, "train.bin")
      @val_path = File.join(data_dir, "val.bin")

      @train_size = File.size(@train_path) / BYTES_PER_TOKEN
      @val_size = File.size(@val_path) / BYTES_PER_TOKEN
    end

    def train_size
      @train_size
    end

    def val_size
      @val_size
    end

    # Get a batch of data
    # Memory-efficient: recreates data view per batch to avoid memory leak
    # (matches Python's memmap recreation pattern)
    def get_batch(split)
      path = split == :train ? @train_path : @val_path
      data_size = split == :train ? @train_size : @val_size

      # Random starting indices
      max_start = data_size - @block_size - 1
      indices = Array.new(@batch_size) { rand(0..max_start) }

      # Read only the bytes we need from file (memory-efficient)
      # This mimics Python's memmap recreation per batch
      x_arrays = []
      y_arrays = []

      File.open(path, "rb") do |f|
        indices.each do |i|
          # Read x: tokens[i:i+block_size]
          f.seek(i * BYTES_PER_TOKEN)
          x_bytes = f.read((@block_size + 1) * BYTES_PER_TOKEN)
          tokens = x_bytes.unpack("S<*")  # uint16 little-endian

          x_arrays << tokens[0...@block_size]
          y_arrays << tokens[1..@block_size]
        end
      end

      # Create tensors directly from arrays (avoiding Numo intermediate)
      x = Torch.tensor(x_arrays, dtype: :long)
      y = Torch.tensor(y_arrays, dtype: :long)

      # Move to device (CPU, CUDA, or MPS)
      if @device != "cpu"
        x = x.to(@device)
        y = y.to(@device)
      end

      [x, y]
    end
  end
end
