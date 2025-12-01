# frozen_string_literal: true

module NanoGPT
  # Device detection and management
  module Device
    class << self
      # Auto-detect the best available device
      # Priority: CUDA > MPS > CPU
      def auto
        return "cuda" if cuda_available?
        return "mps" if mps_available?

        "cpu"
      end

      # Check if CUDA is available
      def cuda_available?
        Torch::CUDA.available?
      rescue StandardError
        false
      end

      # Check if MPS (Metal Performance Shaders) is available
      # MPS is Apple Silicon GPU acceleration
      def mps_available?
        # Try to create a tensor on MPS device
        Torch.tensor([1.0], device: "mps")
        true
      rescue StandardError
        false
      end

      # Get device type string (for optimizer configuration, etc.)
      def type(device)
        case device.to_s
        when /cuda/ then "cuda"
        when /mps/ then "mps"
        else "cpu"
        end
      end

      # Check if device is GPU (CUDA or MPS)
      def gpu?(device)
        %w[cuda mps].include?(type(device))
      end

      # Print device info
      def info
        puts "Device detection:"
        puts "  CUDA available: #{cuda_available?}"
        puts "  MPS available: #{mps_available?}"
        puts "  Auto-selected: #{auto}"
      end
    end
  end
end
