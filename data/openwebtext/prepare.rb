#!/usr/bin/env ruby
# frozen_string_literal: true

$stdout.sync = true

# Prepare the OpenWebText dataset for GPT-2 style language modeling.
# Uses GPT-2 BPE tokenization via tiktoken (vocab_size=50257).
#
# The full dataset is ~9B tokens (~54GB for train.bin).
#
# Usage:
#   # Download dataset first (requires Python datasets library):
#   pip install datasets
#   python -c "from datasets import load_dataset; ds = load_dataset('Skylion007/openwebtext', split='train'); ds.to_parquet('data/openwebtext/raw/train.parquet')"
#
#   # Or download individual tar files from HuggingFace:
#   # https://huggingface.co/datasets/Skylion007/openwebtext/tree/main/subsets
#
#   # Then process:
#   bundle exec ruby data/openwebtext/prepare.rb
#
#   # Process only first N documents (for testing):
#   bundle exec ruby data/openwebtext/prepare.rb --max_docs=10000
#
# Options:
#   --max_docs=N     Only process first N documents (for testing)
#   --val_ratio=F    Validation split ratio (default: 0.0005)

require "numo/narray"
require "tiktoken_ruby"
require "fileutils"
require "rubygems/package"
require "zlib"

SCRIPT_DIR = File.dirname(__FILE__)
OUTPUT_DIR = ENV["NANOGPT_DATA_DIR"] || SCRIPT_DIR
RAW_DIR = File.join(OUTPUT_DIR, "raw")

FileUtils.mkdir_p(OUTPUT_DIR) if ENV["NANOGPT_DATA_DIR"]
DEFAULT_VAL_RATIO = 0.0005  # ~0.5% for validation

def parse_args
  config = {
    max_docs: nil,
    val_ratio: DEFAULT_VAL_RATIO
  }

  ARGV.each do |arg|
    next unless arg.start_with?("--") && arg.include?("=")

    key, val = arg[2..].split("=", 2)
    key = key.to_sym

    case key
    when :max_docs
      config[:max_docs] = val.to_i
    when :val_ratio
      config[:val_ratio] = val.to_f
    end
  end

  config
end

def find_data_files
  # Look for various supported formats in both SCRIPT_DIR (gem) and OUTPUT_DIR (local)
  patterns = [
    File.join(RAW_DIR, "**", "*.parquet"),  # Parquet (from Python export)
    File.join(RAW_DIR, "**", "*.tar"),       # Original tar files
    File.join(RAW_DIR, "**", "*.txt"),       # Plain text files
    File.join(OUTPUT_DIR, "*.parquet"),
    File.join(OUTPUT_DIR, "*.tar"),
    File.join(OUTPUT_DIR, "*.txt"),
    File.join(SCRIPT_DIR, "raw", "**", "*.parquet"),
    File.join(SCRIPT_DIR, "raw", "**", "*.tar"),
    File.join(SCRIPT_DIR, "raw", "**", "*.txt"),
    File.join(SCRIPT_DIR, "*.parquet"),
    File.join(SCRIPT_DIR, "*.tar"),
    File.join(SCRIPT_DIR, "*.txt")
  ]

  files = patterns.flat_map { |p| Dir.glob(p) }.uniq.sort
  files
end

def process_parquet_files(files, enc, eot, max_docs)
  require "parquet"

  tokens = []
  doc_count = 0

  files.select { |f| f.end_with?(".parquet") }.each do |path|
    puts "Processing parquet: #{File.basename(path)}..."

    Parquet.each_row(path) do |row|
      break if max_docs && doc_count >= max_docs

      text = row["text"]
      next if text.nil? || text.empty?

      doc_tokens = enc.encode(text)
      doc_tokens << eot
      tokens.concat(doc_tokens)
      doc_count += 1

      if doc_count % 10_000 == 0
        print "\r  Processed #{doc_count} documents, #{tokens.length} tokens..."
      end
    end

    puts "\r  Processed #{doc_count} documents, #{tokens.length} tokens    "
    break if max_docs && doc_count >= max_docs
  end

  [tokens, doc_count]
end

def process_tar_files(files, enc, eot, max_docs)
  tokens = []
  doc_count = 0

  files.select { |f| f.end_with?(".tar") }.each do |tar_path|
    puts "Processing tar: #{File.basename(tar_path)}..."

    File.open(tar_path, "rb") do |file|
      Gem::Package::TarReader.new(file) do |tar|
        tar.each do |entry|
          break if max_docs && doc_count >= max_docs
          next unless entry.file? && entry.full_name.end_with?(".txt", ".xz")

          # Read content (decompress if .xz)
          content = entry.read
          if entry.full_name.end_with?(".xz")
            # Skip xz files - would need xz gem
            next
          end

          text = content.force_encoding("UTF-8")
          next if text.empty?

          doc_tokens = enc.encode(text)
          doc_tokens << eot
          tokens.concat(doc_tokens)
          doc_count += 1

          if doc_count % 1_000 == 0
            print "\r  Processed #{doc_count} documents, #{tokens.length} tokens..."
          end
        end
      end
    end

    puts "\r  Processed #{doc_count} documents, #{tokens.length} tokens    "
    break if max_docs && doc_count >= max_docs
  end

  [tokens, doc_count]
end

def process_txt_files(files, enc, eot, max_docs)
  tokens = []
  doc_count = 0

  files.select { |f| f.end_with?(".txt") }.each do |path|
    break if max_docs && doc_count >= max_docs

    puts "Processing text: #{File.basename(path)}..."
    text = File.read(path, encoding: "UTF-8")
    next if text.empty?

    doc_tokens = enc.encode(text)
    doc_tokens << eot
    tokens.concat(doc_tokens)
    doc_count += 1
  end

  [tokens, doc_count]
end

def write_binary(tokens, path)
  arr = Numo::UInt16.cast(tokens)
  File.binwrite(path, arr.to_binary)
  size_mb = File.size(path) / 1_000_000.0
  puts "  Wrote #{path} (#{tokens.length} tokens, #{size_mb.round(1)}MB)"
end

def main
  config = parse_args
  puts "OpenWebText Data Preparation"
  puts "=" * 50
  puts "Config: #{config}"
  puts ""

  # Initialize tokenizer
  enc = Tiktoken.get_encoding(:r50k_base)
  eot = enc.encode("<|endoftext|>").first
  puts "Using GPT-2 BPE tokenizer (vocab_size=50257, EOT=#{eot})"
  puts ""

  # Find data files
  files = find_data_files

  if files.empty?
    puts "No data files found!"
    puts ""
    puts "Please download the OpenWebText dataset first:"
    puts ""
    puts "Option 1: Using Python datasets (recommended):"
    puts "  pip install datasets"
    puts "  python -c \"\""
    puts "    from datasets import load_dataset"
    puts "    ds = load_dataset('Skylion007/openwebtext', split='train')"
    puts "    ds.to_parquet('data/openwebtext/raw/train.parquet')"
    puts "  \"\""
    puts ""
    puts "Option 2: Download tar files from HuggingFace:"
    puts "  https://huggingface.co/datasets/Skylion007/openwebtext/tree/main/subsets"
    puts "  Place them in: data/openwebtext/raw/"
    puts ""
    puts "Option 3: Place plain text files in data/openwebtext/raw/"
    exit 1
  end

  puts "Found #{files.length} data files:"
  files.each { |f| puts "  - #{File.basename(f)}" }
  puts ""

  # Process files by type
  all_tokens = []
  total_docs = 0

  # Try parquet first (most efficient)
  parquet_files = files.select { |f| f.end_with?(".parquet") }
  if parquet_files.any?
    tokens, docs = process_parquet_files(parquet_files, enc, eot, config[:max_docs])
    all_tokens.concat(tokens)
    total_docs += docs
  end

  # Then tar files
  remaining = config[:max_docs] ? config[:max_docs] - total_docs : nil
  if remaining.nil? || remaining > 0
    tar_files = files.select { |f| f.end_with?(".tar") }
    if tar_files.any?
      tokens, docs = process_tar_files(tar_files, enc, eot, remaining)
      all_tokens.concat(tokens)
      total_docs += docs
    end
  end

  # Finally plain text
  remaining = config[:max_docs] ? config[:max_docs] - total_docs : nil
  if remaining.nil? || remaining > 0
    txt_files = files.select { |f| f.end_with?(".txt") }
    if txt_files.any?
      tokens, docs = process_txt_files(txt_files, enc, eot, remaining)
      all_tokens.concat(tokens)
      total_docs += docs
    end
  end

  if all_tokens.empty?
    puts "Error: No tokens extracted from data files"
    exit 1
  end

  puts ""
  puts "=" * 50
  puts "Total: #{total_docs} documents, #{all_tokens.length} tokens"
  puts ""

  # Split into train/val
  val_size = (all_tokens.length * config[:val_ratio]).to_i
  val_size = [val_size, 1].max  # At least 1 token for val

  train_tokens = all_tokens[0...-val_size]
  val_tokens = all_tokens[-val_size..]

  puts "Train: #{train_tokens.length} tokens"
  puts "Val: #{val_tokens.length} tokens"
  puts ""

  # Write binary files
  puts "Writing binary files..."
  write_binary(train_tokens, File.join(OUTPUT_DIR, "train.bin"))
  write_binary(val_tokens, File.join(OUTPUT_DIR, "val.bin"))

  puts ""
  puts "Done! OpenWebText dataset prepared."
  puts "Note: No meta.json (uses GPT-2 BPE tokenizer, vocab_size=50257)"
  puts ""
  puts "To train:"
  puts "  bundle exec ruby bin/train --dataset=openwebtext"
end

main
