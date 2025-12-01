#!/usr/bin/env ruby
# frozen_string_literal: true

# Prepare the Shakespeare dataset with GPT-2 BPE tokenization.
# Downloads the tiny shakespeare dataset and creates train.bin and val.bin
# using GPT-2's BPE tokenizer (vocab_size=50257).
#
# Usage: bundle exec ruby data/shakespeare/prepare.rb

require "net/http"
require "openssl"
require "numo/narray"
require "tiktoken_ruby"
require "fileutils"

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SCRIPT_DIR = File.dirname(__FILE__)
OUTPUT_DIR = ENV["NANOGPT_DATA_DIR"] || SCRIPT_DIR

FileUtils.mkdir_p(OUTPUT_DIR) if ENV["NANOGPT_DATA_DIR"]

def download_file(url)
  uri = URI(url)
  http = Net::HTTP.new(uri.host, uri.port)
  http.use_ssl = true
  http.verify_mode = OpenSSL::SSL::VERIFY_NONE  # GitHub CDN has CRL issues
  response = http.get(uri.request_uri)
  response.body
end

# Download data if not exists
input_path = File.join(SCRIPT_DIR, "input.txt")
unless File.exist?(input_path)
  puts "Downloading tiny shakespeare..."
  data = download_file(DATA_URL)
  File.write(input_path, data)
end

data = File.read(input_path)
puts "Length of dataset in characters: #{data.length}"

# Initialize GPT-2 BPE tokenizer
enc = Tiktoken.get_encoding(:r50k_base)
puts "Using GPT-2 BPE tokenizer (vocab_size=50257)"

# Train/val split (90/10)
n = data.length
train_data = data[0...(n * 0.9).to_i]
val_data = data[(n * 0.9).to_i..]

# Encode to integers using BPE
train_ids = enc.encode(train_data)
val_ids = enc.encode(val_data)
puts "Train has #{train_ids.length} tokens"
puts "Val has #{val_ids.length} tokens"

# Export to binary files (uint16)
train_arr = Numo::UInt16.cast(train_ids)
val_arr = Numo::UInt16.cast(val_ids)
File.binwrite(File.join(OUTPUT_DIR, "train.bin"), train_arr.to_binary)
File.binwrite(File.join(OUTPUT_DIR, "val.bin"), val_arr.to_binary)

# No meta.json - indicates GPT-2 BPE tokenizer should be used
puts "Done! Created train.bin and val.bin"
puts "Note: No meta.json (uses GPT-2 BPE tokenizer, vocab_size=50257)"
