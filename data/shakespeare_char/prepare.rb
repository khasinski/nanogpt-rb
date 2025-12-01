#!/usr/bin/env ruby
# frozen_string_literal: true

# Prepare the Shakespeare dataset for character-level language modeling.
# Downloads the tiny shakespeare dataset and creates train.bin, val.bin, and meta.json

require "net/http"
require "openssl"
require "numo/narray"
require "json"
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

# Build vocabulary from all unique characters
chars = data.chars.uniq.sort
vocab_size = chars.size
puts "All unique characters: #{chars.join.inspect}"
puts "Vocab size: #{vocab_size}"

# Create mappings
stoi = chars.each_with_index.to_h
itos = chars.each_with_index.map { |c, i| [i, c] }.to_h

# Encode function
encode = ->(s) { s.chars.map { |c| stoi[c] } }

# Train/val split (90/10)
n = data.length
train_data = data[0...(n * 0.9).to_i]
val_data = data[(n * 0.9).to_i..]

# Encode to integers
train_ids = encode.call(train_data)
val_ids = encode.call(val_data)
puts "Train has #{train_ids.length} tokens"
puts "Val has #{val_ids.length} tokens"

# Export to binary files (uint16)
train_arr = Numo::UInt16.cast(train_ids)
val_arr = Numo::UInt16.cast(val_ids)
File.binwrite(File.join(OUTPUT_DIR, "train.bin"), train_arr.to_binary)
File.binwrite(File.join(OUTPUT_DIR, "val.bin"), val_arr.to_binary)

# Save meta information as JSON
meta = {
  "vocab_size" => vocab_size,
  "itos" => itos.transform_keys(&:to_s),
  "stoi" => stoi
}
File.write(File.join(OUTPUT_DIR, "meta.json"), JSON.pretty_generate(meta))

puts "Done! Created train.bin, val.bin, and meta.json"
