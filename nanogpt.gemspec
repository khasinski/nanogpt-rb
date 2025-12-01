# frozen_string_literal: true

require_relative "lib/nano_gpt/version"

Gem::Specification.new do |spec|
  spec.name = "nanogpt"
  spec.version = NanoGPT::VERSION
  spec.authors = ["Chris Hasiński"]
  spec.email = ["krzysztof.hasinski@gmail.com"]

  spec.summary = "A Ruby port of Karpathy's nanoGPT"
  spec.description = "Train and run GPT models in Ruby using torch.rb. " \
                     "A minimal, educational implementation of GPT-2 style models."
  spec.homepage = "https://github.com/khasinski/nanogpt-rb"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1.0"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/main/CHANGELOG.md"

  spec.files = Dir.chdir(__dir__) do
    `git ls-files -z`.split("\x0").reject do |f|
      (File.expand_path(f) == __FILE__) ||
        f.start_with?("spec/", "test/", ".git", ".github", "vendor/", "python/")
    end
  end
  spec.bindir = "exe"
  spec.executables = ["nanogpt"]
  spec.require_paths = ["lib"]

  spec.add_dependency "torch-rb", "~> 0.14"
  spec.add_dependency "numo-narray", "~> 0.9"
  spec.add_dependency "tiktoken_ruby", "~> 0.0"

  spec.add_development_dependency "rspec", "~> 3.12"
end
