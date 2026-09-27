// Copyright 2026 LLM Japanese Input Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//     * Neither the name of LLM Japanese Input Authors nor the names of its
// contributors may be used to endorse or promote products derived from this
// software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "llama.h"

int main(int argc, char** argv) {
  if (argc != 3) {
    return 2;
  }

  std::ifstream input(argv[2], std::ios::binary);
  if (!input) {
    return 3;
  }
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());

  llama_backend_init();
  llama_model_params params = llama_model_default_params();
  params.vocab_only = true;
  llama_model* model = llama_model_load_from_file(argv[1], params);
  if (model == nullptr) {
    llama_backend_free();
    return 4;
  }

  const llama_vocab* vocabulary = llama_model_get_vocab(model);
  const int32_t required = llama_tokenize(
      vocabulary, text.data(), static_cast<int32_t>(text.size()), nullptr, 0,
      /*add_special=*/false, /*parse_special=*/false);
  if (required == std::numeric_limits<int32_t>::min()) {
    llama_model_free(model);
    llama_backend_free();
    return 5;
  }

  const int32_t token_count = required < 0 ? -required : required;
  std::vector<llama_token> tokens(static_cast<size_t>(token_count));
  if (tokens.empty()) {
    std::cout << "[]\n";
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }
  const int32_t actual = llama_tokenize(
      vocabulary, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
      static_cast<int32_t>(tokens.size()), /*add_special=*/false,
      /*parse_special=*/false);
  if (actual != static_cast<int32_t>(tokens.size())) {
    llama_model_free(model);
    llama_backend_free();
    return 6;
  }

  std::cout << '[';
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i != 0) {
      std::cout << ',';
    }
    std::cout << tokens[i];
  }
  std::cout << "]\n";

  llama_model_free(model);
  llama_backend_free();
  return 0;
}
