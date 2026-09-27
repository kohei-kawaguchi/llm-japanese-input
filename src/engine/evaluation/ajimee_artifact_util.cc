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

#include "engine/evaluation/ajimee_artifact_util.h"

#include <array>
#include <cstddef>
#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/file_util.h"
extern "C" {
#include "sha256/sha256.h"
}

namespace mozc::engine::evaluation {
namespace {

std::string DigestToHex(const unsigned char* digest, size_t size) {
  constexpr char kHexDigits[] = "0123456789abcdef";
  std::string result(size * 2, '\0');
  for (size_t i = 0; i < size; ++i) {
    result[2 * i] = kHexDigits[digest[i] >> 4];
    result[2 * i + 1] = kHexDigits[digest[i] & 0x0f];
  }
  return result;
}

}  // namespace

std::string Sha256Bytes(absl::string_view contents) {
  sha256_t hash;
  sha256_init(&hash);
  sha256_update(
      &hash, reinterpret_cast<const unsigned char*>(contents.data()),
      contents.size());
  std::array<unsigned char, SHA256_DIGEST_SIZE> digest;
  sha256_final(&hash, digest.data());
  return DigestToHex(digest.data(), digest.size());
}

absl::StatusOr<std::string> Sha256File(absl::string_view path) {
  absl::StatusOr<std::string> contents = FileUtil::GetContents(path);
  if (!contents.ok()) {
    return contents.status();
  }
  return Sha256Bytes(*contents);
}

}  // namespace mozc::engine::evaluation
