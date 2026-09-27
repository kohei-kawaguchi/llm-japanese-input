// Copyright 2026 LLM Japanese Input Authors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the distribution.
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

#include "engine/python_unicode14_lower.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/strings/unicode.h"

namespace mozc::engine {
namespace {

struct LowerMapping {
  char32_t source;
  uint32_t offset;
  uint8_t length;
};

#include "engine/python_unicode14_lower_data.inc"

const LowerMapping* FindMapping(char32_t code_point) {
  const LowerMapping* mapping = std::lower_bound(
      std::begin(kLowerMappings), std::end(kLowerMappings), code_point,
      [](const LowerMapping& mapping, char32_t value) {
        return mapping.source < value;
      });
  return mapping != std::end(kLowerMappings) && mapping->source == code_point
             ? mapping
             : nullptr;
}

void AppendCodePoint(char32_t code_point, std::string* output) {
  if (code_point == 0) {
    output->push_back('\0');
    return;
  }
  strings::StrAppendChar32(output, code_point);
}

}  // namespace

absl::StatusOr<std::string> PythonUnicode14Lower(absl::string_view input) {
  if (!strings::IsValidUtf8(input)) {
    return absl::InvalidArgumentError("input is not valid UTF-8");
  }

  const std::u32string code_points = strings::Utf8ToUtf32(input);
  std::string output;
  output.reserve(input.size());
  for (const char32_t code_point : code_points) {
    const LowerMapping* mapping = FindMapping(code_point);
    if (mapping == nullptr) {
      AppendCodePoint(code_point, &output);
      continue;
    }
    for (uint8_t offset = 0; offset < mapping->length; ++offset) {
      AppendCodePoint(kLowerMappingData[mapping->offset + offset], &output);
    }
  }
  return output;
}

}  // namespace mozc::engine
