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

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "testing/gunit.h"

namespace mozc::engine {
namespace {

void ExpectLower(absl::string_view input, absl::string_view expected) {
  const absl::StatusOr<std::string> result = PythonUnicode14Lower(input);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(*result, expected);
}

TEST(PythonUnicode14LowerTest, PreservesJapaneseAndLowersLatin) {
  ExpectLower("", "");
  ExpectLower("日本語入力", "日本語入力");
  ExpectLower("Google日本語IME", "google日本語ime");
  ExpectLower("Ｇｏｏｇｌｅ日本語ＩＭＥ", "ｇｏｏｇｌｅ日本語ｉｍｅ");
}

TEST(PythonUnicode14LowerTest, AppliesFullMultiCodePointMapping) {
  ExpectLower("İ", "i\u0307");
  ExpectLower("AİZ", "ai\u0307z");
}

TEST(PythonUnicode14LowerTest, LowersEachUnicodeScalarIndependently) {
  ExpectLower("ΟΣ", "οσ");
  ExpectLower("ος", "ος");
  ExpectLower("ΟΣΑ", "οσα");
  ExpectLower("A\u0301Σ", "a\u0301σ");
  ExpectLower("AΣ\u0301B", "aσ\u0301b");
  ExpectLower("A\u0345Σ", "a\u0345σ");
  ExpectLower("AΣ\u0345B", "aσ\u0345b");
  ExpectLower("AΣ’", "aσ’");
  ExpectLower("AΣ’B", "aσ’b");
  ExpectLower("[PAD]", "[pad]");
}

TEST(PythonUnicode14LowerTest, PreservesEmbeddedNull) {
  const std::string input("A\0B", 3);
  const std::string expected("a\0b", 3);
  ExpectLower(input, expected);
}

TEST(PythonUnicode14LowerTest, RejectsInvalidUtf8) {
  for (const std::string& input : {
           std::string("\xC0\xAF", 2),
           std::string("\xED\xA0\x80", 3),
           std::string("\xF0\x9F\x92", 3),
       }) {
    const absl::StatusOr<std::string> result = PythonUnicode14Lower(input);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace mozc::engine
