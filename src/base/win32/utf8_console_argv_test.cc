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

#include "base/win32/utf8_console_argv.h"

#include "testing/gunit.h"

namespace mozc::win32 {
namespace {

TEST(Utf8ConsoleArgvTest, ConvertsWideArgumentsToUtf8) {
  wchar_t executable[] = L"score_probe.exe";
  wchar_t reading[] = L"--reading=あめ";
  wchar_t context[] = L"--preceding_text=子どもに";
  wchar_t* wide_argv[] = {executable, reading, context};

  Utf8ConsoleArgv command_line(3, wide_argv);

  ASSERT_EQ(command_line.argc(), 3);
  EXPECT_STREQ(command_line.argv()[0], "score_probe.exe");
  EXPECT_STREQ(command_line.argv()[1], "--reading=あめ");
  EXPECT_STREQ(command_line.argv()[2], "--preceding_text=子どもに");
  EXPECT_EQ(command_line.argv()[3], nullptr);
}

}  // namespace
}  // namespace mozc::win32
