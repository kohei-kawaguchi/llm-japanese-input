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

#ifndef MOZC_BASE_WIN32_UTF8_CONSOLE_ARGV_H_
#define MOZC_BASE_WIN32_UTF8_CONSOLE_ARGV_H_

#include <string>
#include <vector>

#include "base/win32/wide_char.h"

namespace mozc::win32 {

class Utf8ConsoleArgv final {
 public:
  Utf8ConsoleArgv(int argc, wchar_t* const argv[]) : argc_(argc) {
    args_.reserve(argc_);
    argv_.reserve(argc_ + 1);
    for (int i = 0; i < argc_; ++i) {
      args_.push_back(WideToUtf8(argv[i]));
    }
    for (std::string& argument : args_) {
      argv_.push_back(argument.data());
    }
    argv_.push_back(nullptr);
  }

  int argc() const { return argc_; }
  char** argv() { return argv_.data(); }

 private:
  int argc_;
  std::vector<std::string> args_;
  std::vector<char*> argv_;
};

}  // namespace mozc::win32

#endif  // MOZC_BASE_WIN32_UTF8_CONSOLE_ARGV_H_
