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

#include "engine/evaluation/evaluation_protobuf_util.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "base/protobuf/coded_stream.h"
#include "base/protobuf/descriptor.h"
#include "base/protobuf/message.h"
#include "base/protobuf/text_format.h"
#include "base/protobuf/zero_copy_stream_impl.h"

#ifdef GetMessage
#undef GetMessage
#endif

namespace mozc::engine::evaluation {

bool HasUnknownFieldsRecursively(const protobuf::Message& message) {
  const protobuf::Reflection* reflection = message.GetReflection();
  if (!reflection->GetUnknownFields(message).empty()) {
    return true;
  }
  std::vector<const protobuf::FieldDescriptor*> fields;
  reflection->ListFields(message, &fields);
  for (const protobuf::FieldDescriptor* field : fields) {
    if (field->cpp_type() != protobuf::FieldDescriptor::CPPTYPE_MESSAGE) {
      continue;
    }
    if (field->is_repeated()) {
      for (int i = 0; i < reflection->FieldSize(message, field); ++i) {
        if (HasUnknownFieldsRecursively(
                reflection->GetRepeatedMessage(message, field, i))) {
          return true;
        }
      }
    } else if (HasUnknownFieldsRecursively(
                   reflection->GetMessage(message, field))) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::string> SerializeDeterministically(
    const protobuf::Message& message) {
  if (!message.IsInitialized()) {
    return absl::InvalidArgumentError("protobuf message is not initialized");
  }
  std::string output;
  {
    protobuf::io::StringOutputStream stream(&output);
    protobuf::io::CodedOutputStream coded_stream(&stream);
    coded_stream.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded_stream) ||
        coded_stream.HadError()) {
      return absl::DataLossError("deterministic protobuf serialization failed");
    }
  }
  return output;
}

absl::StatusOr<std::string> ReviewTextproto(
    const protobuf::Message& message) {
  if (!message.IsInitialized()) {
    return absl::InvalidArgumentError("protobuf message is not initialized");
  }
  protobuf::TextFormat::Printer printer;
  printer.SetUseUtf8StringEscaping(true);
  std::string output;
  if (!printer.PrintToString(message, &output)) {
    return absl::DataLossError("protobuf text serialization failed");
  }
  return output;
}

absl::Status ParseTextproto(absl::string_view contents,
                            protobuf::Message* message) {
  if (message == nullptr) {
    return absl::InvalidArgumentError("protobuf output must not be null");
  }
  protobuf::TextFormat::Parser parser;
  if (!parser.ParseFromString(contents, message)) {
    return absl::InvalidArgumentError("protobuf text parsing failed");
  }
  if (!message->IsInitialized()) {
    return absl::InvalidArgumentError("protobuf message is not initialized");
  }
  return absl::OkStatus();
}

}  // namespace mozc::engine::evaluation
