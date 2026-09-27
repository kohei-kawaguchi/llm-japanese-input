#!/usr/bin/env python3

import hashlib
from pathlib import Path

import cursor_grok_ranker


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _read_varint(data, index):
  value = 0
  shift = 0
  while True:
    if index >= len(data):
      raise cursor_grok_ranker.RankerError("frozen corpus varint is truncated")
    byte = data[index]
    index += 1
    value |= (byte & 0x7F) << shift
    if byte < 0x80:
      return value, index
    shift += 7
    if shift > 63:
      raise cursor_grok_ranker.RankerError("frozen corpus varint is invalid")


def _skip_field(data, index, wire_type):
  if wire_type == 0:
    _, index = _read_varint(data, index)
    return index
  if wire_type == 1:
    return index + 8
  if wire_type == 2:
    length, index = _read_varint(data, index)
    return index + length
  if wire_type == 5:
    return index + 4
  raise cursor_grok_ranker.RankerError("frozen corpus wire type is unknown")


def _read_fields(data):
  fields = {}
  index = 0
  while index < len(data):
    key, index = _read_varint(data, index)
    field_number = key >> 3
    wire_type = key & 7
    if wire_type == 0:
      value, index = _read_varint(data, index)
      fields.setdefault(field_number, []).append(("varint", value))
    elif wire_type == 2:
      length, index = _read_varint(data, index)
      value = data[index:index + length]
      if len(value) != length:
        raise cursor_grok_ranker.RankerError("frozen corpus bytes are truncated")
      index += length
      fields.setdefault(field_number, []).append(("bytes", value))
    else:
      index = _skip_field(data, index, wire_type)
  return fields


def _one(fields, number, kind, label):
  values = fields.get(number, [])
  if len(values) != 1 or values[0][0] != kind:
    raise cursor_grok_ranker.RankerError(f"{label} is missing")
  return values[0][1]


def _optional_one(fields, number, kind, default):
  values = fields.get(number, [])
  if not values:
    return default
  if len(values) != 1 or values[0][0] != kind:
    raise cursor_grok_ranker.RankerError("frozen corpus field is invalid")
  return values[0][1]


def _decode_string(value, label):
  text = value.decode("utf-8")
  if "\0" in text:
    raise cursor_grok_ranker.RankerError(f"{label} contains a NUL")
  return text


def _decode_token(data):
  fields = _read_fields(data)
  return cursor_grok_ranker.RankerToken(
      session_generation=_one(fields, 1, "varint", "token.session_generation"),
      state_revision=_one(fields, 2, "varint", "token.state_revision"),
      request_sequence=_one(fields, 3, "varint", "token.request_sequence"),
  )


def _decode_candidate(data):
  fields = _read_fields(data)
  return cursor_grok_ranker.RankerCandidate(
      id=_one(fields, 1, "varint", "candidate.id"),
      key=_decode_string(_one(fields, 2, "bytes", "candidate.key"), "candidate.key"),
      value=_decode_string(
          _one(fields, 3, "bytes", "candidate.value"), "candidate.value"
      ),
      cost=_to_signed32(_one(fields, 4, "varint", "candidate.cost")),
      attributes=_one(fields, 5, "varint", "candidate.attributes"),
      consumed_key_size=_one(fields, 6, "varint", "candidate.consumed_key_size"),
      is_protected=bool(_one(fields, 7, "varint", "candidate.is_protected")),
  )


def _to_signed32(value):
  value = value & 0xFFFFFFFF
  if value >= 0x80000000:
    return value - 0x100000000
  return value


def _decode_segment(data):
  fields = _read_fields(data)
  candidates = tuple(
      _decode_candidate(item[1])
      for item in fields.get(3, [])
      if item[0] == "bytes"
  )
  if len(candidates) != len(fields.get(3, [])):
    raise cursor_grok_ranker.RankerError("segment candidates are invalid")
  return cursor_grok_ranker.RankerSegment(
      id=_one(fields, 1, "varint", "segment.id"),
      key=_decode_string(_one(fields, 2, "bytes", "segment.key"), "segment.key"),
      candidates=candidates,
  )


def _decode_request(data):
  fields = _read_fields(data)
  segments = tuple(
      _decode_segment(item[1])
      for item in fields.get(7, [])
      if item[0] == "bytes"
  )
  if len(segments) != len(fields.get(7, [])):
    raise cursor_grok_ranker.RankerError("request segments are invalid")
  return cursor_grok_ranker.RankerRequest(
      token=_decode_token(_one(fields, 1, "bytes", "request.token")),
      mode=_one(fields, 2, "varint", "request.mode"),
      preceding_text=_decode_string(
          _optional_one(fields, 3, "bytes", b""), "preceding_text"
      ),
      following_text=_decode_string(
          _optional_one(fields, 4, "bytes", b""), "following_text"
      ),
      reading=_decode_string(_one(fields, 5, "bytes", "reading"), "reading"),
      focused_segment_id=_one(fields, 6, "varint", "focused_segment_id"),
      segments=segments,
  )


def _decode_case(data):
  fields = _read_fields(data)
  return cursor_grok_ranker.FrozenCase(
      source_line=_one(fields, 1, "varint", "source_line"),
      conversion_reading=_decode_string(
          _one(fields, 2, "bytes", "conversion_reading"), "conversion_reading"
      ),
      mozc_baseline_output=_decode_string(
          _one(fields, 3, "bytes", "mozc_baseline_output"),
          "mozc_baseline_output",
      ),
      request=_decode_request(_one(fields, 4, "bytes", "request")),
  )


def _decode_ajimee_case(data):
  fields = _read_fields(data)
  return cursor_grok_ranker.FrozenCase(
      source_line=_one(fields, 1, "varint", "source_index"),
      conversion_reading=_decode_string(
          _one(fields, 2, "bytes", "normalized_hiragana_reading"),
          "normalized_hiragana_reading",
      ),
      mozc_baseline_output=_decode_string(
          _one(fields, 4, "bytes", "baseline_output"), "baseline_output"
      ),
      request=_decode_request(_one(fields, 5, "bytes", "request")),
  )


QUALITY_REGRESSION_FROZEN_SCHEMA_VERSION = 1
AJIMEE_FROZEN_SCHEMA_VERSION = 2


def load_frozen_corpus(
    path,
    expected_sha256,
    expected_case_count,
    decode_case=_decode_case,
    schema=QUALITY_REGRESSION_FROZEN_SCHEMA_VERSION,
):
  with open(path, "rb") as file:
    data = file.read()
  actual_sha256 = _sha256(data)
  if actual_sha256 != expected_sha256:
    raise cursor_grok_ranker.RankerError("frozen corpus SHA256 mismatch")
  fields = _read_fields(data)
  schema_version = _one(fields, 1, "varint", "frozen schema_version")
  if schema_version != schema:
    raise cursor_grok_ranker.RankerError("frozen corpus schema is invalid")
  cases = tuple(
      decode_case(item[1])
      for item in fields.get(5, [])
      if item[0] == "bytes"
  )
  if len(cases) != expected_case_count:
    raise cursor_grok_ranker.RankerError("frozen corpus case count mismatch")
  return cases, actual_sha256


def _parse_expected_rank(command, extra_tokens, conversion_expected_command):
  if extra_tokens:
    if len(extra_tokens) < 1:
      raise cursor_grok_ranker.RankerError("answer source rank field is invalid")
    return int(extra_tokens[0], 10)
  if command == conversion_expected_command:
    return 0
  prefix = conversion_expected_command
  if not command.startswith(prefix):
    return None
  suffix = command[len(prefix):]
  if not suffix or not suffix.startswith(" "):
    raise cursor_grok_ranker.RankerError("answer source command suffix is invalid")
  return int(suffix[1:], 10)


def load_answers(path, expected_sha256, conversion_expected_command, cases):
  with open(path, "rb") as file:
    data = file.read()
  actual_sha256 = _sha256(data)
  if actual_sha256 != expected_sha256:
    raise cursor_grok_ranker.RankerError("answer source SHA256 mismatch")
  if data.startswith(b"\xef\xbb\xbf") or b"\r" in data or b"\0" in data:
    raise cursor_grok_ranker.RankerError("answer source must be BOM-free LF-only UTF-8")
  text = data.decode("utf-8")
  retained = []
  seen_pairs = set()
  for line_number, line in enumerate(text.split("\n"), start=1):
    if not line or line.startswith("#"):
      continue
    tokens = line.split("\t")
    if len(tokens) < 4:
      raise cursor_grok_ranker.RankerError(
          f"answer source line {line_number} has fewer than four fields"
      )
    if any(not token for token in tokens):
      raise cursor_grok_ranker.RankerError(
          f"answer source line {line_number} contains an empty field"
      )
    command = tokens[3]
    extra = tokens[4:]
    if not command.startswith(conversion_expected_command):
      continue
    rank = _parse_expected_rank(command, extra, conversion_expected_command)
    if rank != 0:
      continue
    pair = (tokens[1], tokens[2])
    if pair in seen_pairs:
      continue
    seen_pairs.add(pair)
    retained.append((line_number, tokens[2]))
  answers = {source_line: value for source_line, value in retained}
  ordered = []
  for case in cases:
    if case.source_line not in answers:
      raise cursor_grok_ranker.RankerError(
          f"answer source is missing frozen source line {case.source_line}"
      )
    ordered.append((answers[case.source_line],))
  return tuple(ordered), actual_sha256


def load_ajimee_frozen_corpus(path, expected_sha256, expected_case_count):
  return load_frozen_corpus(
      path,
      expected_sha256,
      expected_case_count,
      _decode_ajimee_case,
      AJIMEE_FROZEN_SCHEMA_VERSION,
  )


def load_ajimee_answers(path, expected_sha256, cases):
  with open(path, "rb") as file:
    data = file.read()
  actual_sha256 = _sha256(data)
  if actual_sha256 != expected_sha256:
    raise cursor_grok_ranker.RankerError("AJIMEE answer corpus SHA256 mismatch")
  fields = _read_fields(data)
  answers = {}
  for kind, value in fields.get(3, []):
    if kind != "bytes":
      raise cursor_grok_ranker.RankerError("AJIMEE answer case is invalid")
    case_fields = _read_fields(value)
    accepted = tuple(
        _decode_string(item[1], "accepted_whole_outputs")
        for item in case_fields.get(2, [])
        if item[0] == "bytes"
    )
    if not accepted:
      raise cursor_grok_ranker.RankerError("AJIMEE answer case has no accepted output")
    answers[_one(case_fields, 1, "varint", "source_index")] = accepted
  ordered = []
  for case in cases:
    if case.source_line not in answers:
      raise cursor_grok_ranker.RankerError(
          f"AJIMEE answers are missing source index {case.source_line}"
      )
    ordered.append(answers[case.source_line])
  return tuple(ordered), actual_sha256


def load_development_inputs(config, root):
  cases, frozen_sha256 = load_frozen_corpus(
      Path(root) / config["frozen_corpus_path"],
      config["frozen_corpus_sha256"],
      config["expected_case_count"],
  )
  answers, answer_sha256 = load_answers(
      Path(root) / config["answer_source_path"],
      config["answer_source_sha256"],
      config["conversion_expected_command"],
      cases,
  )
  baseline_correct = sum(
      case.mozc_baseline_output in accepted
      for case, accepted in zip(cases, answers)
  )
  if baseline_correct != config["expected_baseline_correct_count"]:
    raise cursor_grok_ranker.RankerError("baseline correct count mismatch")
  return cases, answers, frozen_sha256, answer_sha256
