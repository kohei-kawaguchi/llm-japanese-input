#!/usr/bin/env python3

import json
from dataclasses import dataclass


class RankerError(Exception):
  pass


@dataclass(frozen=True)
class RankerToken:
  session_generation: int
  state_revision: int
  request_sequence: int


@dataclass(frozen=True)
class RankerCandidate:
  id: int
  key: str
  value: str
  cost: int
  attributes: int
  consumed_key_size: int
  is_protected: bool


@dataclass(frozen=True)
class RankerSegment:
  id: int
  key: str
  candidates: tuple[RankerCandidate, ...]


@dataclass(frozen=True)
class RankerRequest:
  token: RankerToken
  mode: int
  preceding_text: str
  following_text: str
  reading: str
  focused_segment_id: int
  segments: tuple[RankerSegment, ...]


@dataclass(frozen=True)
class RankerSegmentOrder:
  segment_id: int
  candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class RankerResponse:
  token: RankerToken
  segment_orders: tuple[RankerSegmentOrder, ...]


@dataclass(frozen=True)
class FrozenCase:
  source_line: int
  conversion_reading: str
  mozc_baseline_output: str
  request: RankerRequest


def mode_name(config, mode):
  key = str(mode)
  if key not in config["modes"]:
    raise RankerError("candidate ranking mode is unknown")
  return config["modes"][key]


def rankable_candidates(segment):
  return tuple(
      candidate for candidate in segment.candidates if not candidate.is_protected
  )


def build_prompt(config, request):
  payload = {
      "prompt_version": config["prompt_version"],
      "token": {
          "session_generation": request.token.session_generation,
          "state_revision": request.token.state_revision,
          "request_sequence": request.token.request_sequence,
      },
      "mode": mode_name(config, request.mode),
      "reading": request.reading,
      "preceding_text": request.preceding_text,
      "following_text": request.following_text,
      "focused_segment_id": request.focused_segment_id,
      "segments": [],
  }
  for segment in request.segments:
    payload["segments"].append(
        {
            "id": segment.id,
            "key": segment.key,
            "candidates": [
                {
                    "id": candidate.id,
                    "key": candidate.key,
                    "value": candidate.value,
                    "cost": candidate.cost,
                }
                for candidate in rankable_candidates(segment)
            ],
        }
    )
  body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
  return (
      "Reorder Mozc conversion candidates. "
      "Use only the candidate identifiers in the request. "
      "Do not invent strings. "
      "Do not inspect files. "
      "Do not edit files. "
      "Do not call tools. "
      "Reply with one JSON object and nothing else. "
      "The object must have exactly these keys: token, segment_orders. "
      "token must copy the request token fields as integers. "
      "segment_orders must be an array of objects. "
      "Each object must have exactly these keys: segment_id, candidate_ids. "
      "segment_id is the request segment id. "
      "candidate_ids is an array of integers, every rankable candidate "
      "identifier for that segment exactly once, best first. "
      "Do not use a nested array in place of those objects. "
      "Example shape: "
      "{\"token\":{\"session_generation\":1,\"state_revision\":2,"
      "\"request_sequence\":3},\"segment_orders\":[{\"segment_id\":0,"
      "\"candidate_ids\":[2,0,1]}]}.\n"
      f"{body}"
  )


def _require_int(value, label):
  if type(value) is not int:
    raise RankerError(f"{label} must be an integer")
  return value


def _require_int_list(value, label):
  if not isinstance(value, list) or not value:
    raise RankerError(f"{label} must be a non-empty list")
  return tuple(_require_int(item, f"{label}[{index}]") for index, item in enumerate(value))


def _parse_token(value):
  if not isinstance(value, dict):
    raise RankerError("response token must be an object")
  required = (
      "session_generation",
      "state_revision",
      "request_sequence",
  )
  if set(value) != set(required):
    raise RankerError("response token keys are invalid")
  return RankerToken(
      session_generation=_require_int(
          value["session_generation"], "token.session_generation"
      ),
      state_revision=_require_int(value["state_revision"], "token.state_revision"),
      request_sequence=_require_int(
          value["request_sequence"], "token.request_sequence"
      ),
  )


def parse_response(request, raw_text):
  if not isinstance(raw_text, str) or not raw_text:
    raise RankerError("ranker reply is empty")
  parsed = json.loads(raw_text)
  if not isinstance(parsed, dict) or set(parsed) != {"token", "segment_orders"}:
    raise RankerError("ranker reply is not the required structure")
  token = _parse_token(parsed["token"])
  if token != request.token:
    raise RankerError("response token does not match the request token")
  if not isinstance(parsed["segment_orders"], list):
    raise RankerError("segment_orders must be a list")
  request_segments = {segment.id: segment for segment in request.segments}
  seen_segments = []
  orders = []
  for index, item in enumerate(parsed["segment_orders"]):
    if not isinstance(item, dict) or set(item) != {"segment_id", "candidate_ids"}:
      raise RankerError(f"segment_orders[{index}] is not the required structure")
    segment_id = _require_int(item["segment_id"], f"segment_orders[{index}].segment_id")
    if segment_id in seen_segments:
      raise RankerError("candidate ranking references a segment more than once")
    if segment_id not in request_segments:
      raise RankerError("candidate ranking references an unknown segment")
    seen_segments.append(segment_id)
    segment = request_segments[segment_id]
    candidate_ids = _require_int_list(
        item["candidate_ids"], f"segment_orders[{index}].candidate_ids"
    )
    expected = tuple(candidate.id for candidate in rankable_candidates(segment))
    seen_candidates = []
    for candidate_id in candidate_ids:
      if candidate_id in seen_candidates:
        raise RankerError("candidate ranking references a candidate more than once")
      if candidate_id not in expected:
        if any(candidate.id == candidate_id for candidate in segment.candidates):
          raise RankerError("candidate ranking references a protected candidate")
        raise RankerError("candidate ranking references an unknown candidate")
      seen_candidates.append(candidate_id)
    if tuple(sorted(candidate_ids)) != tuple(sorted(expected)):
      raise RankerError("candidate ranking omits a request candidate")
    orders.append(RankerSegmentOrder(segment_id=segment_id, candidate_ids=candidate_ids))
  if set(seen_segments) != set(request_segments):
    raise RankerError("candidate ranking omits a request segment")
  return RankerResponse(token=token, segment_orders=tuple(orders))


def merge_response(request, response):
  if response.token != request.token:
    raise RankerError("response token does not match the request token")
  request_segments = {segment.id: index for index, segment in enumerate(request.segments)}
  merged = [
      RankerSegmentOrder(
          segment_id=segment.id,
          candidate_ids=tuple(candidate.id for candidate in segment.candidates),
      )
      for segment in request.segments
  ]
  seen_segments = []
  for segment_order in response.segment_orders:
    if segment_order.segment_id in seen_segments:
      raise RankerError("candidate ranking references a segment more than once")
    if segment_order.segment_id not in request_segments:
      raise RankerError("candidate ranking references an unknown segment")
    seen_segments.append(segment_order.segment_id)
    segment = request.segments[request_segments[segment_order.segment_id]]
    seen_candidates = []
    ranked = []
    candidates_by_id = {candidate.id: candidate for candidate in segment.candidates}
    for candidate_id in segment_order.candidate_ids:
      if candidate_id not in candidates_by_id:
        raise RankerError("candidate ranking references an unknown candidate")
      if candidate_id in seen_candidates:
        raise RankerError("candidate ranking references a candidate more than once")
      if candidates_by_id[candidate_id].is_protected:
        raise RankerError("candidate ranking references a protected candidate")
      seen_candidates.append(candidate_id)
      ranked.append(candidate_id)
    for candidate in segment.candidates:
      if not candidate.is_protected and candidate.id not in seen_candidates:
        ranked.append(candidate.id)
    restored = []
    rankable_index = 0
    for candidate in segment.candidates:
      if candidate.is_protected:
        restored.append(candidate.id)
      else:
        restored.append(ranked[rankable_index])
        rankable_index += 1
    merged[request_segments[segment.id]] = RankerSegmentOrder(
        segment_id=segment.id,
        candidate_ids=tuple(restored),
    )
  return tuple(merged)


def merged_top_output(request, merged_orders):
  values = []
  top_by_segment = {}
  orders = {order.segment_id: order for order in merged_orders}
  for segment in request.segments:
    if segment.id not in orders:
      raise RankerError("merged order omits a request segment")
    top_id = orders[segment.id].candidate_ids[0]
    candidate = next(item for item in segment.candidates if item.id == top_id)
    values.append(candidate.value)
    top_by_segment[segment.id] = top_id
  return "".join(values), top_by_segment
