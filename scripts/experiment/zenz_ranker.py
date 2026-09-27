#!/usr/bin/env python3

from dataclasses import dataclass

import cursor_grok_ranker


HIRAGANA_START = 0x3041
HIRAGANA_END = 0x3096
HIRAGANA_ITERATION_MARKS = (0x309D, 0x309E)
KATAKANA_OFFSET = 0x60


@dataclass(frozen=True)
class CalibrationCell:
  copy_penalty: float
  order_prior: float
  top_k: int | None


def to_katakana(text):
  return "".join(
      chr(ord(character) + KATAKANA_OFFSET)
      if HIRAGANA_START <= ord(character) <= HIRAGANA_END
      or ord(character) in HIRAGANA_ITERATION_MARKS
      else character
      for character in text
  )


def normalize_for_model(text):
  return text.replace(" ", "　").replace("\n", "")


def build_prompt(config, request):
  return normalize_for_model(
      config["left_context_tag"]
      + request.preceding_text
      + config["right_context_tag"]
      + request.following_text
      + config["input_tag"]
      + to_katakana(request.reading)
      + config["output_tag"]
  )


def selected_candidates(segment, top_k):
  rankable = cursor_grok_ranker.rankable_candidates(segment)
  return rankable if top_k is None else rankable[:top_k]


def _right_window(baseline_values, index, right_window):
  last = index == len(baseline_values) - 1
  if right_window == "full":
    return "".join(baseline_values[index + 1:]), True
  if right_window == "next":
    return ("", True) if last else (baseline_values[index + 1], False)
  if right_window == "none":
    return "", False
  raise cursor_grok_ranker.RankerError("right window is unknown")


def candidate_outputs(request, top_k, right_window):
  baseline_values = [segment.candidates[0].value for segment in request.segments]
  outputs = []
  for index, segment in enumerate(request.segments):
    prefix = normalize_for_model("".join(baseline_values[:index]))
    suffix, add_end = _right_window(baseline_values, index, right_window)
    for candidate in selected_candidates(segment, top_k):
      outputs.append(
          (
              segment.id,
              candidate.id,
              prefix,
              normalize_for_model(candidate.value + suffix),
              add_end,
          )
      )
  return tuple(outputs)


def reading_copies(segment):
  return {segment.key, to_katakana(segment.key)}


def calibrated_response(request, raw_scores, cell):
  orders = []
  for segment in request.segments:
    scored = selected_candidates(segment, cell.top_k)
    if not scored:
      continue
    copies = reading_copies(segment)
    penalize_copies = segment.candidates[0].value not in copies
    calibrated = []
    for index, candidate in enumerate(scored):
      score = raw_scores[(segment.id, candidate.id)] - cell.order_prior * index
      if penalize_copies and candidate.value in copies:
        score -= cell.copy_penalty
      calibrated.append((-score, index, candidate.id))
    orders.append(
        cursor_grok_ranker.RankerSegmentOrder(
            segment_id=segment.id,
            candidate_ids=tuple(item[2] for item in sorted(calibrated)),
        )
    )
  return cursor_grok_ranker.RankerResponse(
      token=request.token, segment_orders=tuple(orders)
  )


def score_request(config, score_outputs, request, top_k, right_window):
  outputs = candidate_outputs(request, top_k, right_window)
  prompt_milliseconds, results = score_outputs(
      build_prompt(config, request),
      [(prefix, body, add_end) for _, _, prefix, body, add_end in outputs],
  )
  if len(results) != len(outputs):
    raise cursor_grok_ranker.RankerError("scorer returned a wrong number of scores")
  return prompt_milliseconds, tuple(
      (segment_id, candidate_id, score, milliseconds)
      for (segment_id, candidate_id, *_), (score, milliseconds) in zip(
          outputs, results
      )
  )


def request_milliseconds(request, prompt_milliseconds, candidate_scores, cell):
  scored = set()
  for segment in request.segments:
    scored.update(
        (segment.id, candidate.id)
        for candidate in selected_candidates(segment, cell.top_k)
    )
  return prompt_milliseconds + sum(
      milliseconds
      for segment_id, candidate_id, _, milliseconds in candidate_scores
      if (segment_id, candidate_id) in scored
  )
