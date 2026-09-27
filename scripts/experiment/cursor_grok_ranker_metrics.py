#!/usr/bin/env python3

import cursor_grok_ranker


def levenshtein(left, right):
  left_scalars = tuple(left)
  right_scalars = tuple(right)
  if len(left_scalars) < len(right_scalars):
    left_scalars, right_scalars = right_scalars, left_scalars
  previous = list(range(len(right_scalars) + 1))
  for left_index, left_scalar in enumerate(left_scalars, start=1):
    current = [left_index]
    for right_index, right_scalar in enumerate(right_scalars, start=1):
      current.append(
          min(
              current[-1] + 1,
              previous[right_index] + 1,
              previous[right_index - 1] + (left_scalar != right_scalar),
          )
      )
    previous = current
  return previous[-1]


def exact_bootstrap_endpoints(win_count, regression_count, tie_count):
  sample_size = win_count + regression_count + tie_count
  if sample_size <= 0:
    raise cursor_grok_ranker.RankerError("bootstrap sample is empty")
  mass = {0: 1}
  for _ in range(sample_size):
    next_mass = {}
    for value, count in mass.items():
      for delta, multiplier in (
          (-1, regression_count),
          (0, tie_count),
          (1, win_count),
      ):
        if multiplier:
          next_mass[value + delta] = (
              next_mass.get(value + delta, 0) + count * multiplier
          )
    mass = next_mass
  total_mass = sample_size ** sample_size
  cumulative = 0
  lower = None
  upper = None
  for value in sorted(mass):
    cumulative += mass[value]
    if lower is None and 40 * cumulative >= total_mass:
      lower = value
    if upper is None and 40 * cumulative >= 39 * total_mass:
      upper = value
  if lower is None or upper is None:
    raise cursor_grok_ranker.RankerError("bootstrap quantiles are incomplete")
  return lower, upper


def clean_key_copy_promotions(request, top_by_segment):
  baseline_left = []
  result = set()
  for segment in request.segments:
    candidates_by_id = {candidate.id: candidate for candidate in segment.candidates}
    baseline_value = segment.candidates[0].value
    model_value = candidates_by_id[top_by_segment[segment.id]].value
    selected_values = {candidate.value for candidate in segment.candidates}
    clean_context = request.preceding_text + "".join(baseline_left)
    if (
        baseline_value != segment.key
        and model_value == segment.key
        and segment.key in selected_values
        and segment.key not in clean_context
    ):
      result.add(segment.id)
    baseline_left.append(baseline_value)
  return frozenset(result)


def has_exact_key_regression(request, top_by_segment):
  for segment in request.segments:
    candidates_by_id = {candidate.id: candidate for candidate in segment.candidates}
    if (
        segment.candidates[0].value != segment.key
        and candidates_by_id[top_by_segment[segment.id]].value == segment.key
    ):
      return True
  return False


def evaluate_cases(config, cases, answers, model_outputs, top_by_case):
  if len(cases) != len(answers) or len(cases) != len(model_outputs):
    raise cursor_grok_ranker.RankerError("metric case counts do not align")
  if len(cases) != config["expected_case_count"]:
    raise cursor_grok_ranker.RankerError("metric case count mismatch")
  baseline_correct_count = 0
  model_correct_count = 0
  retained_correct_count = 0
  win_count = 0
  regression_count = 0
  unchanged_incorrect_count = 0
  exact_key_regression_count = 0
  clean_key_copy_promotion_count = 0
  baseline_edit = 0
  model_edit = 0
  reference_scalars = 0
  for case, accepted, model_output, top_by_segment in zip(
      cases, answers, model_outputs, top_by_case
  ):
    baseline_correct = case.mozc_baseline_output in accepted
    model_correct = model_output in accepted
    baseline_correct_count += baseline_correct
    model_correct_count += model_correct
    if baseline_correct and model_correct:
      retained_correct_count += 1
    elif not baseline_correct and model_correct:
      win_count += 1
    elif baseline_correct and not model_correct:
      regression_count += 1
    else:
      unchanged_incorrect_count += 1
    if (
        baseline_correct
        and not model_correct
        and has_exact_key_regression(case.request, top_by_segment)
    ):
      exact_key_regression_count += 1
    clean_key_copy_promotion_count += len(
        clean_key_copy_promotions(case.request, top_by_segment)
    )
    baseline_edit += min(
        levenshtein(case.mozc_baseline_output, answer) for answer in accepted
    )
    model_edit += min(levenshtein(model_output, answer) for answer in accepted)
    reference_scalars += len(tuple(accepted[0]))
  if baseline_correct_count != config["expected_baseline_correct_count"]:
    raise cursor_grok_ranker.RankerError("baseline correct count mismatch")
  tie_count = len(cases) - win_count - regression_count
  lower, upper = exact_bootstrap_endpoints(win_count, regression_count, tie_count)
  net_gain = win_count - regression_count
  gate_passed = (
      net_gain >= config["minimum_net_gain_cases"]
      and lower > 0
      and regression_count <= config["maximum_baseline_correct_regressions"]
      and clean_key_copy_promotion_count == 0
      and exact_key_regression_count == 0
  )
  return {
      "case_count": len(cases),
      "baseline_correct_count": baseline_correct_count,
      "model_correct_count": model_correct_count,
      "retained_correct_count": retained_correct_count,
      "win_count": win_count,
      "regression_count": regression_count,
      "unchanged_incorrect_count": unchanged_incorrect_count,
      "mozc_correct_exact_key_regression_count": exact_key_regression_count,
      "clean_key_copy_promotion_count": clean_key_copy_promotion_count,
      "baseline_character_error_sum": baseline_edit,
      "model_character_error_sum": model_edit,
      "reference_unicode_scalar_count": reference_scalars,
      "net_gain_cases": net_gain,
      "paired_bootstrap": {
          "definition_version": config["exact_bootstrap_definition_version"],
          "sample_size": len(cases),
          "win_count": win_count,
          "regression_count": regression_count,
          "tie_count": tie_count,
          "lower_gain_cases": lower,
          "upper_gain_cases": upper,
          "gain_denominator_cases": len(cases),
      },
      "development_gate_passed": gate_passed,
  }
