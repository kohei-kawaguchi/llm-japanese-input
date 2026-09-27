#!/usr/bin/env python3

import os
import time

import httpx

import cursor_grok_ranker


class ListedParameterValue:
  def __init__(self, value):
    self.value = value


class ListedParameter:
  def __init__(self, parameter_id, values):
    self.id = parameter_id
    self.values = values


class ListedModel:
  def __init__(self, model_id, parameters):
    self.id = model_id
    self.parameters = parameters


def require_api_key(config):
  name = config["api_key_environment"]
  if name not in os.environ or not os.environ[name]:
    raise cursor_grok_ranker.RankerError(f"{name} is required")
  return os.environ[name]


def _contains_all(value, parts):
  return all(part in value for part in parts)


def _contains_any(value, parts):
  return any(part in value for part in parts)


def select_model(config, models):
  selection = config["model_selection"]
  matches = []
  for model in models:
    model_id = model.id
    if _contains_all(model_id, selection["id_required_substrings"]) and not (
        _contains_any(model_id, selection["id_forbidden_substrings"])
    ):
      matches.append(model)
  if len(matches) != 1:
    raise cursor_grok_ranker.RankerError(
        "cursor grok 4.6 high model id is not unique"
    )
  model = matches[0]
  values_by_id = {
      parameter.id: {item.value for item in parameter.values}
      for parameter in model.parameters
  }
  params = []
  for required in selection["required_params"]:
    if required["id"] not in values_by_id:
      raise cursor_grok_ranker.RankerError(
          "cursor grok 4.6 high parameter is missing"
      )
    if required["value"] not in values_by_id[required["id"]]:
      raise cursor_grok_ranker.RankerError(
          "cursor grok 4.6 high parameter value is missing"
      )
    params.append({"id": required["id"], "value": required["value"]})
  return {"id": model.id, "params": params}


def _retry_after_seconds(response, attempt):
  retry_after = response.headers.get("Retry-After")
  if retry_after:
    return float(retry_after)
  return 2 ** attempt


def _cursor_request(config, api_key, method, url, body=None):
  last_response = None
  for attempt in range(config["retry_limit"]):
    response = httpx.request(
        method,
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=config["request_timeout_seconds"],
    )
    if response.status_code == 429 and attempt + 1 < config["retry_limit"]:
      last_response = response
      time.sleep(_retry_after_seconds(response, attempt))
      continue
    if response.status_code >= 400:
      raise cursor_grok_ranker.RankerError(
          f"cursor api status {response.status_code}"
      )
    return response.json()
  raise cursor_grok_ranker.RankerError(
      f"cursor api status {last_response.status_code}"
  )


def list_models(config, api_key):
  payload = _cursor_request(config, api_key, "GET", config["models_list_url"])
  if "items" not in payload or not isinstance(payload["items"], list):
    raise cursor_grok_ranker.RankerError("cursor models list is invalid")
  models = []
  for item in payload["items"]:
    parameters = []
    for parameter in item.get("parameters", []):
      values = [
          ListedParameterValue(value["value"])
          for value in parameter.get("values", [])
      ]
      parameters.append(ListedParameter(parameter["id"], values))
    models.append(ListedModel(item["id"], parameters))
  return models


def rank_request(config, request, model, api_key):
  if config["runtime"] != "cloud":
    raise cursor_grok_ranker.RankerError("cursor grok ranker runtime must be cloud")
  prompt = cursor_grok_ranker.build_prompt(config, request)
  created = _cursor_request(
      config,
      api_key,
      "POST",
      config["agents_url"],
      {
          "prompt": {"text": prompt},
          "name": config["agent_name"],
          "model": {
              "id": model["id"],
              "params": model["params"],
          },
          "repos": [
              {"url": repo["url"], "startingRef": repo["starting_ref"]}
              for repo in config["cloud_repos"]
          ],
          "autoCreatePR": False,
      },
  )
  if "agent" not in created or "run" not in created:
    raise cursor_grok_ranker.RankerError("cursor agent create response is invalid")
  agent_id = created["agent"]["id"]
  run_id = created["run"]["id"]
  identity = {"agent_id": agent_id, "run_id": run_id}
  run_url = f"{config['agents_url']}/{agent_id}/runs/{run_id}"
  deadline = time.monotonic() + config["run_timeout_seconds"]
  while time.monotonic() < deadline:
    run = _cursor_request(config, api_key, "GET", run_url)
    status = run.get("status")
    if status == "FINISHED":
      text = run.get("result")
      response = cursor_grok_ranker.parse_response(request, text)
      merged = cursor_grok_ranker.merge_response(request, response)
      return identity, response, merged
    if status in ("ERROR", "CANCELLED", "EXPIRED"):
      raise cursor_grok_ranker.RankerError("cursor grok ranker run failed")
    time.sleep(config["poll_interval_seconds"])
  raise cursor_grok_ranker.RankerError("cursor grok ranker run timed out")
