#!/usr/bin/env python3

import hashlib
import struct
import time
import urllib.request
from pathlib import Path

import numpy as np

import cursor_grok_ranker


GGUF_MAGIC = b"GGUF"
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_ALIGNMENT_KEY = "general.alignment"
GGUF_DEFAULT_ALIGNMENT = 32


def _sha256(data):
  return hashlib.sha256(data).hexdigest()


def _read_string(data, position):
  length = struct.unpack_from("<Q", data, position)[0]
  start = position + 8
  return data[start:start + length].decode("utf-8"), start + length


def _skip_value(data, value_type, position):
  if value_type == GGUF_STRING:
    return _read_string(data, position)[1]
  if value_type == GGUF_ARRAY:
    element_type, count = struct.unpack_from("<IQ", data, position)
    position += 12
    for _ in range(count):
      position = _skip_value(data, element_type, position)
    return position
  return position + GGUF_SCALAR_SIZES[value_type]


def replace_gguf_string(data, key, expected, replacement):
  if data[:4] != GGUF_MAGIC:
    raise cursor_grok_ranker.RankerError("model file is not GGUF")
  tensor_count, kv_count = struct.unpack_from("<QQ", data, 8)
  position = 24
  output = bytearray(data[:24])
  alignment = GGUF_DEFAULT_ALIGNMENT
  replaced = False
  for _ in range(kv_count):
    start = position
    name, position = _read_string(data, position)
    value_type = struct.unpack_from("<I", data, position)[0]
    value_start = position + 4
    position = _skip_value(data, value_type, value_start)
    if name == key:
      if value_type != GGUF_STRING or _read_string(data, value_start)[0] != expected:
        raise cursor_grok_ranker.RankerError("GGUF key does not hold the expected value")
      encoded = replacement.encode("utf-8")
      output += data[start:value_start] + struct.pack("<Q", len(encoded)) + encoded
      replaced = True
    else:
      output += data[start:position]
    if name == GGUF_ALIGNMENT_KEY:
      alignment = struct.unpack_from("<I", data, value_start)[0]
  if not replaced:
    raise cursor_grok_ranker.RankerError("GGUF key is missing")
  for _ in range(tensor_count):
    start = position
    _, position = _read_string(data, position)
    dimensions = struct.unpack_from("<I", data, position)[0]
    position += 4 + 8 * dimensions + 4 + 8
    output += data[start:position]
  tensor_data = position + (-position) % alignment
  output += b"\0" * ((-len(output)) % alignment)
  output += data[tensor_data:]
  return bytes(output)


def prepare_model(config, root, name):
  spec = config["models"][name]
  source = Path(root) / spec["path"]
  if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(spec["url"], source)
  data = source.read_bytes()
  if len(data) != spec["size"] or _sha256(data) != spec["sha256"]:
    raise cursor_grok_ranker.RankerError(
        "model file does not match the pinned size and SHA256"
    )
  runtime = replace_gguf_string(
      data,
      config["pre_tokenizer_key"],
      config["source_pre_tokenizer"],
      config["runtime_pre_tokenizer"],
  )
  if _sha256(runtime) != spec["runtime_sha256"]:
    raise cursor_grok_ranker.RankerError("runtime model SHA256 mismatch")
  target = Path(root) / spec["runtime_path"]
  target.write_bytes(runtime)
  return target


def load_model(config, root, name):
  from llama_cpp import Llama

  spec = config["models"][name]
  path = Path(root) / spec["runtime_path"]
  if _sha256(path.read_bytes()) != spec["runtime_sha256"]:
    raise cursor_grok_ranker.RankerError("runtime model SHA256 mismatch")
  model = Llama(
      model_path=str(path),
      n_ctx=config["context_size"],
      n_threads=config["thread_count"],
      logits_all=True,
      verbose=False,
  )
  end_tokens = model.tokenize(
      config["end_token_piece"].encode("utf-8"), add_bos=False, special=True
  )
  if len(end_tokens) != 1:
    raise cursor_grok_ranker.RankerError("end token piece is not a single token")
  return model, end_tokens[0]


def _tokenize(model, text):
  return model.tokenize(text.encode("utf-8"), add_bos=False, special=False)


def _log_probability(model, start, tokens):
  logits = np.asarray(
      model.scores[start - 1:start + len(tokens) - 1], dtype=np.float64
  )
  maximum = logits.max(axis=1, keepdims=True)
  log_normalizer = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
  picked = logits[np.arange(len(tokens)), tokens]
  return float((picked - log_normalizer).sum())


def make_scorer(model, end_token):
  def score_outputs(prompt, items):
    started = time.perf_counter()
    prompt_tokens = _tokenize(model, prompt)
    model.reset()
    model.eval(prompt_tokens)
    prompt_milliseconds = (time.perf_counter() - started) * 1000
    current_prefix = None
    start = len(prompt_tokens)
    results = []
    for prefix, body, add_end in items:
      started = time.perf_counter()
      if prefix != current_prefix:
        prefix_tokens = _tokenize(model, prefix)
        model.n_tokens = len(prompt_tokens)
        if prefix_tokens:
          model.eval(prefix_tokens)
        start = len(prompt_tokens) + len(prefix_tokens)
        current_prefix = prefix
      body_tokens = _tokenize(model, body) + ([end_token] if add_end else [])
      model.n_tokens = start
      model.eval(body_tokens)
      results.append(
          (
              _log_probability(model, start, body_tokens),
              (time.perf_counter() - started) * 1000,
          )
      )
    return prompt_milliseconds, results

  return score_outputs
