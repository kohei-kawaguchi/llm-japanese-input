# Reorder Mozc candidates with Cursor Grok 4.6 High

## Problem

Stage 9 still has no production ranking objective. The local llama.cpp backends
score each Mozc candidate by continuation log probability and then sort those
scores. On the frozen development corpus that method promoted exact key copies
and lost to Mozc. Issue 1 asked for a larger local causal model and a calibrated
score grid. This issue replaces that local scoring path for candidate reordering
with Cursor Grok 4.6 High through the Cursor API.

The existing product seam stays. Mozc still creates candidates. The ranker still
returns an order of existing candidate identifiers. The merger still validates
the complete response and keeps unmentioned candidates in their original
relative order. Protected candidates still stay in their original slots.

## Why the Cursor API, not a local model

The Cursor API that can name Grok 4.6 High is the Cloud Agents API and the
Cursor SDK. Official Cursor docs state that this surface runs agent workflows.
It is not a chat completions or logit API. There is no documented way to read
token log probabilities from it.

Therefore this ranker cannot reuse the llama.cpp continuation objective. It
asks Grok 4.6 High to return a permutation of the request candidate identifiers
from the same immutable `CandidateRankerRequest` fields the local ranker already
receives: mode, reading, preceding text, following text, segment keys, and the
candidate identifiers, keys, values, costs, attributes, consumed key lengths,
and protection flags.

The model identifier is Cursor Grok 4.6 High. The exact API model id and
reasoning effort parameter must be read from `GET https://api.cursor.com/v1/models`
with the same `CURSOR_API_KEY` used at runtime, then stored in the experiment
config. Do not guess a slug. The named default effort for Grok 4.6 in Cursor is
high. Do not use the Fast variant unless a later issue changes the config.

## Placement

This is an experiment. Put new files under `scripts/experiment/`,
`scripts/experiment/config/`, and `report/experiment/`. Do not edit
`src/engine/llama_candidate_ranker.cc`, `src/engine/engine.cc`, the quality
regression score pipeline, or `.cursor/plans/ime-architecture.md` in this
issue.

Reuse the already frozen development ranker requests and answers. Do not rebuild
the Mozc corpus.

## Ranking contract

Each call maps one frozen `CandidateRankerRequest` to one
`CandidateRankerResponse`.

The prompt contains only request fields. It does not ask the model to invent
strings. The required reply is structured text that lists, for every request
segment, that segment identifier and a permutation of that segment's candidate
identifiers.

After the API returns, parse the structured reply and reject it when any of
these fail:

- the reply is not the required structure
- a segment or candidate identifier is unknown
- a segment or candidate identifier is repeated
- a request segment is omitted
- a request candidate is omitted
- the response token does not match the request token

Do not repair a bad reply. Do not fall back to Mozc order inside the ranker. A
failed case stays a failed case. Apply
`BuildCandidateRankerMergeOrder` rules only to a valid response so protected
candidates and any later Engine merge stay consistent with the main contract.

Omit protected candidates from the model prompt. Put them back in their original
slots after a valid reply, using the same protection attributes the frozen
request already stores.

## Cursor call shape

Use the official Python Cursor SDK (`cursor-sdk`) because the experiment scripts
are already Python and the IME tree has no TypeScript package.

Use a one shot `Agent.prompt` with an explicit API key and an explicit model
selection. Set the runtime explicitly. Prefer a cloud agent with no repository
so the agent cannot edit this tree. If the account cannot create a cloud agent,
use a local agent whose `cwd` is an empty dedicated work directory under
`.tools/cache/experiment/cursor-grok-ranker/`, with no project setting sources
and no MCP servers.

Disable extra agent work in the prompt: return only the structured identifier
order, do not inspect files, and do not call tools. If a run status is `error`
or a `CursorAgentError` is raised, fail that case. Honor `is_retryable` and
`retry_after` only for transport startup failures, with a finite retry count
from config. Do not retry a finished run that returned an unparsable body.

`CURSOR_API_KEY` is required. If it is missing, stop. Do not read a default
model, do not switch to a local GGUF, and do not skip cases silently.

Record `agent.agent_id` and `run.id` for every call before waiting.

## Outputs that the local score pipeline cannot consume

The current quality regression score artifacts store deterministic continuation
values and require byte identical repeats. Grok replies are not a log
probability table and are not byte identical across runs. Do not write Grok
orders into the llama.cpp score proto or reuse those SHA256 gates.

Write an experiment report that stores, for every case:

- frozen corpus and answer corpus SHA256
- model id, model params, prompt version, and SDK or API revision
- request token
- parsed segment orders, or the typed failure
- merged top output after the existing merge rules
- elapsed milliseconds
- agent id and run id

Run the same corpus twice. Compare case level orders and gate metrics. Record
disagreement. Do not require byte identical report files.

Apply the existing development quality gates to the merged top output against
the frozen answers: wins, regressions, net gain, exact paired bootstrap lower
gain, retention, character error, and Mozc correct exact key copy regressions.
Do not inspect holdout or AJIMEE answers.

## Privacy and latency

Each request sends composition text and candidate strings to Cursor. Password
fields remain excluded because the frozen development corpus is built from the
same privacy rules as the Engine seam. Do not log prompt text, candidate text,
or raw API bodies to shared operational logs.

Cursor agent latency is far above the IME `max_wait_millisec` budget. This
issue does not wire the backend into `Engine` and does not change live TSF
behavior.

## Implementation stages

1. Add `scripts/experiment/config/cursor_grok_ranker.json` with the API key
   environment name, confirmed model id, effort params, prompt version, retry
   limit, cache directory, frozen corpus path, answer corpus path, and output
   directory. No literals for those values in the runner.
2. Confirm the Grok 4.6 High model id and params with `Cursor.models.list()`
   and write the confirmed values into that config.
3. Add a prompt builder and a strict parser with unit tests over synthetic
   frozen requests. Cover unknown ids, duplicates, omissions, protected
   candidate restoration, and token mismatch.
4. Add a ranker that calls `Agent.prompt` once per request and returns either a
   `CandidateRankerResponse` or a typed error.
5. Add a runner that reads the existing frozen development corpus, calls the
   ranker, applies merge rules, and writes the experiment report.
6. Run the development corpus twice. Compute the development gates. If no
   configuration passes, record the failure mechanism and stop.

## Out of scope

Do not replace `LlamaCandidateRanker` in `Engine`.
Do not change branding, packaging, or Windows TSF tests.
Do not submit HPC4 jobs.
Do not edit the architecture draft.
Do not open holdout, AJIMEE, quantization, or latency production work unless
the development gates pass and a later issue asks for those steps.
