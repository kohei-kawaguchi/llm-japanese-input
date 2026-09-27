# Rank Mozc candidates with a pretrained kana-kanji conversion model

## Problem

Issue 6 assumed that no local model had a trained conversion objective, so the
ranker had to be fine-tuned from corpus sentences. Every local model tried so
far was a general causal language model scored zero-shot on continuation log
probability, and each lost to Mozc on the frozen development corpus. The
recorded root cause is relevance accuracy.

A pretrained model with the missing objective already exists. The zenz models
from the Zenzai neural kana-kanji conversion engine (AzooKeyKanaKanjiConverter,
MIT) are GPT-2 conditional language models trained specifically for kana-kanji
conversion. zenz-v3.2 reads the prompt
`<left context><right context><katakana reading><output></s>`,
so it models the probability of an output given its reading and both sides of
context, which is exactly the quantity the ranker needs.
`Miwa-Keita/zenz-v3.2-small-gguf` is a 74 MB Q5_K_M GGUF under Apache 2.0,
which the pinned llama.cpp runtime family can load. Training a model is
therefore deferred until this pretrained model has been measured.

## Scoring objective

For each request segment and each rankable candidate, build the output string
from the baseline values of the segments to its left, the candidate value, and
the baseline values of the segments to its right. The katakana reading is the
complete composer reading converted to katakana. The left context is the
request preceding text and the right context is the request following text.
The score is the sum of the model log probabilities of the output tokens and
the end token after ``. Candidates are ordered by that score, and the
existing merge rules restore protected candidates and unmentioned candidates.

## Evaluation

Reuse the frozen development corpus and answers and the existing experiment
metrics under `scripts/experiment/`. Report wins, regressions, net gain, exact
paired bootstrap lower gain, retention, character error, clean key copy
promotions, and Mozc-correct exact key regressions next to the Mozc baseline
and both Grok runs. Measure CPU milliseconds per request. Do not inspect
holdout or AJIMEE answers.

## Uncalibrated result

The uncalibrated run scored 163 of 219 against the 164-correct Mozc
baseline, with 29 wins, 30 regressions, exact bootstrap lower gain -16, 12
Mozc-correct exact key regressions, and 23 clean key copy promotions. The
wins are semantic corrections. About 19 of the 30 regressions promote a
candidate that copies the reading in hiragana or katakana, which zenz prefers
when it is unsure of a word. The median request took 284 milliseconds and
the 90th percentile 2,122 milliseconds, because every candidate is scored.

## Calibration

The calibration is declared here before any calibrated result is seen. It is
chosen after the uncalibrated development result, so a passing cell still
requires holdout confirmation before any product work.

Raw zenz scores are computed once for every rankable candidate and cached
with their scoring time. For candidate index `i` in Mozc order within its
segment, the calibrated score is the raw score minus the order prior
coefficient times `i`, minus the reading copy penalty when the candidate value
equals the segment key or its katakana form and the Mozc top value equals
neither. Only the first `K` rankable candidates in Mozc order are scored and
reordered; the rest keep their Mozc order after them.

The grid is the reading copy penalty in {0, 5, 10, 100}, the order prior
coefficient in {0, 0.25, 0.5, 1.0, 2.0}, and `K` in {5, 10, all}, giving 60
cells. Each cell is judged by the unchanged development gate. Passing cells
are ordered by net gain, then by smaller `K`, smaller order prior, and smaller
penalty, and the first is selected. Each cell also reports the summed scoring
time of its candidates per request.

## Calibrated result

No cell passes the development gate. The strongest cell uses the reading copy
penalty 100, the order prior 1.0, and `K` = 10. It has 186 correct cases
against the 164-correct baseline, 28 wins, 6 regressions, net gain 22, exact
bootstrap lower gain 11, zero Mozc-correct exact key regressions, zero clean
key copy promotions, and model character error 59 against the baseline 115.
Its median request takes 236 milliseconds and the 90th percentile 752
milliseconds on eight CPU threads. It fails only the maximum three regression
gate. Of its six regressions, two differ from the answer only in digit width,
two are notation variants (百均 as 100均 and 乾き物 as 乾きもの), one is a
kanji variant (葦北郡 as 芦北郡), and one is a wrong conversion (先輩超え as
先輩声). The cell with the fewest regressions (penalty 100, order prior 2.0)
has 184 correct cases, 24 wins, and 4 regressions, and also fails that gate.

## Confirmation on holdout and AJIMEE

The confirmation is declared here before either result is seen. It runs the
strongest development cell, reading copy penalty 100, order prior 1.0, and
`K` = 10, once and does not revise any parameter from its result.

The holdout is the 72-case frozen corpus from `regression.tsv`, SHA256
`aec5be85738ada4fad68943fff3a0e556cdf61b587f93cfa9ca380164e61e812`, with
answers from `src/data/test/quality_regression_test/regression.tsv` under the
same rank-zero filter, and a Mozc baseline of 42 correct. Its gate is the
architecture holdout gate: at least two net additional correct outputs, an
exact paired bootstrap lower gain above zero, retention of all 42
Mozc-correct outputs, and zero Mozc-correct exact key regressions.

AJIMEE is the 200-case frozen corpus at AJIMEE-Bench commit
`401666cd56d1a570c2021798b64b6da4396bfd45`, whose requests carry the published
preceding context, with its accepted whole outputs, and a Mozc baseline of 103
correct. A case is correct when the output equals any accepted output, and
character error uses the nearest accepted output. Its gate is the architecture
quality gate that AJIMEE can close: at least a two point overall top-one gain
with the exact paired bootstrap lower bound above zero, at least 98 percent
retention of Mozc-correct cases, and no loss above two points in the context
or no-context slice. The frozen corpora are cached under
`.tools/cache/evaluation/` and are pinned by SHA256 in config.

## Confirmation result

On the holdout the selected cell has 53 of 72 correct against the 42-correct
baseline, 12 wins, 1 regression, net gain 11, exact bootstrap gain interval 5
through 18, zero Mozc-correct exact key regressions, and model character error
36 against the baseline 56 over 435 scalars. It fails only the full retention
requirement: `さそおう` changes from 誘おう to the Mozc candidate 支う.

On AJIMEE it has 145 of 200 correct against the 103-correct baseline, 43 wins,
1 regression, a 21 point gain with exact bootstrap gain interval 31 through
54 cases, 99 percent retention, and character error 130 against the baseline
251 over 4,220 scalars. The context slice rises from 46 to 63 of 100 and the
no-context slice from 57 to 82 of 100. It passes every AJIMEE gate that the
benchmark can close.

Latency is the remaining blocker. Scoring the top 10 candidates of every
segment as whole sentences on eight CPU threads takes a median of 452 and a
90th percentile of 870 milliseconds on the holdout, and a median of 1,778 and
a 90th percentile of 11,026 milliseconds on the longer AJIMEE requests.

## Latency

The speed experiment is declared here before any variant is measured. The
calibration stays fixed at reading copy penalty 100 and order prior 1.0.

Every variant shares computation across candidates. The prompt and the
baseline values of the segments to the left of the scored segment are
evaluated once per segment, and each candidate evaluates only its own value and
its right window. The left prefix adds the same log probability to every
candidate of a segment, so it is excluded from the score without changing any
order.

The variants are the cross product of the model, the right window, and `K`.
The models are zenz-v3.2-small (91M) and zenz-v3.2-xsmall (26M,
`Miwa-Keita/zenz-v3.2-xsmall-gguf`, Apache 2.0). The right window is `full`,
which scores the baseline values of all segments to the right followed by the
end token, `next`, which scores only the baseline value of the next segment or
the end token for the last segment, or `none`, which scores the candidate
value alone. `K` is 5 or 10. This gives 12 variants.

Each variant runs on the development corpus, which reports accuracy with the
development metrics, and on AJIMEE, whose long requests measure latency and
whose accuracy is reported but not used for selection. Latency is the wall
clock time of one request on eight CPU threads. The target is a 90th
percentile of at most 100 milliseconds on both corpora. Among variants that
meet it, the variant with the largest development net gain is selected, then
the smaller model, then the smaller `K`, then the shorter window.

## Latency result

Sharing the prompt and left prefix leaves every output unchanged: the small
model with the full window and `K` = 10 reproduces the 186-correct development
outputs exactly. No variant meets the 100 millisecond 90th percentile target,
so none is selected.

Accuracy barely depends on the window or `K`. The small model keeps 185 or
186 development cases and 145 to 148 AJIMEE cases in every variant, and the
`none` window has zero AJIMEE regressions. The xsmall model drops to 178 to 180
development and 130 to 131 AJIMEE cases. The fastest small variant (`next`,
`K` = 5) has a development median of 104 and a 90th percentile of 249
milliseconds, and an AJIMEE median of 506 and a 90th percentile of 1,743
milliseconds. The fastest xsmall variant (`next`, `K` = 5) has 58 and 137
milliseconds on development and 230 and 764 milliseconds on AJIMEE.

Latency now scales with the number of separate decode calls, one per segment
prefix and one per candidate, each run sequentially from Python. The next
reduction is to decode all candidates of a segment as parallel sequences in one
batch, and to measure it in the C++ runtime that the Engine backend would use.

## C++ batched backend

The production backend is a new `ZenzCandidateRanker` in `src/engine`, beside
and independent of `LlamaCandidateRanker`, implementing
`CandidateRankerBackendInterface` on the pinned llama.cpp overlay. Its
settings are fixed by the development selection: the small model, the `next`
window, `K` = 5, reading copy penalty 100, and order prior 1.0. It resolves the
end token from the piece `</s>`. The llama.cpp overlay carries
`llama_cpp_zenz_pretokenizer.patch`, which maps the pre-tokenizer name
`gpt2-small-japanese-char` to the GPT-2 pre-tokenizer exactly as the azooKey
fork does, so the backend loads the pinned upstream GGUF without the rewritten
copy that the Python experiment uses.

For each request it builds the same prompt as the experiment and decodes it
once as sequence 0. For each segment with rankable candidates, it decodes the
baseline left prefix on sequence 0, copies sequence 0 to sequences 1 through
`K`, and decodes every candidate body and its right window in one batch, one
sequence per candidate, requesting logits for each body position. The score
of a candidate is the sum of the log probabilities of its body tokens. It then
removes sequences 1 through `K` and truncates sequence 0 back to the prompt
before the next segment. Calibration and ordering are the experiment's.
Cancellation is checked before each decode.

A manual benchmark binary reads the development and AJIMEE frozen corpora,
ranks every request with the backend, and writes one JSON line per case with
the merged top output and the wall clock milliseconds. The Python experiment
computes the metrics from that file with its existing metric code. This stage
does not change `Engine`. Engine wiring follows once the C++ latency is known.

    ZenzCandidateRanker::Rank(request, cancellation):
      prompt = tag_left + preceding + tag_right + following
               + tag_input + katakana(reading) + tag_output
      clear memory; decode(tokens(prompt), seq 0, positions 0..)
      baseline = [segment.candidates[0].value for segment]
      for index, segment in segments:
        selected = first K unprotected candidates
        if selected is empty: continue
        check cancellation
        remove seq 0 positions >= prompt length
        prefix = tokens(join(baseline[:index]))
        if prefix: decode(prefix, seq 0, from prompt length)
        start = prompt length + prefix length
        suffix, add_end = next window of baseline at index
        for i, candidate in selected:
          copy seq 0 to seq i + 1
          body_i = tokens(candidate.value + suffix) + [end if add_end]
        batch = for each i, body_i tokens on seq i + 1 at start.., logits on
        decode(batch)
        for each i: score_i = sum log_softmax(logits before each body token)
        for each i: remove seq i + 1
        calibrated_i = score_i - order_prior * i
                       - copy_penalty if candidate is a reading copy and
                         the Mozc top is not
        order = selected ids by calibrated descending, then Mozc index
      return response with those orders

## C++ backend result

The C++ backend reproduces the selected accuracy: 186 of 219 development cases
(28 wins, 6 regressions) and 146 of 200 AJIMEE cases (44 wins, 1 regression).
The first build was slower than the Python runtime because
`bazel/BUILD.llama_cpp.bazel` compiled the ggml CPU backend with only `/O2`,
so the SSE2 baseline was used and the AVX2, FMA, and F16C kernels were
compiled out. The Windows overlay now compiles llama.cpp with `/arch:AVX2`,
which also changes the numerics of the rejected rinna ranker, and the build no
longer runs on x64 processors without AVX2.

With AVX2 on eight threads, the development median is 91 and the 90th
percentile 228 milliseconds, and the AJIMEE median is 382 and the 90th
percentile 1,447 milliseconds. Four threads are slightly slower, and sixteen
threads on this eight-core processor are more than twenty times slower, so the
backend uses eight threads. The 100 millisecond 90th percentile target is
still not met.

## Engine integration

The conversion server (`SessionServer`) creates the backend when
`zenz-v3.2-small-Q5_K_M.gguf` is present in the server directory and hands it
to its `Engine`, which replaces the backend of its `CandidateRankingService`.
Engines created elsewhere, including every test, keep no backend, so tests do
not depend on an installed model. Without the file, or on platforms
other than Windows, the service keeps no backend and ranking reports the
existing unavailable diagnostic. The model is fetched by `http_file` at its
pinned revision and SHA256 and installed beside `mozc_server.exe`, and the
installer credits include the llama.cpp MIT and zenz Apache 2.0 notices.
On Windows the default configuration enables `candidate_ranking_config`
with a 250 millisecond `max_wait_millisec`, and configuration normalization
adds the same section when a stored configuration has none, so a configuration
saved before ranking existed, or no configuration file, also ranks; when ranking takes longer, Mozc
order is shown. The settings dialog has a checkbox that turns ranking off and
on, and enabling it writes the same 250 millisecond wait when none is stored.
The evaluation freezers clear the ranking section from the default
configuration, so their pinned desktop configuration and frozen corpora are
unchanged.

## Prediction evaluation

Every earlier result is in conversion mode. This evaluation, declared before
any prediction result is seen, measures the shipped ranker in prediction mode,
where Mozc proposes completed text while the reading is still partial.

The inputs come from the development and holdout rank-zero cases. For a case
whose reading has at least four characters, the typed prefix is the reading
without its last two characters, and the target is the complete expected
output. The evaluation freezer gains a prediction mode that composes the
prefix, calls `StartPrediction` with a `PREDICTION` request, and freezes the
resulting request in `MODE_PREDICTION`, so the ranker sees exactly what the
Engine seam would send. The frozen prediction corpus reuses the frozen case
layout of the quality regression corpus.

Each case is ranked by the shipped setting: the small model, the `next`
window, `K` = 5, reading copy penalty 100, and order prior 1.0. Top-one
accuracy is the share of cases whose first merged candidate equals the target,
and top-three accuracy is the share whose target is among the first three
merged candidates, both compared with Mozc's order on the same frozen
requests. The ranker helps prediction if development top-one accuracy rises
with more wins than regressions; the holdout is then reported once as a
confirmation. If it does not help, the shipped product ranks only conversion
requests.

## Prediction result

The ranker hurts prediction. On the 214 development prefixes, Mozc's first
prediction is the target in 44 cases and the ranked first prediction in 10,
with 2 wins, 36 regressions, and an exact bootstrap gain interval of -45
through -23 cases; the target is among the first three in 69 and 33 cases. On
the 71 holdout prefixes, Mozc has 3 and the ranker 0, with 3 regressions.
zenz scores the conversion of exactly the typed reading, so it promotes the
candidate that converts only the prefix over the completion: `かねのな`
becomes 金の名 instead of 金のなる木, and `よろしくおねがいし` becomes よろしくお願いし
instead of よろしくお願いします. As declared, the shipped ranker now ranks only
conversion requests and returns no order for prediction and suggestion
requests, which keep Mozc's order.

## Runtime for the experiment

The experiment scores the GGUF from Python with `llama-cpp-python`, which is
not installed on this host. Model weights and caches go under
`.tools/cache/experiment/zenz-ranker/` and are not committed.

## Stages

1. Pin the model repository revision, file name, size, and SHA256 in
   `scripts/experiment/config/zenz_ranker.json`.
2. Add the prompt builder and scorer with unit tests over synthetic requests.
3. Run the development corpus, write `report/experiment/zenz-ranker/`, and
   report the metrics and latency.
4. Only if the development result beats Mozc, plan Engine integration through
   the existing `CandidateRankerBackendInterface`. Fine-tuning on corpus
   sentences remains the fallback if it does not.
