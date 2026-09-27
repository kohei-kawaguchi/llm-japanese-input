# LLM Japanese Input architecture

## Objective

Build a Windows x64 Japanese IME that preserves Mozc's composition, conversion,
dictionary, learning, shortcut, renderer, and TSF behavior. Change only the
ordering of word candidates by inserting a small local language model ranking
boundary after Mozc creates conventional candidates.

The upstream base is pinned to `google/mozc` commit
`851c3fe33060d2a6090363e4d7ec44fafde2c03d`. Mozc has no stable-release channel,
so the exact commit is the reproducibility boundary.

## Product boundary

The project retains Mozc behavior unless a difference is required for candidate
ranking or independent product identity. Google and Mozc names, icons, and
other protected branding will not be reused as this product's identity.

The first implementation supports Windows x64. ARM64 and non-Windows clients
are outside the first slice.

## Architecture

The existing Mozc client, server, converter, renderer, dictionaries, and user
history remain authoritative. A model-neutral `CandidateRankerBackend`
receives an immutable view of one conversion result and returns a ranking
decision. The backend is called only by the ranking worker and is not allowed
to mutate composition state directly.

`Engine` owns one process-wide ranking service and loaded ranker.
`EngineConverter`, which is owned by one input session, owns the session
generation and state revision used to reject stale work. The shared `Converter`
does not own ranking state because it serves multiple sessions and some of its
generation paths invoke postprocessing more than once.

Each request contains:

- a token composed of session generation, state revision, and request sequence;
- the complete composer reading supplied independently of segment keys, plus
  segment boundaries;
- preceding and following text from the current session command. The session
  clears this ranker-only context when the command does not contain it instead
  of reusing text from an older operation;
- request-local identifiers, values, readings, consumed key lengths, costs,
  and attributes for Mozc candidates;
- the active suggestion, prediction, or conversion mode.

Each response contains:

- the matching request identifier;
- an ordered list of existing candidate identifiers.

The ranker returns either a response or a typed error status. The merger
validates the complete response before mutation, rejects duplicate references,
and keeps all unmentioned Mozc candidates in their original relative order. A
response for an older request is rejected and cannot update a newer
composition.

Candidates marked `NO_MODIFICATION`, `COMMAND_CANDIDATE`, or
`DISABLE_RESCORING`, or with a non-default command remain in their original
slots. Meta candidates are not exposed to the ranker. Existing candidate
objects are moved rather than reconstructed, so all costs, descriptions,
attributes, and learning metadata survive unchanged.

Model ranking is an explicit configuration state. When it is disabled, base
Mozc ordering is normal behavior, ranking diagnostics are omitted, and the
keystroke path does not retain ranker context or contact the ranking service.
The persisted configuration contains `enabled` and a required positive
`max_wait_millisec`; it does not invent a timeout when the value is missing.
When ranking is enabled, invalid configuration, missing model artifacts, or an
unavailable backend produce stable client diagnostics rather than silently
changing the configured mode.

The ranking service owns one persistent worker, at most one in-flight job, and
at most one pending job per session generation. New work replaces pending work
for the same session and cancels that session's in-flight work. Other sessions
remain first-in, first-out. A request deadline covers both queueing and
inference. `Invalidate` records the newest live token even when no job exists,
and admission rejects any request older than that recorded token before it can
reach the backend. Timed-out, cancelled, superseded, or stale results are
discarded; the worker never captures or calls an `EngineConverter`.

The backend contract accepts cooperative cancellation. Service shutdown stops
admission, cancels waiters and inference, wakes the worker, and joins it. Session
objects are destroyed before the Engine service so each session can cancel its
work while the service is alive.
`CancelSession` is terminal for that unique session generation; its owner must
not submit or invalidate work after cancellation, and generation identifiers
are never reused.

Client diagnostics expose only stable outcome and reason enums, request-token
components, and elapsed milliseconds. They never contain candidate text,
surrounding text, prompts, model output, or raw backend errors.
Operational logs likewise record only stable status codes, never raw backend
messages or request content.
Each output reflects only the current session diagnostic. Output serialization
removes a pre-existing diagnostic field when the session has no current
diagnostic, including when callers reuse a protocol message.

## Narrative flow

1. Mozc processes keystrokes and creates its normal conversion result.
2. Mozc runs its complete rewriter chain, including environmental filtering,
   redundant-candidate cleanup, accessibility descriptions, and the user
   suppression dictionary. Candidate limits are deferred only for an eligible
   request that can be admitted to the configured ranking service.
3. The session-owned `EngineConverter` snapshots the surviving candidates into
   a ranker request after one complete generation operation.
4. The engine-owned ranking service evaluates the immutable request on its
   worker and returns before the configured deadline or reports a typed status.
5. The session validates the token and live snapshot. If the generation is no
   longer current, the superseded outer operation exits without merging,
   trimming, or updating diagnostics for the newer state.
6. For the still-current generation, the deterministic merger applies a valid
   order, then Mozc applies final candidate limits whether ranking succeeded or
   reported an error.
7. The normal Mozc renderer displays the resulting candidate list.
8. User selection continues through Mozc's existing learning path.

The production seam is in `EngineConverter`, after a complete conversion,
suggestion, prediction, or segment-resize generation call and before
`UpdateCandidateList()`. Ranking is not called from shared converter
postprocessing, focus changes, output serialization, commit paths, reverse
conversion, auxiliary incognito generation, forced transliteration, kana-type
switching, or intermediate recursive generation. Each outer conversion,
suggestion, prediction, or public resize operation ranks at most once. A cloned
converter receives a fresh session generation and inherits no pending model
work, surrounding text, or diagnostic. It does inherit the password privacy
classification so undo cannot make restored secret composition rankable.

The live token at apply time is read from the session. Every context, request,
config, key, generation, focus, resize, candidate movement, transliteration,
kana switching, commit, cancel, reset, revert, and history-deletion mutation
advances the state revision and invalidates service work. Output serialization
does not. Passing the request's original token back as the live token is not
valid for asynchronous execution.

The session copies only the current command's preceding and following strings
into dedicated `EngineConverter` ranking state. These strings are not added to
Mozc `ConversionRequest`s that previously omitted context, so baseline
conversion behavior does not change. The complete model reading always comes
from `Composer::GetQueryForConversion()`; prediction keys and partial segment
keys are not substitutes for the full composition.

No password-field operation may invoke the ranker or retain password context,
including explicit conversion, prediction, and public resize operations.
An explicitly observed password field remains private across later commands
that omit the field type. Only an explicit non-password field transition may
clear that privacy state; surrounding text is still refreshed or cleared for
each command independently.

Selecting a candidate while the converter is still in suggestion state may
internally promote suggestions to prediction state, but that promotion is part
of candidate movement and must not invoke ranking. It uses the same unranked
internal prediction path as prediction expansion.

Mozc currently truncates suggestions inside `MergerRewriter::Rewrite` and
applies the request candidate limit after suppression in `Converter`. These
operations are extracted into shared candidate-limit helpers. Normal converter
callers invoke each helper at its original phase, preserving the existing order
in which suggestion truncation precedes user suppression. An eligible ranking
request skips both phase calls, then invokes the combined finalizer after
`EngineConverter` has applied or rejected the ranking result, so the model sees
the complete filtered and suppressed candidate set.
Auxiliary incognito generation explicitly clears the deferral option and keeps
its existing limit behavior; its candidates are never sent to the ranker.

Generated candidate strings are outside the first production path. Adding them
later requires an earlier insertion seam so they pass through the same grammar,
environmental, suppression, redundancy, and accessibility processing as Mozc
candidates.

## Local ranking backend

The first benchmark candidate is
`sbintuitions/sarashina2.2-0.5b-instruct-v0.1` at immutable revision
`e4b9aacc3f644893d0179847946ef6c58d868f29`. It is an MIT licensed Japanese
Llama model with 793,048,320 BF16 parameters. The official weight is
1,586,121,792 bytes with SHA256
`3d886d5ba826134c408fee95666240ae2dc1372e944b7521c8448d4def08502f`.
The tokenizer SHA256 is
`008293028e1a9d9a1038d9b63d989a2319797dfeaa03f171093a57b33a3a8277`.
This pin is a benchmark candidate, not a packaged default, until it passes the
quality and latency gates below.

The pinned llama.cpp converter produces a 1,588,787,072-byte BF16 GGUF with
SHA256 `e336021cc91a35f620462826d7c98a2a6e9024c297dd9dba198f04ede04dd332`.
The pinned Q4_K_M quantizer produces a 528,204,672-byte GGUF with SHA256
`5d7aacde26781232dcfb9b830698f94a4a17b2429f4d9567b21ab078a3166a56`.
Its 219 tensors comprise 133 Q4_K, 13 Q6_K, 12 Q5_0, 12 Q8_0, and 49 F32
tensors. Every tensor name, shape, tokenizer control, label token ID, and
semantic metadata field matches the BF16 GGUF; only the expected file type and
tensor storage types differ. Both artifacts remain ignored benchmark inputs
until their model behavior passes the promotion gates below.

The first optimized Q4_K_M smoke on an Intel i9-11900H used a 189-token,
single-segment prompt. Backend creation took 3.48 to 3.56 seconds and seven
warm ranking calls took 4.705 to 5.169 seconds. With the same four candidates,
one ordering ranked the contextually wrong D-labelled candidate first; after a
permutation it ranked the C-labelled candidate first and changed the semantic
winner. This fails both the latency and label-permutation gates. The 793M
instruction model and distinct-label next-token scoring are therefore rejected
as production candidates. Their checked artifacts and smoke result remain a
reproducible negative baseline only; they are not eligible for Engine wiring or
the full AJIMEE run.

The first continuation-scoring prototype uses
`rinna/japanese-gpt2-xsmall` at immutable revision
`8e91527b3276e0565154935e84a08bf0137ed99f`. It is an MIT licensed Japanese
base causal model trained on Japanese CC-100 and Wikipedia. The network has six
GPT-2 layers, width 512, eight attention heads, inner width 2304, a 1024-token
context, and a 32,000-token vocabulary. Its safetensors file is 155,892,312
bytes with SHA256
`06662948dfa3227e3a42200787f28e86657d0f1e37a4ba1439452419581a4229`;
the 805,634-byte SentencePiece model has SHA256
`b5cbdfa8aa7c54c8c5af85b78c309c54a5f2749a20468bf6f60eee007fe6fec1`.
The weight contains 37,398,016 learned floating-point parameters. Six scalar
masked-bias buffers and six U8 causal-mask buffers are not learned parameters
and are excluded from the converted model. The provider reports
approximately 28 perplexity on its selected CC-100 validation set, which is not
comparable to other providers' validation results and is not an IME quality
claim.

This model is not a zero-change input to the pinned generic GPT-2 converter.
Its architecture is GPT-2, but its tokenizer is a lowercasing T5Tokenizer over
a SentencePiece unigram `spiece.model`, and its configured inner width is 2304
rather than four times the embedding width. A checked model specialization must
reuse the pinned converter's T5 unigram vocabulary writer, preserve its
precompiled normalizer map, dummy-prefix and whitespace settings, record the
model's lowercase normalization explicitly, write `n_inner` as the feed
forward width, and discard the stored attention-mask buffers. The ranker applies
`PYTHON_UNICODE_14_SCALAR_LOWER` to the complete serialized record before the
pinned runtime's unigram tokenizer. This reproduces the pinned Transformers
slow tokenizer, whose minimal regex matches call Python 3.11 `lower()` once per
non-special Unicode scalar rather than once per complete string. The checked
implementation uses Unicode 14 simple and unconditional full lowercase
mappings, includes multi-code-point expansions such as U+0130, and deliberately
does not apply the contextual final-sigma rule. Model special strings are not
recognized in request text because model control parsing remains disabled. The
normalization does not depend on the host Windows Unicode version. Token IDs
from the converted GGUF must
match the immutable Hugging Face tokenizer over a checked Japanese, Latin,
mixed-width, whitespace, and normalization corpus before any model benchmark.
Tensor names, shapes, counts, BOS ID 1, EOS ID 2, and reference logits must also
match. This bounded converter and tokenizer correction is part of the checked
model import, not an unverified compatibility mode.

The specialization also derives and writes the tokenizer's nonstandard special
IDs: unknown 0, BOS 1, EOS 2, padding 3, separator 5, and mask 6. Without those
fields the generic T5 runtime defaults unknown to ID 2 and corrupts out-of-vocab
text. A metadata-only GGUF from the patched converter now matches the immutable
Hugging Face tokenizer on ten checked Japanese, Latin, fullwidth, whitespace,
NFKC, combining-mark, U+0130 expansion, and Greek scalar-lower strings. The
checked C++ normalizer replaces the temporary Python reference before weight
conversion.

The checked F32 reference artifact `rinna-xsmall-f32.gguf` is 150,626,624
bytes with SHA256
`b7f36bcf7fa6e13ef434f8a95080f5892df131f7841abc611f975e2736a82ddd`.
Its 76 learned tensors contain 37,398,016 parameters, and every F32 value is
bit-identical to the immutable safetensors source after only the required
GPT-2 Conv1D transposes. The checked F16 benchmark artifact
`rinna-xsmall-f16.gguf` is 76,964,160 bytes with SHA256
`62d04a340abcbaf2f11242dac30043f4a9a1419d844d3487782275cc6ef39e65`.
Its 25 F16 tensors and 51 required F32 tensors match the same source after the
declared F16 cast. Both artifacts match all 32,000 SentencePiece pieces,
scores, types, special IDs, and the 237,538-byte precompiled normalizer map.
Their former schema-version-3 continuation manifests are invalid after the
schema-version-4 bounded-execution contract. New F32 and F16 manifest hashes
are recorded only after validation and the corrected score probes pass.
A checked offline verifier retains this evidence instead of relying on an
interactive inspection. It rejects missing, duplicate, or extra tensors;
recomputes every required Conv1D transpose and F16 cast from the immutable
safetensors source; verifies all excluded buffers, typed GGUF metadata,
tokenizer arrays, normalizer bytes, source and artifact hashes, converter source
archive, and checked patch; and reproduces deterministic reports byte-for-byte.
Each report binds the complete verifier implementation by hashing both the CLI
entry point and every imported project verification library that controls
schema selection, tensor types, packing, or canonical report serialization.
The checked entry point has SHA256
`327cd4288acd4f934432f62e12f75965da594a383697c9167e80de1ee890f206`
and its verification library has SHA256
`c1e4e65b3b2054bc7c3355fbc9f9931eef674051e1f29037ea8b56974680a8e6`.
The verification configuration has SHA256
`1b0162029dc6a452ae224e8b584e42b46d59c8975108d54cf0e21a2a452ee6ea`.
The retained F32 report has SHA256
`70660cc893b8390a3d1efb4d3f0dc13007efb8cc97abfa68e0e94825cf8b2735` and
the F16 report has SHA256
`7fa7020549ad2f2e7bd61b8d36222050e400958a028719bbf4d919b21b49de65`.
These checks prove the artifact import, not ranking quality or latency.

The Q4_K_M latency artifact `rinna-xsmall-q4_k_m.gguf` is 29,257,024
bytes with SHA256
`3c53d019a0ff0832ff5f7b08508e8b5c3b78b747871060deb2db3375fcfb6677`.
It retains the same 76 logical tensors and 37,398,016 parameters as the checked
F16 input. Its storage contains 51 F32 tensors, 20 Q4_K tensors, and five Q6_K
tensors. Its former schema-version-3 continuation manifest is invalid after the
schema-version-4 bounded-execution contract. The new Q4_K_M manifest hash is
recorded only after validation and the vocabulary-only corpus audit passes.
The Q4 promotion gate must rerun the pinned Q4_K_M quantizer from the
checked F16 artifact in a private temporary directory and requires
byte-for-byte identity with the retained Q4 artifact. The same verifier
requires exact typed metadata and tokenizer identity, tensor names and logical
shapes, the declared Q4_K and Q6_K block packing, exact values for every
retained F32 tensor, and deterministic packed-storage hashes. Before executing
the quantizer, it verifies the exact executable SHA256 recorded by the checked
configuration as well as the wrapper, runtime source, and patch hashes. The
Windows quantizer link must use MSVC `/Brepro`; a wall-clock COFF timestamp is
not an acceptable executable identity. Two independent Bazel output bases must
produce byte-identical executables before their size and SHA256 can enter the
configuration. The earlier 3,447,808-byte executable with SHA256
`1467a55af98c10babdbf0967976511e13a0856145a65ccb9e77b8754331e0159`
was invalidated because its link omitted `/Brepro` and embedded a build-time
timestamp.
The accepted `/Brepro` build is 3,447,808 bytes with SHA256
`0a7423886cb945661f1211fa8c9f7c61ee7f3a507e030f209a9e2eb39dc2d683`.
Two fresh output bases produced those exact bytes and the same deterministic
COFF timestamp `0x114e3d89`.
The Windows quantizer enters through the shared UTF-16 console adapter and
converts
both paths to UTF-8 before llama.cpp sees them. Its report records the checked
F16 input, quantizer executable and sources, regenerated output, and exact byte
comparison. The regenerated artifact matched all 29,257,024 retained bytes.
The shared verifier and configuration use schema 2; all F32, F16, and Q4
reports reproduce byte-for-byte. Model-free Bazel tests isolate schema and
artifact branching, override use, packing checks, and deterministic report
comparison. A manual local check target runs all three retained reports because
the immutable source weights and GGUF files are deliberately not repository
test data. The retained Q4_K_M report has SHA256
`964f34c7b2687328c52db977979cd3a59d7e6d835d3ce870acfb1e6f810a1b85`.
Tensor-type overrides are valid only for a quantized artifact and every
declared override must affect that artifact's expected storage type exactly
once. F32 and F16 verification rejects any tensor-type override rather than
marking an ignored declaration as used.
This proves Q4 artifact provenance and identity, not ranking quality or
latency.

Before quantization, an offline reference probe compares the F32 GGUF against
the immutable Transformers F32 model on fixed serialized records from the
checked tokenizer corpus. Both paths receive the same explicit `[BOS] +
tokenize(record, add_special = false)` sequence. Token IDs must match exactly.
For each record, the probe compares the complete 32,000-value logit rows at the
first, middle, and final input positions, deduplicating coincident positions.
Every value must be finite, NumPy `allclose` must pass with `rtol=1e-4` and
`atol=1e-4`, and the top token must match at every checked row. The probe uses
the portable CPU runtime with F32 KV storage and flash attention disabled so
the import gate isolates model architecture and tensor mapping from later
production precision choices. It writes deterministic binary float output and
does not rank candidates or select a prompt from the evaluation corpus.

The 494M-parameter Apache-2.0 `Qwen/Qwen2.5-0.5B` at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987` is the next checked quality
candidate. It is a base causal model rather than the instruction checkpoint, so
the continuation experiment does not apply a chat template or score an
instruction model outside its declared prompt format. The model is evaluated
in F16 only during development. It is not a latency candidate or packaged
default unless its selected score policy later passes the holdout, AJIMEE,
quantization, and idle host latency gates.

Small Qwen scoring fixtures are repository-checked: the development frozen
corpus, the Qwen native coverage suite, and the three F16 manifests. The F16
GGUF and Hugging Face source stay cache-only. A prepare command downloads the
pinned source revision, converts F16 with the pinned llama.cpp converter,
verifies every configured SHA256, and places the GGUF in the cache model
directory. The Windows-only freezer is why the frozen corpus is checked rather
than rebuilt on Linux. Native coverage can be audited on Linux after those
inputs exist, but the Qwen coverage artifact is also checked so scoring does
not need a second Bazel target before the score job. Checked binary protobufs
are marked binary so Git does not rewrite their bytes.

The HPC4 setup job runs that prepare command and the opt score binary build.
That binary links the same llama.cpp overlay on Linux gcc as on Windows MSVC.
HPC4 has no bazelisk module and no bazelisk on PATH. Setup therefore downloads
the pinned Linux amd64 bazelisk v1.29.0 from the official GitHub release, SHA256
`5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992`, into the
cache path, verifies that hash, and invokes that file. A required command that
is absent prints its name and stops. The score job reads manifests from the
checked directory and the GGUF from the cache model directory. Setup uses 8
CPUs and 32G because F16 conversion plus the opt build exceeds the earlier 4
CPU and 16G setup allocation.

The Qwen F16 import probe compares complete vocabulary logit rows against the
F32 Transformers source with relative tolerance 0.02 and absolute tolerance
0.125. Every value must be finite and every top token must match. This is a
separate precision gate from the 1e-4 F32 rinna import gate above.

The runtime candidate is llama.cpp at commit
`1511ce3bc3f087376c8526b4ad07100bfabb277f` (`v0.1.2`) with source archive
SHA256 `820c78508ee7a8234f4b928d182f6ff0a3d19f58a92f1e69795f93ae0dc46d16`.
Its bundled ggml commit is `8c63e70982c95ceb862e3a1073a2c1beef75d60a`
(`0.20.2`). The production build uses a checked Bazel source overlay and static
CPU libraries. The overlay is compatible with Windows x64 MSVC and Linux x86_64
gcc. Other host operating systems are incompatible. Windows keeps `/bigobj` and
`Advapi32.lib`. Linux defines `_GNU_SOURCE`, as the pinned llama.cpp Linux build
does, before system headers expose GNU scheduling and affinity declarations. It
also links `dl`, `pthread`, and `m`. It does not invoke CMake at runtime or load
unversioned runtime DLLs. The portable x64 build omits native CPU and AVX2
requirements, OpenMP, BLAS, GPU, RPC, and dynamic backend loading.

The source overlay applies a checked patch that makes every public dynamic
backend loader inert and excludes the DLL loader source from the link. A link
smoke test initializes the single statically registered CPU backend. The
runtime license filegroup exposes the root llama.cpp MIT license together with
the Mozilla llamafile matrix multiplication notice, the YaRN attribution, and
the ggllm.cpp tokenizer adaptation attribution for later installer packaging.
It also exposes the bundled SHA-256 and rotate-bits licenses used for streaming
artifact verification.

The pinned CPU kernel otherwise forces F32 GELU through an F16 lookup table.
With exact F32 weights, F32 KV storage, and flash attention disabled, this
approximation produced maximum logit error 0.005001545 and mean absolute error
0.000851522 on the first reference record, although the checked top tokens
matched. The source model declares Transformers `gelu_new`, whose tanh formula
matches ggml's exact F32 GELU implementation. A checked overlay therefore
disables the F16 GELU lookup path and retains the direct F32 formula. The
unchanged F32 parity gate then passed all six records. The worst full-vocabulary
absolute error was 0.0000406504, the largest case mean was 0.00000519312, and
every checked top token matched. The deterministic retained report has SHA256
`a2efd51525e46bb3edbab2354f4d6560497f967d46a824b330c5ee6bb9f63027`.
This cleared the model import gate for an exploratory quantization. The pinned
Q4_K_M quantizer produced `rinna-xsmall-q4_k_m.gguf`, which is 29,257,024 bytes
with SHA256
`3c53d019a0ff0832ff5f7b08508e8b5c3b78b747871060deb2db3375fcfb6677`.
The artifact remains an evaluation input rather than a packaged model.

The experimental scoring design ranks existing Mozc candidates by the probability
of complete candidate continuations and never generates or parses model text.
It does not expose candidate lists, positions, counts, or artificial labels to
the model. For each target segment and candidate, the exact serialization is
`mode_prefix + mode + field_separator + reading_prefix + full_reading +
field_separator + segment_key_prefix + segment_key + field_separator +
text_prefix + external_preceding_text + baseline_left_values + candidate_value +
baseline_right_values + external_following_text`. Baseline values are the
candidate with the smallest immutable candidate ID in every other conversion
segment of the pre-rank Mozc snapshot, concatenated in immutable segment-ID
order. They are frozen once for the complete request. Each target candidate is
therefore scored as one substitution into the same Mozc baseline sentence. No
previously reranked segment is fed into a later segment, so results do not
depend on segment evaluation order. A
single-segment suggestion or prediction reduces to external preceding text,
the candidate, and external following text. All natural-text surfaces are
adjacent. The causal model scores their joint continuation probability instead
of placing artificial separators around the candidate. Candidate-token
probabilities use the editor context and baseline left segments; baseline right
segments and editor following text affect the objective through their
probability after the candidate. The target candidate is the only varying
surface in a segment.

Each record's field lengths are first summed with overflow checks. A serialized
byte count above the manifest limit is a typed capacity limiter and stops that
strict candidate prefix without scanning the oversized bytes. Otherwise every
serialized field is validated as UTF-8, the complete record is transformed by
the manifest-selected text normalization, its normalized byte count is checked,
and it is tokenized once with special-token insertion and parsing disabled.
This order keeps candidate text after the first limiter opaque as required by
bounded preparation. A fixed BOS token is
prepended directly by ID. This preserves tokenizer merges across
the preceding-text, candidate, and following-text boundaries. The
`per_record_token_limit` counts the prepended BOS, every normalized-record
token, and the scored terminal token. The terminal is not fed into KV, so the
maximum fed root-to-leaf path is one less than that limit. The
implementation finds the token longest common prefix of `[BOS] +
tokenized_record` across candidates, emits it once as a shared stem, then builds
a prefix trie over the remaining record tokens. A fixed terminal token is
appended afterward as a scored edge and is never fed as an input token. Thus
duplicate records share a scored terminal edge, while a candidate that is a
prefix of another candidate still forms a distinct completed continuation.
Candidate
sequence IDs are assigned in canonical `(segment ID, value bytes, candidate
ID)` order, and trie children are emitted in token-ID order. Model graph and
microbatch layout are therefore invariant to the incoming Mozc candidate
order. Exact score ties retain the original Mozc order.

Each scored edge after the shared stem uses normalized log probability, computed as its target
logit minus a full-vocabulary log-sum-exp in double precision. A candidate score
is the raw sum of those log probabilities from the common-prefix divergence
through the fixed terminal. It is not averaged by model-token count and never
uses raw logits. Shared edges are evaluated once and credited to every
descendant candidate. Non-finite logits, normalizers, or scores are backend
errors. A future length calibration may add a manifest-owned coefficient times
an explicitly defined length feature only if a separate development corpus
demonstrates that it is needed. The feature cannot be assumed to be Unicode
scalar count because the observed cost is produced by model tokens. The fixed
terminal remains part of the baseline objective. The checked diagnostic reports
candidate-or-boundary, following-context, and terminal contributions separately
without changing their sum. Any length calibration or change to the terminal
objective requires evidence from a development corpus separate from AJIMEE.

The first real-model smoke was invalid because its manual Windows executable
accepted Japanese request fields through narrow `argv`. The runtime replaced
the reading, segment key, preceding text, and following text with `?` before
the ranker saw them. The F16 artifact therefore appeared to rank `雨` first
for both `明日は + candidate + が降ります。` and
`子どもに + candidate + を渡す。`, and ranked `校章` above `交渉` for
`契約条件について + candidate + を続ける。`. Q4_K_M retained those three
winners. Candidate values remained intact only because they came from a UTF-8
file. The decisive trace showed the manifest literals and candidate token IDs
were correct while every Japanese command-line field became repeated token ID
3017, the model's `?` piece. An independent F32 Transformers decode over the
intended records ranked `雨` for the rain sentence, `飴` for the candy sentence,
and `交渉` for the contract sentence. For the candy sentence its complete branch
scores were -19.307 for `飴` and -25.256 for `雨`; for the contract sentence they
were -21.732 for `交渉` and -37.803 for `校章`.

Those corrupted comparisons proved only layout equivalence. Every Windows
manual evaluation tool now enters through the UTF-16 console
command line and converts every argument to UTF-8 before `InitMozc` and Abseil
flag parsing. This fixes both Unicode request values and Unicode paths at the
actual boundary instead of routing selected fields through auxiliary files. The
corrected F32 probes rank `雨` for the rain sentence with score -11.8673022,
`飴` for the candy sentence with scores -19.3073864 versus -25.2559458 for
`雨`, and `交渉` for the contract sentence with scores -21.7316685 versus
-37.8029898 for `校章`. These values reproduce the independent Transformers
reference. The configured shared path and independent complete-sequence path
are bit-identical. Under F32 K/V with flash attention disabled, every shared
and independent total and edge agrees within 0.000007. The production `Rank`
response equals the configured shared result, and every cyclic candidate
rotation preserves mapped scores and the winner exactly. The reports include
all three contribution classes. Those reports used a schema-version-3 manifest
and cease to be retained promotion evidence when bounded execution becomes
schema version 4. The three corrected cases are rerun after the version 4
manifests are generated. A retained diagnostic report must identify the
exact manifest SHA-256, GGUF file name and SHA-256, source model revision,
runtime revision, and numeric profile before any scores. Its file name is not
artifact provenance. The objective may change only from corrected evidence and
a development corpus separate from AJIMEE.

The checked F16 development objective scores were repeated on the 219 frozen
development cases and matched byte-for-byte. The binary score artifact SHA256 is
`bee92369f6e642a9bb13bc70bfede6735dbee3b71a616760a93ad96a1b735984`; the
review textproto SHA256 is
`43940dfcb22fa620044490e639efa2ff53eb00a5a4fe9a7faf415442920ce813`. The
answer-aware schema-version-1 development report also repeated byte-for-byte.
Its binary SHA256 is
`2493fd36d21a90827d47a5363880a1bf8197baeedae2433ea7d51a8d682f0038`, and its
canonical JSON SHA256 is
`aeab7868abe8ce18199f5333c5cd6894a73ef68e264cd47b4e57285b8058d17a`.
No objective passed the development gate. `STRUCTURED_WITH_TARGET` had zero
wins, 157 regressions, bootstrap lower gain -170, and 156 Mozc-correct exact-key
regressions. `STRUCTURED_WITHOUT_TARGET` removed 29 clean key-copy promotions
against the reference but still had zero wins, 150 regressions, bootstrap lower
gain -163, and 145 Mozc-correct exact-key regressions. `NATURAL_TEXT_ONLY` had
17 wins and 26 regressions, bootstrap lower gain -22, and four Mozc-correct
exact-key regressions. The failure mechanism is not artifact nondeterminism or
coverage loss; the F16 continuation score is still promoting answer-irrelevant
exact-key copies and regressing Mozc-correct outputs. Production wiring remains
blocked. The next bounded experiment must change the scoring objective or model
candidate and produce a new development report before any holdout, AJIMEE,
quantization, latency, branding, or TSF product work.

A config-owned exact-key-copy penalty was then applied at development-report
time to recompute the candidate order from the retained raw score bits. This
calibration leaves the score artifact unchanged and is recorded by the report
config hash. It removed the exact-key-copy regressions but did not pass the
development gate. The calibrated binary report SHA256 is
`f35ba9a3dd90d9577856584fbb18433ff17690063200e26aa4a20efe9ba06987`, and the
canonical JSON SHA256 is
`ba306dcda2a74dfa2cf6128b519bac980fb225d2e4935ee8c7e09e6a20fe8047`.
`STRUCTURED_WITH_TARGET` had 15 wins and 30 regressions, bootstrap lower gain
-28, and zero exact-key regressions. `STRUCTURED_WITHOUT_TARGET` had 13 wins
and 31 regressions, bootstrap lower gain -31, and zero exact-key regressions.
`NATURAL_TEXT_ONLY` had 16 wins and 23 regressions, bootstrap lower gain -19,
and zero exact-key regressions. The remaining root cause is relevance accuracy,
not exact key copy bias.

The completed development experiment compares the retained 37M rinna F16
control with the pinned 494M Qwen F16 candidate. Each model is scored once for
the three record layouts. The score artifact stores the existing full
continuation log probability and the positive number of scored continuation
edges. Neither score runner receives answers. A candidate or boundary mean was
rejected before artifact generation because source line 1, segment 0, candidate
1 is a valid duplicate Mozc value whose common prefix and following context
both start at position 33. Shared prefix compression therefore gives that
candidate no varying candidate edge. The terminal edge still guarantees at
least one scored continuation edge.

The answer aware reporter evaluates the finite cross product of two
aggregations, four Mozc order priors, and two exact key copy penalties. The
aggregations are the full continuation sum and that sum divided by the number
of scored continuation edges. The Mozc order prior coefficients are 0, 0.25,
0.5, and 1.0 log probability units per original candidate index. The exact key
copy penalties are 0 and 100 log probability units. For candidate index `i`,
the calibrated score is the selected model aggregation minus the order prior
coefficient times `i`, then minus the exact key penalty when the Mozc top
differs from the segment key and the candidate value equals that key.

The development gates remain unchanged. A cell must have net gain of at least
five cases, a positive exact bootstrap lower gain, no more than three
Mozc-correct regressions, fewer newly added than removed clean key copies, and
zero Mozc-correct exact key regressions. Passing cells are ordered by net gain,
retained Mozc-correct cases, model character error, clean key copy promotions,
model parameter count, declared model order, record layout, aggregation, Mozc
order prior coefficient, and exact key copy penalty. The first cell is the only
selection. Its exact aggregation and coefficients become required fields in a
schema-version-6 model manifest before holdout execution. If no cell passes,
the experiment records the failure and stops before holdout or production work.

The two rinna score runs are byte-identical with binary SHA256
`de91b14952c2fe0e73700d3456055a91eb85697d9daf1bd13cec022cc2d8b46e`.
The two Qwen score runs are byte-identical with binary SHA256
`e6df5795092a3938f2618c167c1d8fd91940bed6df2ad4cfc5b5b0aab65e2b7d`.
The report config for both models has SHA256
`7b65a859ddcaf4c4d12febec30cc559a883a20002cc7f4494c1accc96a085a66`.
Independent reports are byte-identical. Their binary SHA256 is
`0863da2dc27c581934490f028e293658aadd8944911299d2cf38f5f4c6405d95`,
and their canonical JSON SHA256 is
`c9dce5deb448477e6c743aa805d8d6f5c2f049e7592d14143dcc6ed4b2fa4d45`.

No cell passes among the 96 predeclared combinations. The strongest rinna cell
uses natural text only, the full continuation sum, a Mozc order prior of 1.0,
and the exact key copy penalty of 100. It has 174 correct cases, 21 wins, 11
regressions, net gain 10, exact bootstrap lower gain -1, zero Mozc-correct exact
key regressions, 240 removed clean key copies, and zero added clean key copies.
It fails the positive bootstrap lower gain and maximum three regression gates.
The strongest Qwen cell uses the same layout, aggregation, and coefficients. It
has 156 correct cases, 17 wins, 25 regressions, net gain -8, exact bootstrap
lower gain -21, zero Mozc-correct exact key regressions, 271 removed clean key
copies, and zero added clean key copies. Qwen is worse than the 164-correct
Mozc baseline and the rinna control. The report therefore selects no model,
layout, aggregation, or coefficients. No schema-version-6 manifest is created,
and the holdout, AJIMEE, latency, packaging, and production paths remain closed.

Score attribution is diagnostic only. It separately tokenizes the normalized
record prefix ending immediately after the target candidate, including the
editor context and frozen baseline left segments but excluding baseline right
segments and editor following text. If those prefix tokens are an exact prefix
of the complete record tokens, the next target begins the following-context
contribution. If tokenization changes across that boundary, the first divergent
complete-record target is classified as candidate-or-boundary and
following-context attribution begins after it. The fixed terminal has its own
class. These labels never alter the summed score.

The llama.cpp batch contains each shared-stem and trie input node once, parent
before child.
Each node belongs to the canonically sorted set of descendant candidate
sequence IDs, and its position is its trie depth. Only predecessor nodes with a
scored outgoing edge request logits. The target terminal leaves are not fed as
inputs. Positive `llama_get_logits_ith` indices are the original logical batch
token indices. One full-vocabulary row and log-sum-exp are reused for every
outgoing edge from that predecessor.

This coupled sequence layout requires unified KV storage. The pinned separate
sequence allocator rejects tokens shared by multiple sequence IDs. The context
sets `kv_unified=true`, uses globally unique sequence IDs across all segment
tries in a decode, and rejects more than llama.cpp's 256 sequence IDs. The
logical batch and unified KV capacity cover the aggregate unique trie input
nodes, while the longest root-to-leaf path must fit the model context. Output
capacity covers every marked predecessor row. The manifest deliberately makes
its output-row limit at least its candidate-sequence limit because the pinned
runtime reserves the larger of actual output rows and sequence capacity.
Full-vocabulary output memory is bounded by a separate byte limit applied to
`max(output_row_limit, candidate_sequence_limit) * vocabulary_size * 4`.
The context sets `n_ctx` and `n_batch` to the per-decode input-node limit,
`n_ubatch` to the microbatch limit, and `n_seq_max` to the per-decode sequence
limit. The input limit is a multiple of the runtime's 256-token padding.
`n_outputs_max` equals the per-decode output-row limit. The pinned
runtime reserves output storage for the full logical batch before processing
its microbatches and asserts that the larger of the actual output rows and
sequence capacity fits this context parameter. The separate byte limit
preflights that allocation. No sampling backend is installed.

The checked model manifest owns a versioned `ScoringTemplate`, a required
`BoundedExecutionPolicy`, and per-decode limits. The scoring template owns the
BOS and terminal token IDs, the exact mode, reading, segment-key, text, and
field-separator literals, and text normalization. The bounded policy requires
`MOZC_ORDER_UNPROTECTED_PREFIX`,
`LONGEST_FEASIBLE_UNPROTECTED_PREFIX_OR_OMIT_SEGMENT`, and
`SEGMENT_ID_ORDER_GREEDY_WHOLE_SEGMENTS`. The limits own the maximum selected
candidates per segment, segments and candidate sequences per decode, per-record
byte and token limits, per-decode input-node and output-row limits, and maximum
decode output-logit bytes. The manifest also owns runtime parameters, source
revision, GGUF SHA256, quantization, tokenizer SHA256, and license references.
Runtime code does not duplicate those constants. The per-decode sequence cap
cannot exceed 256, and both runtime thread counts are limited to 256.

Before copying a request into model preparation, the backend scans its structure
once with cancellation checks and enforces `max_request_segments`,
`max_request_candidates`, and `max_request_string_bytes`. Request string bytes
are the overflow-checked sum of
the UTF-8 byte lengths of reading, preceding and following context, every
segment key, and every candidate key and value. These are hard request work
bounds, not selection limits. The preflight checks segment count first,
candidate count second, and string bytes last; a typed violation leaves every
later usage field zero. The vocabulary-only auditor returns that violation as
data, while production ranking translates it to `ResourceExhausted` and leaves
Mozc order unchanged. The request-segment bound
also bounds selected segment plans and decode batch count because no segment is
split and no batch is empty.

For each immutable request segment, candidate window selection walks candidates
in original Mozc order, skips protected candidates, and takes the first
`max_selected_candidates_per_segment` unprotected candidates. Cost, value, and
candidate ID never choose the window. The smallest-ID candidate of every
segment remains the frozen neighboring baseline even when it is protected or
outside the window. Canonical value-byte and candidate-ID sorting occurs only
after window selection for graph layout, while original indices remain the
stable score-tie order.

Capacity reduction considers each segment independently against an otherwise
empty decode. It prepares the window in Mozc order and retains the longest
strict prefix whose records and exact whole-segment trie fit every per-record
and per-decode capacity. It stops at the first record byte, normalization,
token, trie-node, output-row, reserved-row, or logit-byte capacity failure. It
never skips the blocking candidate to admit a lower candidate and never shrinks
a segment to consume leftover capacity in another batch. If fewer than two
candidates fit, the segment is omitted from ranking and its entire Mozc order
remains unchanged. These are explicit bounded omissions, not backend errors.
Malformed IDs or structure, tokenizer/runtime inconsistency, and other
non-capacity failures still fail the complete call. UTF-8 validation applies to
every string that can influence a record: request text, segment keys, frozen
neighbor baseline values, and each prepared window candidate through the first
limiter. Candidate text outside the window or after that limiter is opaque to
the model and remains untouched; it is not rejected merely because it is never
serialized.

Selected segments are packed in segment-ID order. The next whole segment is
appended to the current decode only when the exact combined plan fits every
per-decode segment, sequence, node, row, reserved-row, and logit-byte limit.
Otherwise the current batch is finalized and the unchanged segment starts the
next batch. A segment is never split across decodes because its candidates must
share one prefix and comparable score space. Every batch uses the same immutable
complete request context and frozen neighboring baselines. Local sequence IDs
reset in each batch and have no semantic effect.

All windows, reductions, and batches are prepared before inference. Under the
existing context mutex, model memory is cleared for each batch, batches decode
sequentially, and their segment orders accumulate only in request-local state.
The backend returns one final validated response after every batch succeeds.
Cancellation, deadline cancellation, supersession, shutdown, decode failure,
or response inconsistency in any batch discards the entire aggregate. No
partial response can reach the merger. Because segment tries are disjoint and
scores are compared only within one segment, a whole segment has identical
records, prefix, edges, scores, and order whether decoded alone or packed with
other segments. The existing merger places scored IDs first in unprotected
slots, appends every omitted unprotected candidate in original relative order,
pins protected slots, and leaves omitted segments unchanged.

Backend creation accepts only a decoder-only causal model: the pinned runtime
must report a decoder and no encoder. Encoder models override output capacity
with the logical batch size and would invalidate the manifest's checked row and
byte limits. Both creation and every batch's accounting use the manifest's full
per-decode sequence capacity because llama.cpp reserves at least the context
`n_seq_max`, not merely the current batch's sequence count.

Manifest schema version 5 retains the schema-version-4 bounded-execution
contract, requires `BoundedExecutionPolicy` on top-level tag 7, and requires an
explicit `RecordLayout` on scoring-template tag 14. Schema version 4 is
rejected by the current runtime. The policy enums accept only the three values
defined above. Limits
rename their existing wire tags to `max_selected_candidates_per_segment` on tag
1, `max_segments_per_decode` on tag 2, `max_sequences_per_decode` on tag 6,
`per_record_byte_limit` on tag 7, `per_record_token_limit` on tag 8,
`per_decode_input_node_limit` on tag 9, `per_decode_output_row_limit` on tag 10,
`max_decode_output_logit_bytes` on tag 11, `max_request_segments` on tag 12,
`max_request_candidates` on tag 13, and `max_request_string_bytes` on tag 14.
The retired text-format names are reserved. Validation requires at least two
selected candidates per segment, the selected-candidate limit no greater than
the sequence limit, the sequence limit no greater than 256, and the request
segment and candidate limits no smaller
than their corresponding per-decode and window limits, positive request and
per-decode limits, the existing context padding constraint, and output capacity
at least sequence capacity. There is no schema-version-4 compatibility path.

Schema version 3 removed the label and instruction prompt messages, reserved
their old top-level wire tag, and added `ScoringTemplate` on a new tag.
The template permits only `NONE` and `PYTHON_UNICODE_14_SCALAR_LOWER` text
normalization; the latter reproduces Python 3.11's Unicode 14 per-scalar lower
mapping before tokenizer normalization. It also requires the
`MOZC_BASELINE_SUBSTITUTION` context policy. This policy freezes the top value
of every segment from the complete pre-rank request and makes those neighboring
values part of each target segment's natural text. Version 2 external-only
manifests fail validation instead of silently retaining the defective
multi-segment semantics. The obsolete
top-level `prompt` name is reserved as well as its wire tag. Runtime KV storage
has one accepted value, unified, on a fresh numeric value; both retired enum
values and names are reserved so an old nested Runtime cannot decode as the new
mode. Older manifests fail version validation and cannot be interpreted as the
new schema.

Record construction, normalization, tokenization, strict-prefix testing,
combined-plan packing, trie construction, and logits scanning all check
cancellation. Candidate records are constructed and
tokenized one at a time after their serialized and normalized byte sizes are
validated, and their duplicated text is
released immediately after tokenization. The existing cancellation object also
drives llama.cpp's CPU abort callback. An aborted or otherwise failed decode is
discarded. A per-batch RAII scope clears the abort callback before request-owned
cancellation storage dies and clears model memory on every exit. Cancellation
is checked again before the complete local aggregate is published. Neither
record text nor request content is logged. Model discovery uses one fixed install-relative
location and verifies the manifest and GGUF hash before the backend becomes
available. Hashing and llama.cpp loading use the same protected open file
handle so a path replacement cannot substitute different bytes between those
operations.

Evaluation data uses checked, versioned protobuf schemas with benchmark-specific
labels. Imported inputs, accepted answers, frozen ranker requests, deterministic
capacity audits, deterministic semantic results, answer-aware quality reports,
and nondeterministic latency samples are separate artifacts.
Only a separately annotated corpus may contain graded candidate relevance tiers
and support nDCG. Canonical corpora use deterministic protobuf text format for
review, while tools exchange deterministic binary serialization so 64-bit
identifiers and candidate snapshots retain their exact types. JSON reports
encode 64-bit identifiers as strings. Required slices cover suggestion,
prediction, conversion, multiple segments, homophones, okurigana, named
entities, numeric and date forms, punctuation, mixed Japanese and Latin text,
absent context, long context, and permutations of candidate input positions.

The external-data importer verifies the pinned source hash and writes an input
corpus containing only case identity, reading, allowed context, and slice labels,
plus a separate answer corpus containing accepted outputs. It never serializes
source `original_text`. Schema version 1 fixes the source roles to the literal
AJIMEE fields `index`, `input`, `context_text`, `expected_output`,
`original_text`, and `splitted_input_for_limited_input_length`; a config cannot
remap those roles while retaining the same source hash. A production-config
test pins the revision, path, hash, counts, role mapping, and attribution. Both
corpora carry the title and creator attribution, exact revision source URL,
CC BY-SA 3.0 identifier and license URL, upstream dataset URL, and a notice that
the JSON was transformed into separated deterministic protobuf corpora while
`original_text` and split chunk contents were omitted. The Mozc freezer accepts
only the input corpus, runs
with candidate ranking disabled and a fresh temporary user profile. It reads
the exact `mozc.data` bytes once, hashes that retained buffer, and constructs the
pinned OSS data manager from the same buffer while keeping it alive longer than
the converter. Before constructing the Engine, the CLI parses the required
canonical UTC RFC3339 evaluation clock from the checked freezer config and
installs one process-scoped `ScopedClockMock`. The checked value is
`2000-01-01T00:00:00Z`. The freezer library installs the same scoped value around
its complete conversion loop so direct library callers receive the same
contract. `ScopedClockMock` atomically saves and restores the preceding clock
override, so the nested CLI and library scopes do not clear caller-owned global
state. The fixed clock supplies UTC as its time zone, so constructor-time and
conversion-time Mozc clock reads are part of the checked evaluation identity
rather than a source of wall-clock variation. It creates a
default desktop request and default config with no ranking section, reconstructs
Mozc history from the published left context, and converts the complete
normalized reading. The conversion request defers only the final candidate
limits, so the freezer snapshots
`BuildCandidateRankerRequest` after Mozc's full rewriter and suppression chain
at the same seam used by production ranking. The null-backend ranking service is
never called and creates no worker thread; no ranking backend or job exists in
the freezer process. It records that exact model-neutral request plus the
concatenation of candidate zero from every conversion segment as Mozc's baseline
output. It never commits or finishes a conversion, so no evaluation case can
learn into a later case. The model runner
accepts only frozen requests. A vocabulary-only auditor first verifies the
checked Q4 GGUF hash, loads its tokenizer metadata without model tensors or a
decode context, and audits every complete request. Only after the entire corpus
passes is that auditor destroyed and one checked backend loaded. The runner
emits typed semantic outcomes without timing fields. A separate latency runner
writes integer samples with host and runtime identity. Only the answer-aware
quality reporter loads the frozen corpus, accepted capacity audit, answers, and
semantic results. It never loads a model, manifest, source JSON, input corpus,
or latency sample. Latency remains a separate later artifact and is consumed by
a separate latency reporter. Quality reports are a deterministic transform of
their checked input hashes and compute aggregate and context-slice metrics plus
the exact paired bootstrap interval.

The checked real import contains 200 input and 200 answer cases, with 100 input
cases in each context slice and 33 cases carrying the source split-data marker.
The deterministic input binary is 24,586 bytes with SHA256
`55fc26b4d86c5d30c2eae5f40fcf1cd9f9b4822ad5a8bae424cdc9ed10e82848`;
its 56,125-byte review textproto has SHA256
`c3be4496fa9c65f4de32db01aad4275fbe20885d5b2ab3a6634f248df544af18`.
The separately retained answer binary is 32,391 bytes with SHA256
`819465115d76cca65eb9d6c8f3f60f8850723376b440eb5c41f0cd8fd7b2c389`;
its 49,020-byte review textproto has SHA256
`3269c94baa578b448328fd1fe49388d78b568ab4528d257e5da369a0e172188b`.
The checked source contains repeated accepted-output strings in source cases
1152, 1163, and 1581. The answer artifact preserves those source lists exactly.
Answer validation therefore requires a nonempty list of nonempty valid UTF-8
strings but does not reject duplicates. Metric computation makes a stable
byte-exact view that keeps only the first occurrence of each accepted string.

The frozen corpus schema records its own version, the copied AJIMEE source and
license identity, the exact input-corpus SHA256, Mozc source revision,
`mozc.data` SHA256 and data type, and the deterministic serialized hashes of
the default desktop request and config, plus the exact fixed UTC evaluation
clock. Each case records the source index,
normalized Hiragana reading, whether Mozc reconstructed a history segment, the
Mozc baseline output, and every field of the model-neutral ranker request:
token, mode, left and right context, reading, focused segment, segment IDs and
keys, and candidate IDs, keys, values, costs, attributes, consumed key sizes,
and protected flags. Cases are ordered by source index. Evaluation tokens are
deterministic and unique by case ordinal; they carry no session data. The
freezer rejects duplicate or unordered IDs, missing conversion candidates,
source or data hash changes, an enabled ranking config, and any partially
initialized artifact.

A shared pure frozen-corpus utility performs the config-independent schema,
source, Mozc identity, case order, token, mode, baseline, segment, and candidate
structure checks. Freezer validation calls it before comparing the checked
freezer-config identity. The same utility converts a structurally valid frozen
request exhaustively into `CandidateRankerRequest`, preserving every repeated
field order and every token, mode, context, reading, focus, segment, candidate,
cost, attribute, consumed-key, and protection value. It never sorts, selects,
merges, or opens an answer artifact. Unknown modes and malformed ordering fail
instead of being normalized.

The first real freeze is invalid as retained evidence. Repeating it in a fresh
process changed only the three time candidates generated for `いま` in source
case 1699, from `10:38`, `10時38分`, and `午前10時38分` to their `10:43`
forms. Mozc's `DateRewriter` read the live clock, so the former binary SHA256
`53fdd2484446a7f7118d44b4d6d826568368190dd1d86d1aca02649d8c082d98` and
textproto SHA256
`c5e4aa9e46b8d3d8833cff6c7730fb160b13b9c95e1f4009ed1c28b8d9a24af8`
must not be used by an audit, runner, or promotion report. A checked real freeze
is retained only after two independent processes using the configured fixed
clock produce byte-identical binary and text artifacts. That gate now passes.
The accepted schema-version-2 binary is 1,335,914 bytes with SHA256
`9b1656a48ed3cfbf1577b7a404688affb49b793f9c556ba2d2fb46b41238c61c`;
the 6,900,264-byte review textproto has SHA256
`6b148ac3658844a885edb8491a31e020c451ad3a3b6bec9f08ce70e65de73930`.
Both independent processes produced those exact bytes. The artifact contains
all 200 cases, records the fixed `2000-01-01T00:00:00Z` clock, reconstructs a
history segment in two cases and an empty history in 198, and contains no answer
fields.

The first external regression benchmark is AJIMEE-Bench at commit
`401666cd56d1a570c2021798b64b6da4396bfd45`; its 200-item
`JWTD_v2/v1/evaluation_items.json` has SHA256
`e9eb668fd6aa14b1e26436f429b5550108af0a1dfd443b8cea0bcb3ab3028fca`.
The benchmark data is CC BY-SA 3.0 and remains a separately attributed
evaluation artifact rather than part of the IME software license. Preprocessing
converts the supplied Katakana reading to Hiragana, gives the model only the
published preceding context, and freezes candidates from the pinned Mozc build
with ranking disabled. The answer-bearing original text and expected outputs
are never included in a model request. The answer-aware report includes overall,
context, and no-context accuracy, character error rate, full Mozc oracle
coverage, conditional accuracy on oracle-covered cases, wins, regressions, and
retention. Latency and timeouts are reported only from the separate later
latency artifact. AJIMEE was published after the model's underlying web data was
collected only to an unknown cutoff, and its source Wikipedia text could have
been memorized. It therefore remains a frozen regression benchmark with unknown
contamination, not evidence of unseen-data generalization and never training
data.

AJIMEE's `splitted_input_for_limited_input_length` field contains replacement
chunk strings rather than split positions and is not lossless. Case 14 changes
`サンビャク` to `サンゼロゼロ`. The importer records only whether that field
was present and always freezes the complete published `input`; it never joins
or uses the chunks. AJIMEE supplies accepted whole outputs but no per-candidate
relevance tiers, so it supports exact accuracy and character error metrics but
not nDCG. It also cannot compare Q4_K_M with F16 from one semantic artifact.
AJIMEE therefore cannot close either the separately annotated nDCG gate or the
Q4_K_M versus F16 gate. Before loading model tensors or creating a decode
context, the runner
uses the exact checked Q4 GGUF vocabulary to audit every complete frozen request
through the exact schema-version-4 bounded preparation path. The audit and
production ranker share candidate windows, strict-prefix capacity reduction,
whole-segment decode packing, record construction, tokenization, and trie
planning. Evaluation-only attribution is added afterward and cannot introduce
a second tokenization rejection. Window and capacity omissions are measured
product behavior, not corpus failures. A structurally valid case passes when
every emitted batch fits, including when some candidates or segments remain at
their Mozc order by policy. Malformed input or a selected whole-segment batch
that cannot satisfy its checked invariant is a typed failure and blocks
promotion. The evaluator never changes the product window or batching policy. A
pure merge-order helper is shared by production mutation and evaluation, so
protected candidates remain pinned and the reported top output is the
concatenation of candidate position zero after the exact product merge.

The deterministic capacity audit records its schema version, frozen-corpus
SHA256, checked manifest and GGUF identity, tokenizer, quantization, source and
runtime revisions, an all-passed bit, and one ordered result per source case.
The complete frozen corpus must pass the shared structural validator before an
audit artifact can be created. A malformed corpus aborts without writing a
partial artifact. Each result therefore contains only the source index, a typed
pass, request-limit-exceeded, or audit-error outcome, canonical status code,
and aggregate usage when preparation reached it. Request-limit-exceeded is the
typed successful auditor result for the three hard work bounds; production
`Rank` and `ScoreForEvaluation` translate the same result to
`ResourceExhausted`. Any non-OK auditor status is recorded as audit-error
without its message, and auditing continues so the deterministic artifact
contains one result for every structurally valid source case. Raw request usage is
descriptive and never compared with per-decode limits: request segments,
candidates, unprotected candidates, rankable segments, and the maximum
unprotected candidates in one segment, plus total request string bytes.
Coverage records window candidates, window omissions, selected segments and
candidates, capacity omissions, and decode batch count. It records maximum
selected serialized and normalized
record bytes and tokens, plus the maximum segments, sequences, input nodes,
output rows, reserved rows, and output-logit bytes in any decode. Each rankable
segment has one ordered diagnostic containing only segment ID, unprotected,
window, selected, and omitted counts, `SELECTED` or `OMITTED_CAPACITY`, and the
first limiting-cap enum with observed and allowed integers. It never stores
records, token IDs, candidate IDs or values, or status messages. Promotion
reports expose this coverage instead of relabeling policy omissions as errors.

The checked capacity-audit config pins the frozen-corpus and artifact schema
versions, frozen binary SHA256 and case count, schema-version-4 manifest SHA256,
GGUF file name and SHA256, tokenizer SHA256, quantization, source revision, and
runtime revision. The vocabulary-only auditor receives that expected identity,
compares every manifest field immediately after parsing and validation, and
fails before opening the GGUF on any mismatch. It then opens only the
manifest-owned simple GGUF filename through the protected handle, verifies the
full file hash, rewinds the same handle, and loads vocabulary metadata only.
Pinned llama.cpp deliberately returns from hyperparameter loading when
`vocab_only` is set, so `llama_model_n_ctx_train` is zero on that path even
though the checked GGUF contains the training context. The shared metadata
validator therefore parses metadata from the same verified `FILE*` with the
pinned GGUF reader in no-allocation mode before model creation. It requires
`general.architecture` to be a nonempty GGUF STRING, constructs the standard
`<architecture>.context_length` key, and requires that key to be a positive
GGUF UINT32 no greater than `INT32_MAX`, matching llama.cpp's public context
API. It then frees the metadata context and rewinds the same handle before
llama.cpp reads it. A full model load requires
`llama_model_n_ctx_train` to equal that typed value. Both paths require the
maximum fed record length, which excludes the scored but un-fed terminal token,
to fit that checked model context. This preserves exact production parity
without loading tensors or creating a decode context in the auditor.
The CLI accepts only config, frozen corpus, manifest, model directory, and
binary and review-text output paths. It has no flag or code path for answers,
source JSON, semantic results, latency artifacts, policy overrides, limits, or
deadlines.

The deterministic audit persists identity, counts, segment IDs, typed outcomes,
canonical status codes, and observed and allowed integers only. It excludes all
request text, context, reading, baselines, records, token IDs, candidate IDs and
values, costs, attributes, scores, logits, status messages, filesystem paths,
timestamps, timing, and host data. It audits each frozen case exactly once in
corpus order and validates the complete artifact without sorting malformed
input. The binary and review text are built fully in memory and written only
after semantic validation. Repeated runs must be byte-identical.

The accepted Q4_K_M vocabulary-only audit ran all 200 frozen AJIMEE requests
twice in independent processes. Both runs produced the same 37,864-byte binary,
SHA256
`4854ae79bc8a24e292be356180420215dd56e70a04c10b12d2f8b242a75e9a5d`,
and the same 525,959-byte review text, SHA256
`d53513eb7bdccf14a164cbff7d1d8db10401f020a94348712207582cdc6cb9dc`.
The checked config SHA256 is
`c1265cc3834522a5e9958950227c4b30421b36690e56be4d822ac66b61f6dc05`.
All 200 results are `PASS`; there are no request-limit or audit-error results.
The corpus contains 1,432 segments and 34,373 candidates. The fixed windows
contain 22,368 candidates and omit 12,005 by window policy. Capacity reduction
omits 44 candidates across six selected segments in three cases, all at the
output-row limit, and omits no whole segment. The audit selects 1,431 segments
and 22,323 candidates in 539 decode batches. One case needs at most 16 batches.
The corpus maxima are 739 serialized and normalized bytes, 136 full record
tokens, seven segments, 64 sequences, 916 input nodes, 512 output rows, 512
reserved rows, and 65,536,000 output-logit bytes per decode. These results close
the schema-version-4 preparation and vocabulary-only promotion gate without
loading tensors or running inference.

The semantic-runner config has `schema_version` 1 on tag 1,
`semantic_results_schema_version` on tag 2,
`capacity_audit_config_sha256` on tag 3, `capacity_audit_sha256` on tag 4, and
`numeric_profile` on tag 5. It does not duplicate corpus, manifest, GGUF,
tokenizer, quantization, or revision identity. The existing checked
capacity-audit config remains the single source of truth for those fields. The
only accepted numeric profile is `CONFIGURED`, which is production `Rank` with
F16 keys, F16 values, and automatic flash-attention selection as pinned by the
runtime revision.

The deterministic semantic artifact has `schema_version` 1 on tag 1,
`identity` on tag 2, and ordered `cases` on tag 3. Its identity records
`frozen_corpus_sha256` on tag 1, `capacity_audit_config_sha256` on tag 2,
`capacity_audit_sha256` on tag 3, `capacity_audit_identity` on tag 4, and
`numeric_profile` on tag 5. The semantic config pins the accepted
capacity-audit-config SHA256
`c1265cc3834522a5e9958950227c4b30421b36690e56be4d822ac66b61f6dc05`
and capacity-audit SHA256
`4854ae79bc8a24e292be356180420215dd56e70a04c10b12d2f8b242a75e9a5d`.
The checked capacity config and semantic artifact identity establish the frozen
SHA256
`9b1656a48ed3cfbf1577b7a404688affb49b793f9c556ba2d2fb46b41238c61c`,
schema-version-4 Q4 manifest SHA256
`f4d7e53eb14110c855fe684f34ee536465f7eae6601f9555859ccce28edfb465`,
GGUF SHA256
`3c53d019a0ff0832ff5f7b08508e8b5c3b78b747871060deb2db3375fcfb6677`,
tokenizer SHA256
`b5cbdfa8aa7c54c8c5af85b78c309c54a5f2749a20468bf6f60eee007fe6fec1`,
model-source revision `8e91527b3276e0565154935e84a08bf0137ed99f`, runtime
revision `1511ce3bc3f087376c8526b4ad07100bfabb277f`, simple file name
`rinna-xsmall-q4_k_m.gguf`, quantization `Q4_K_M`, and 200 cases.

Semantic outcomes are `SUCCESS` on value 1, `BACKEND_ERROR` on value 2, and
`INVALID_RESPONSE` on value 3; zero remains unspecified. Each case stores
`source_index` on tag 1, `outcome` on tag 2, `canonical_status_code` on tag 3,
optional `response_token_matches_request` on tag 4,
`response_segment_orders` on tag 5, optional `merged_top_output` on tag 6,
optional `response_within_request_bounds` on tag 7, optional
`response_segment_order_count` on tag 8, and optional
`response_candidate_id_count` on tag 9. A segment order stores `segment_id` on
tag 1 and `candidate_ids` on tag 2. A backend error stores a non-OK canonical
`Rank` status and none of the response fields. Every successful `Rank` return
stores the token-match bit, exact segment-order and candidate-ID counts, and the
request-bounds bit. It stores raw orders only when the bounds bit is true, in
which case both stored counts must equal the raw orders. Token mismatch is
classified first as `INVALID_RESPONSE` with canonical `ABORTED`, including
when the response is also out of bounds. A matching-token response outside the
request bounds is `INVALID_RESPONSE` with canonical `INVALID_ARGUMENT`. With a
matching token and true bounds bit, replaying the shared pure merger must either
fail with exactly the stored canonical code or succeed. Success requires a true
token bit, a true bounds bit, canonical `OK`, a valid pure merge, and a present
merged output equal to deterministic recomputation, including when that string
is empty. Every returned response case omits output unless it is successful.

The full-model `LlamaCandidateRanker::Create` path must accept the expected
seven-field model and runtime identity. It parses and validates the manifest,
compares every expected field, and returns `FailedPrecondition` before opening
the GGUF on a mismatch. Only then may it open the manifest-owned simple name,
hash and load through the same protected file handle, validate the full model
metadata, and construct the configured production context. The semantic runner
must use this checked creation path and compare the exposed verified identity
again before its first `Rank` call.

The semantic CLI accepts only `--config`, `--capacity_audit_config`,
`--frozen_corpus`, `--capacity_audit`, `--manifest`, `--model_directory`,
`--output_binary`, and `--output_textproto`. It reads the checked capacity
config, frozen corpus, and capacity audit once and hashes each retained byte
string before parsing. The two capacity hashes must equal the semantic config,
and the frozen hash must equal the retained capacity config. It rejects unknown
fields recursively, and the semantic library independently rejects unknown
frozen-corpus fields recursively before any backend creation or callback. It
validates the actual capacity config through
`ValidateAjimeeCapacityAuditConfig`, validates the complete frozen corpus and
exact configured case count, then calls `ValidateAjimeeCapacityAudit` with that
same config and requires every result to pass. Any failure before backend
creation produces no semantic artifact. It converts each frozen request
exhaustively and calls production `Rank` exactly once in corpus order, without
retries, timing, deadlines, cancellation, warmups, evaluation scoring, or
policy overrides.

For every successful backend response, the runner first records the exact
segment-order and candidate-ID counts and whether those totals stay within the
corresponding frozen request counts. Token mismatch takes precedence over that
bounds decision. An out-of-bounds response stores no raw orders; the runner
records its typed invalid result and continues with the next case. A bounded
response preserves the raw response order without sorting. A matching-token,
bounded response is passed to `BuildCandidateRankerMergeOrder` with the exact
request and token. The helper supplies the complete product order: protected
slots stay pinned, omitted unprotected candidates retain relative Mozc order,
and omitted segments remain unchanged. The merged top output is the
concatenation, in request-segment order, of the request candidate value named by
position zero of each complete merged order. It never uses the stored Mozc
baseline to choose or reconstruct that output. Validation recomputes request
bounds and stored raw-order counts, replays every eligible pure merge, and
recomputes every successful output instead of trusting stored fields.

The semantic artifact contains no timings, timestamps, host identity, scores,
model tokens, status messages, accepted answers, or answer-artifact hash. The
CLI has no answer, source-JSON, latency, deadline, or profile-override path. It
builds and validates the complete artifact in memory, then produces
deterministic binary and review text. It writes typed backend and invalid
response results for every structurally valid case, but returns failure after
writing if any case is not successful. Repeated successful runs must be
byte-identical.

The accepted semantic run used the 284-byte checked config with SHA256
`982b1c8a6c8748a6b30bd3f60be1850328454522c20916c307dfd6d091bea223`.
Two fresh sequential processes produced byte-identical binary and review-text
artifacts. The accepted binary is 78,338 bytes with SHA256
`7b1aa411809ca6e74bad9f5f96bcb9706a735833c98c94b4d4a9d39be9ac38c8`,
and the accepted review text is 642,545 bytes with SHA256
`9012fcf4eed6afe2a9b035904ea424fe0104eca06b81008fe96556883259896e`.
All 200 cases are `AJIMEE_SEMANTIC_SUCCESS`; there are zero backend errors and
zero invalid responses.

The answer-aware quality reporter has a checked schema-version-1 config.
`schema_version` is tag 1, `answer_corpus_schema_version` is tag 2,
`frozen_corpus_schema_version` is tag 3, `capacity_audit_schema_version` is tag
4, `semantic_results_schema_version` is tag 5, and
`quality_report_schema_version` is tag 6. The exact binary hashes for the input
corpus, answer corpus, frozen corpus, capacity-audit config, capacity audit, and
semantic results occupy tags 7 through 12 in that order. Expected total,
context, and no-context counts occupy tags 13 through 15, `expected_source` is
tag 16, `metric_definition_version` is tag 17, and
`exact_bootstrap_definition_version` is tag 18. Schema version 1 accepts report,
metric, and bootstrap version 1 only, and requires 200 total, 100 context, and
100 no-context cases. The known binary identities are input
`55fc26b4d86c5d30c2eae5f40fcf1cd9f9b4822ad5a8bae424cdc9ed10e82848`,
answer
`819465115d76cca65eb9d6c8f3f60f8850723376b440eb5c41f0cd8fd7b2c389`,
frozen
`9b1656a48ed3cfbf1577b7a404688affb49b793f9c556ba2d2fb46b41238c61c`,
capacity-audit config
`c1265cc3834522a5e9958950227c4b30421b36690e56be4d822ac66b61f6dc05`,
and capacity audit
`4854ae79bc8a24e292be356180420215dd56e70a04c10b12d2f8b242a75e9a5d`.
The accepted checked reporter config is 1,675 bytes with SHA256
`c8675a28f645a2edccf843894261204bef5c6aa9843189d456394c2048128f6e`.
Its pinned semantic-results hash is the accepted binary above. The former
provisional label is removed because the semantic identity matched the pinned
capacity and backend identities.

Quality-report slices are `OVERALL` on value 1, `HAS_CONTEXT` on value 2, and
`NO_CONTEXT` on value 3; zero remains unspecified. `AjimeeExactRatio` stores a
signed numerator on tag 1 and a positive denominator on tag 2.
`AjimeeCharacterError` stores the edit-distance sum on tag 1 and the selected
reference Unicode-scalar count on tag 2. The exact paired-bootstrap message
stores definition version, sample size, win count, regression count, tie count,
signed lower endpoint in gain cases, signed upper endpoint in gain cases, and
the case-count denominator on tags 1 through 8 respectively.

Each `AjimeeMetricSlice` stores the slice and case count on tags 1 and 2;
semantic success, backend error, and invalid-response counts on tags 3 through
5; oracle-covered, baseline-correct, effective-model-correct,
retained-correct, win, regression, and unchanged-incorrect counts on tags 6
through 12; baseline and effective-model character error on tags 13 and 14;
the exact paired interval on tag 15; and exact baseline accuracy,
effective-model accuracy, accuracy gain, oracle coverage, baseline conditional
accuracy, model conditional accuracy, and optional retention ratios on tags 16
through 22. Retention is absent only when the baseline-correct denominator is
zero. Conditional accuracy is absent only when the oracle denominator is zero.

Capacity coverage is aggregated without changing the capacity-audit meanings.
A decode-batch histogram bin stores its batch count and case count on tags 1 and
2. A limiter summary stores limit enum, affected-case count, affected-segment
count, and omitted-candidate count on tags 1 through 4. Each
`AjimeeCapacityCoverageSlice` stores slice, case count, pass count,
request-limit count, audit-error count, componentwise sum of the existing 22
usage fields, componentwise maximum of those fields, cases with window
omissions, cases with candidate-capacity omissions, cases with
segment-capacity omissions, the ascending batch histogram, and limiter
summaries in capacity-enum order on tags 1 through 12. The componentwise sum is
retained as an exact transform even for fields whose source meaning is a
per-case maximum; the componentwise maximum supplies the corpus or slice
maximum needed by the report.

`AjimeeQualityReportIdentity` stores the raw reporter-config hash, input hash,
answer hash, frozen hash, capacity-config hash, capacity-audit hash,
semantic-results hash, complete AJIMEE source identity, exact nested capacity
and backend identity, semantic numeric profile, metric-definition version, and
bootstrap-definition version on tags 1 through 12. The quality report stores
its schema version on tag 1, identity on tag 2, `semantic_all_success` on tag 3,
exactly three metric slices in enum order on tag 4, and exactly three capacity
coverage slices in the same order on tag 5. It has no nDCG field. It also has no
accepted answer, baseline, model output, per-case text, score, token, path,
timestamp, host identity, latency, or timeout field.

Top-1 correctness is exact byte equality with any accepted whole output after
all relevant protobuf strings pass UTF-8 validation. No Unicode normalization,
case folding, width folding, or punctuation mapping is applied. A semantic
success uses its independently recomputed merged top. A backend error or invalid
response leaves the product's Mozc order unchanged, so its effective output is
the frozen baseline. Those outcomes remain separate reliability counts even
when the effective output is correct. For each slice, a win is baseline-wrong
and effective-model-correct, a regression is baseline-correct and
effective-model-wrong, retained-correct means both are correct, and
unchanged-incorrect means both are wrong. Accuracy gain is wins minus
regressions over the complete slice. Retention is retained-correct over
baseline-correct.

Full Mozc oracle coverage ignores ranker windows, capacity selection, and
protected status. For each accepted output, a bounded dynamic program begins at
byte offset zero. At each frozen conversion segment it advances every reachable
offset by every candidate value that is an exact prefix at that offset. The
case is covered only if one candidate from every segment reaches the complete
accepted-output byte length. UTF-8 validation makes byte prefix boundaries
unambiguous. Conditional accuracy uses oracle-covered cases as its denominator.
Any correct baseline or valid merged output must therefore be oracle covered.
Window and capacity effects remain visible in the separate accepted-audit
coverage and are never relabeled as oracle failures.

Character error uses unit-cost Levenshtein distance over Unicode scalar values,
not bytes, UTF-16 code units, or grapheme clusters. It applies no normalization.
For each hypothesis, the chosen accepted reference is the lowest source ordinal
among references attaining the minimum edit distance. Stable duplicate removal
retains that first ordinal. The corpus or slice character-error ratio is the sum
of chosen distances over the sum of chosen-reference Unicode-scalar lengths.
The numerator and denominator remain integers in the binary and JSON reports.

Bootstrap version 1 is the exact paired empirical percentile distribution and
has no random seed or finite replicate count. Each case contributes accuracy
delta negative one, zero, or positive one. Let `W`, `R`, and `T` be the win,
regression, and tie counts. Integer dynamic-program mass begins with one at sum
zero. Each of `N` resampling steps sends mass at sum `s`, multiplied by `R`,
`T`, and `W`, to sums `s - 1`, `s`, and `s + 1`. Arbitrary-precision integer
mass totals exactly `N` to the power `N`. The 95% lower endpoint is the smallest
sum whose cumulative mass times 40 is at least the total mass. The upper
endpoint is the smallest sum whose cumulative mass times 40 is at least 39
times the total mass. Endpoints are stored as signed gain-case numerators over
`N`; no floating-point calculation enters evidence or gate comparison.

The reporter reads each input file once, hashes those retained bytes before
parsing, rejects unknown fields recursively, and validates the checked config,
actual capacity config, frozen corpus, answers, capacity audit, and semantic
results in that order. Source identities and source IDs must match exactly in
the same ordinal order. It never repairs or sorts malformed input. The
raw-hash-pinned semantic artifact is already generated and validated by the
shared C++ pure merger. The model-free Python reporter defensively replays that
contract for every successful semantic case, rejects any stored top that
differs from recomputation, and cross-checks the replay against fixtures whose
outputs are fixed by the C++ merger tests. Capacity must be all-pass, its
frozen hash must match the corpus, and its complete backend identity must equal
the semantic identity. The reporter verifies the count algebra, including total cases equal
to retained plus wins plus regressions plus unchanged incorrect, baseline
correct equal to retained plus regressions, model correct equal to retained plus
wins, and both correct counts no greater than oracle coverage.

The `report_ajimee_quality` target is defined by
`report_ajimee_quality_main.py` and uses the checked
`ajimee_quality_report_config.textproto`. Its only flags are `--config`,
`--capacity_audit_config`, `--frozen_corpus`, `--answer_corpus`,
`--capacity_audit`, `--semantic_results`, `--output_binary`, and
`--output_json`, all required paths. It has no model, manifest, model-directory,
source-JSON, input-corpus, policy, limit, deadline, latency, or host flag and
does not link llama.cpp. It builds and validates both outputs fully in memory.
The binary uses deterministic protobuf serialization. Canonical UTF-8 JSON
emits fields in tag
order, represents every integer and exact-ratio component as a decimal string,
uses LF line endings and one final newline, and emits no floating-point value.
Repeated runs over the same hashes must be byte-identical, and neither output
nor any validation status has a field that copies answer, baseline, or
model-output text. Privacy tests inject unique poison strings into those inputs
and require that neither serialized report nor any returned status contains the
poison.

Focused reporter verification covers exact-match and outcome count algebra,
legal duplicate answers, multi-reference character-error ties, non-BMP Unicode
scalars, canonically equivalent but byte-distinct strings, cross-segment oracle
paths, non-success baseline retention, and brute-force agreement with the exact
bootstrap dynamic program for small case counts. It rejects each independent
hash, schema, source, order, unknown-field, semantic-outcome, recomputed-top,
and capacity-aggregation inconsistency. Synthetic coverage tests fix
componentwise sums and maxima, ascending batch histograms, enum-ordered limiter
summaries, and all three slices. Two builds from the same validated inputs must
produce byte-identical binary and JSON reports.

The checked pre-model facts are fixed regression assertions for the reporter.
Mozc baseline top 1 is correct in 103 of 200 cases and the full Mozc oracle
covers 180 of 200. In the context slice the corresponding counts are 46 of 100
and 88 of 100. In the no-context slice they are 57 of 100 and 92 of 100. Mozc
baseline character error is exactly 251 over 4246 Unicode scalars overall, 158
over 2039 with context, and 93 over 2207 without context. These facts require
only the checked frozen and answer artifacts and must not trigger model loading
or inference.

The accepted deterministic Q4_K_M quality run used the 1,675-byte reporter
config with SHA256
`c8675a28f645a2edccf843894261204bef5c6aa9843189d456394c2048128f6e`.
Two fresh reporter processes produced byte-identical outputs. The accepted
binary is 2,513 bytes with SHA256
`c3cd5b9819873647f00f7d2a8b4e110358ac0b86b83d06b19d650f174f6987e4`,
and the accepted JSON is 12,974 bytes with SHA256
`0460c0d5e4f684c5f0ec57f66e3ec8e3da42532c4c71895dc4cf6dcc061df4c5`.
All 200 semantic cases succeeded. Overall Mozc and model top-1 correctness are
103 and 17 of 200 respectively: the model retained 16 correct cases, won one,
and regressed on 87. Mozc and model character error are 251 over 4246 and 1837
over 4254 Unicode scalars respectively. The exact paired-bootstrap 95% gain
interval is -100 through -72 cases over 200. Context correctness falls from 46
to 14 of 100, while no-context correctness falls from 57 to 3 of 100. The full
Mozc oracle remains 180 of 200. Capacity coverage passes all 200 cases; 174
cases have window omissions, three have candidate-capacity omissions, and none
has a segment-capacity omission.

Promotion gates are zero backend errors and invalid responses, deterministic
order over 100 warm
runs, at least a two percentage point overall top-1 gain over Mozc with the
paired bootstrap 95% lower bound above zero, at least 98% retention when Mozc is
already correct, no slice with at least 100 cases losing more than two points,
no nDCG@5 regression on the separately annotated corpus, no more than 0.5 point
Q4_K_M top-1 loss against the checked F16 artifact,
warm p95 below the chosen IME deadline, and warm p99 below the configured
maximum wait. Candidate permutation cases must preserve every mapped score and
the semantic winner.

The accepted Q4_K_M result fails the quality promotion gate decisively. Its
overall accuracy loses 86 cases, its paired-bootstrap interval is entirely
negative, its retention is 16 of 103, and both slices of 100 cases lose far more
than two percentage points. This Q4_K_M artifact is blocked from promotion
and from packaging as the default model regardless of its semantic reliability
and capacity success.

The retained artifacts identify the dominant failure pattern. Of 579
segments whose top candidate changed, 440 promote a candidate whose value is
exactly the target segment key, and 85 of the 87 lost Mozc-correct cases contain
at least one such promotion. Seventy of those cases contain no other kind of
change. The model selects a higher Mozc cost in 538 of the 579 substitutions.
All 150 cases with two or more substitutions are wrong, so multiple
substitutions are strongly associated with the observed whole-output loss. The record
places the full reading and `対象:<segment key>` before `文:<candidate>`.
Aggregate failures show that exact-key copying is the dominant failure pattern.
The controlled F32 intervention below establishes the explicit target-key field
as the dominant causal contributor for source 125. Corpus-wide causality remains
for the predeclared paired development diagnostic. The implementation follows
the rejected narrative exactly, so this is a scoring-design defect rather than
code drift. Capacity, merge, reporting, and runtime determinism do not explain
the failure. The observed selections reject visible Unicode-scalar shortening
as the mechanism, but do not exclude model-token length, terminal effects, or a
preference for longer visible strings.

A model-free counterfactual that restores Mozc only for newly promoted exact-key
copies reaches 113 of 200, with 23 wins and 13 regressions. Restricting the same
diagnostic to zero-based original rank at most one reaches 117 of 200 with 100 of 103 Mozc-correct
cases retained. These calculations show that non-copy scores may contain useful
signal and that Mozc must remain a strong prior. They are post-hoc attribution,
not production policies, and neither exact-key filtering nor an AJIMEE-derived
rank cutoff may be implemented.

The completed source-125 intervention measures the causal contribution of the
explicit target-key field. It compares only `満たせば` and `みたせば`, holds the
complete reading and frozen baseline-right sentence constant, and changes only
the serialized segment key from `みたせば` to `満たせば`. Let the margin be the
score of `みたせば` minus the score of `満たせば`. Under the strict F32-weight,
F32-KV, flash-disabled reference profile, the actual-key margin is approximately
+7.79768 and the counterbalanced-key margin is approximately +0.04521. Shared
trie and independent complete-sequence scoring pass the `1e-4` edge and total
comparison in both conditions. Changing only the target key therefore reduces
the fixed candidate margin by approximately 7.75247 without reversing the
winner. Because `みたせば` is no longer the target key in the counterbalanced
condition, its remaining +0.04521 is a small preference for that fixed kana
candidate, not a residual copy margin. This establishes the target-key field as
the dominant causal contributor in source 125 while leaving that small fixed
candidate preference unattributed.

The checked F16 configured profile likewise moves from approximately +7.79296
to +0.05727 in the shared production scoring graph. In the matched F32
configured profile, shared and independent margins move from approximately
+7.78934 and +7.78950 to +0.05635 and +0.05764. The checked Q4_K_M configured
profile moves from approximately +9.63284 to +1.56262 in the shared graph; its
independent configured margins are approximately +9.57230 and +1.98637. F32
therefore excludes quantization as a necessary cause. The matched Q4_K_M
configured model/runtime cell materially amplifies the counterbalanced
fixed candidate preference in this probe and remains part of the rejected cell;
this single comparison does not attribute that amplification to quantization
alone.

Fixed `1e-4` shared-versus-independent equivalence is retained only for the
strict F32 reference gate. Within each intervention and numeric profile, F16
and Q4_K_M use identical token and edge metadata across the shared-trie and
independent scoring graph implementations. Their graph-specific margins measure
whether the lower-precision model/runtime cells preserve or amplify the
intervention effect; they do not redefine the strict gate. This probe is a
causal diagnosis only and cannot select coefficients or promotion policy.

Objective selection uses checked Mozc fixtures rather than AJIMEE. The
development source is `src/data/test/quality_regression_test/oss.tsv`, whose
SHA256 is
`053a43e72d174da43b351278cadc471abe08cea9fe98829dafe3d152580fcff0`.
Parse rows exactly as `QualityRegressionUtil::TestItem::ParseFromTSV`, retain
only `Conversion Expected` with parsed rank zero, leave the parsed key
unchanged, and run the importer only under the pinned Windows rule where
`TextNormalizer::NormalizeText` equals
`NormalizeTextWithFlag(TextNormalizer::kAll)`. The pinned mapping is U+301C to
U+FF5E and U+2212 to U+FF0D, and neither source character occurs in these
fixtures. Normalization applies only to the expected value. The importer
requires 500 parsed OSS rows, 408 parsed
`Conversion Expected` rows, and 220 rank-zero rows before deduplication.
Stable-deduplicate exact `(parsed key, normalized expected value)` pairs by
keeping their first physical source line. The sole development
duplicate is `じょうきょうをちゅうしする` / `状況を注視する` on one-based
physical lines 340 and 469; keep line 340. The result is 219 cases. A case is
identified by its checked source identity and one-based physical line, never by
the nonunique label column. The one-shot holdout is
`src/data/test/quality_regression_test/regression.tsv`, whose SHA256 is
`72c838c5422f04c8246114074ea9c82e577155e8d7ef653a21c2dbb75bb3dff3`;
require 101 parsed rows, all 101 parsed as `Conversion Expected`, and 72
rank-zero rows before the same filter yields 72 cases without duplicates whose
parsed keys are disjoint from development. Rank-positive,
Match, negative, suggestion, and `myoji.tsv` rows are not whole-output top-one
labels and are excluded. The repository-owned
`src/data/dictionary_oss/evaluation.tsv`, SHA256
`b31842bef8b2bdaef4d13a1f19581366749285bfa7365de8b302456956a3e8ce`,
aligns in order with all 500 parsed OSS rows followed by all 101 parsed
regression rows. After the declared filter and stable deduplication, its stored
top output matches 164 of 219 development answers and 42 of 72 holdout answers,
with no status or value inconsistency. This is historical Mozc baseline
provenance generated by the repository BUILD rule and protected by its diff
test, never an answer source. The current deterministic development freeze and
answer-only report must independently re-establish 164 of 219 before any
development model output is opened. The 42-of-72 holdout baseline is checked
only inside the single post-selection holdout report; it is not opened during
development.

Both TSV inputs are valid UTF-8 with LF endings and contain no BOM, carriage
return, NUL, empty field, disabled row, or character affected by the pinned
Windows normalization rule. Filtered development and holdout also have no raw
key, normalized answer, or exact-pair intersection. The fixtures are public
Mozc repository data declared as notice material, so every derived distributed
artifact retains the Mozc notices. The holdout is a procedurally one-shot test,
not a secret or pretraining-independent benchmark; its answer isolation is
enforced by separate input, answer, runner, and reporter paths.

The importer pins Windows text normalization and emits separate input and
answer protobufs. The freezer reads only inputs, uses the pinned Mozc revision,
OSS data, default desktop request and config, fixed UTC clock, fresh profile,
typing correction enabled, and candidate ranking disabled, then snapshots the
production ranking seam without committing or finishing conversion. Two fresh
freezes must be byte-identical. Model runners read only frozen requests; only
the reporter reads answers. Predeclared slices are development versus holdout,
single versus multiple conversion segments, selected-window exact-key-copy
exposure, full and selected-window oracle coverage, baseline-correct retention,
and baseline-incorrect opportunity. These fixtures have no external history;
multiple segments measure sentence-internal context and are not called an
external-context slice.

The paired key-copy diagnostic is predeclared at the frozen-segment level. A
segment is eligible only when its Mozc top is not the segment key, the selected
candidate window contains an exact-key value, and there is no natural-context
key exposure before the candidate. To test exposure, apply the manifest text
normalization separately to the segment key, external preceding text, and each
earlier segment's frozen baseline value, then compare exact UTF-8 byte
substrings. The full reading and target-key fields are the intervention
variables and are not eligibility filters. The target candidate, later segment
baselines, and external following text occur at or after the candidate and are
also excluded from the pre-candidate exposure test. For each eligible
segment, record only whether the scored top equals the key. For a candidate
objective relative to structured-with-target, `removed` counts paired segments
whose indicator changes from one to zero and `added` counts changes from zero to
one. An exact paired improvement requires `removed > added`. The
structured-with-target versus
structured-without-target pair measures the explicit target-field effect. The
structured-without-target versus natural-text-only pair measures the combined
effect of the remaining structured metadata. Because that second contrast
removes mode and fixed labels together with the full reading, it cannot by
itself attribute any improvement solely to the reading. If natural-text-only
still promotes exact-key values on this clean subset, upstream serialized key
exposure cannot explain them and the remaining mechanism is the raw
continuation objective or base-model fit.

Before any development model runner is invoked, a development-only baseline
reporter reads the accepted development frozen corpus and development answer
corpus. Its checked config pins their raw SHA256 values, both schema versions,
the complete development corpus identity, the expected 219 cases, and the
predeclared 164 correct cases. It joins the two corpora only by identical
ordered source lines and compares `mozc_baseline_output` with
`normalized_whole_output` by exact UTF-8 bytes without further normalization or
deduplication. It emits only provenance, aggregate counts, and the shared exact
ratio. It accepts no holdout, model, manifest, capacity, score, or source-file
input. Two fresh processes must produce byte-identical binary and review
artifacts establishing 164 of 219 before any model output is opened.

`quality_regression_baseline_report.proto` schema version 1 defines
`QualityRegressionDevelopmentBaselineReportConfig` with config, frozen,
answer, and report schema versions on tags 1 through 4, complete expected
corpus identity on tag 5, raw frozen and answer SHA256 values on tags 6 and 7,
and expected case and correct counts on tags 8 and 9. Its report identity stores
the raw config, frozen, and answer hashes on tags 1 through 3 and the complete
public corpus identity on tag 4. The report stores schema version, identity,
case count, correct count, and shared `ExactRatio` top-output accuracy on tags 1
through 5. It persists no cases, source lines, readings, answers, baseline
outputs, segments, candidates, model data, host data, paths, or timestamps.
The manual `report_quality_regression_development_baseline` command accepts
only `--config`, `--frozen_corpus`, `--answer_corpus`, `--output_binary`, and
`--output_textproto`; both complete outputs are built and validated in memory
before either write.

That gate passes. The checked development baseline-report config is 1,495 bytes
with SHA256
`9a5058100568bf357f56e9f75b87292f65155ac2d3faab92dc61985711ee944f`.
Two fresh reporter processes produced the same 1,017-byte aggregate binary with
SHA256
`3e77232d764ce39d0075884e3c6507730efbc68bc5f46b0606bee2f5977490a9`
and the same 1,583-byte review text with SHA256
`51207a57580d7e15ba600df8a7c6024eebed4d18341a3c9b0a8b6edde5370501`.
Both artifacts bind the accepted development frozen and answer hashes and
contain only the public development identity, case count 219, correct count
164, and exact ratio 164 over 219. No model output or holdout answer was opened.

After that baseline gate passes, predeclare and compare all three F16 objectives
on development: the rejected structured record, an
otherwise identical target-field ablation, and the natural-text-only record
`external preceding + baseline left + candidate + baseline right + external
following`. This three-way comparison distinguishes the source-125 target-field
effect from remaining structured-field exposure and from the raw continuation
objective without selecting a follow-up from an intermediate result.

An eligible corrected layout passes development only with at least five net
additional correct outputs over the retained Mozc baseline, an exact paired
bootstrap lower gain bound above zero, retention of all but at most three
Mozc-correct outputs, zero backend errors and invalid responses, deterministic
repeat and candidate-permutation results, byte-identical native candidate
coverage across all three F16 objectives, `removed > added` in the paired clean
key-copy diagnostic, and no Mozc-correct regression containing a newly promoted
exact-key value. The rejected structured layout is reference-only. If both
corrected layouts pass, select by larger net gain, then larger retained-correct
count, then fewer promoted clean exact-key segments, then lower model CER, then
record-layout enum order. Emit one immutable selection artifact that binds the
manifest, scores, development report, and any accepted policy parameters.

Run only that selected objective once on the 72-case holdout. Its predeclared
gate is at least two net additional correct outputs over the independently
confirmed historical baseline of 42, an exact paired-bootstrap lower gain bound
above zero, retention of all 42 Mozc-correct outputs, zero backend errors and
invalid responses, deterministic repeat and permutation results, and no
Mozc-correct regression containing a newly promoted exact-key value. Holdout
answers are first opened by this final reporter. Do not revise the objective or
parameters from holdout or AJIMEE results; a holdout failure ends this branch.
Only a passing fixed objective proceeds to AJIMEE, then to the Q4_K_M versus
F16 gate.

Objective evaluation advances the model manifest to schema version 5 with no
schema-version-4 compatibility path in the current runtime. `ScoringTemplate`
adds a required `RecordLayout` on tag 14 with exactly
`STRUCTURED_WITH_TARGET`, `STRUCTURED_WITHOUT_TARGET`, and
`NATURAL_TEXT_ONLY`; unspecified and unknown values fail. The existing
template version remains `rinna-continuation-v2` because token IDs, literals,
normalization, and context policy do not change. Every existing literal remains
required in all three manifests, while runtime serialization dispatches only on
the explicit layout and never infers a layout from empty literals, the template
version, or a missing field.

All layouts first form the same natural body: external preceding text, frozen
baseline values before the target, the target candidate, frozen baseline values
after the target, and external following text. Candidate-ending diagnostic
records stop immediately after the target candidate. `STRUCTURED_WITH_TARGET`
prepends the current mode, full-reading, target-key, and text fields and must
reproduce the archived schema-version-4 normalized record bytes exactly.
`STRUCTURED_WITHOUT_TARGET` omits the complete target-key field and its trailing
separator while retaining the mode, full reading, and text field.
`NATURAL_TEXT_ONLY` serializes only the natural body without a synthetic prefix
or separator. UTF-8 validation, whole-record normalization, byte and token
limits, BOS and terminal handling, baseline selection, bounded candidate
selection, and graph construction remain identical.

Create three checked schema-version-5 F16 manifests named
`manifest-f16-structured-with-target-v1.textproto`,
`manifest-f16-structured-without-target-v1.textproto`, and
`manifest-f16-natural-text-only-v1.textproto`. They differ only in
`record_layout`. Their checked raw identities are respectively 2,077 bytes at
SHA256 `2209e8ebcc9876e9427d934f3b9d739a65fb498d2160d64d742fc719a45271eb`,
2,080 bytes at SHA256
`66c620673895d5462539b8c2073c8d4f1615c51c09895d85c9336246c0c08bff`,
and 2,072 bytes at SHA256
`be316282bcf1e031b55228a8f99c78dda7b19961828c56fd5bf70f90881657f9`.
Capacity and semantic
artifacts bind the raw manifest hash rather than duplicating the layout enum.
The schema-version-4 AJIMEE capacity, semantic, and quality artifacts remain
immutable evidence of the rejected objective and are never relabeled as
schema-version-5 results. Before any checked config is replaced, archive the
exact raw schema-version-4 manifest, configs, binary artifacts, and review text
with a deterministic hash index; hashes without the raw inputs are
insufficient. Current model loaders reject schema version 4, schema-version-5
objective validators do not consume those opaque archived results, and
packaging excludes them. Any
current-code reproduction uses an explicit schema-version-5
structured-with-target manifest and creates new artifacts. New development
capacity, semantic, score, and quality configs require manifest schema version
5 and pin new hashes; no old AJIMEE config or report is silently migrated.

The quality-regression pipeline reuses wire-safe evaluation primitives without
reusing AJIMEE-specific case semantics. Move the existing source and Mozc
identity messages to `evaluation_artifact.proto`, the existing
`FrozenCandidateRankerToken`, candidate, segment, and request messages to
`frozen_candidate_ranker.proto`, and the exact ratio, character-error, and
paired-bootstrap messages to `exact_quality_metrics.proto`. Preserve every
package name, field name, tag, enum value, and populated value when changing an
AJIMEE field's embedded type. Protobuf binary wire bytes contain no message type
name, so the accepted AJIMEE binary hashes must remain byte-identical; focused
tests pin all accepted input, answer, frozen, capacity, semantic, and quality
binary hashes across this refactor.
The config-independent request structure validation and exhaustive
proto-to-`CandidateRankerRequest` conversion move to
`frozen_candidate_ranker_util.h` and `frozen_candidate_ranker_util.cc` so both
AJIMEE and quality-regression evaluation use one implementation. The shared
utility owns recursive request-local token, mode, context, segment, candidate,
and ID validation plus exhaustive conversion. Corpus-specific validators retain
their schema, identity, order, and provenance rules together with every
case-to-request reading, context, token, source-index, and candidate-zero
baseline relationship.

`quality_regression_corpus.proto` schema version 1 defines corpus role
`DEVELOPMENT=1` and `HOLDOUT=2`. `QualityRegressionCorpusIdentity` stores role
on tag 1, checked source identity on tag 2, parser-definition version on tag 3,
normalization-definition version on tag 4, and import-config SHA256 on tag 5.
An input case stores one-based physical source line on tag 1 and reading on tag
2; an answer case stores the same line on tag 1 and one normalized whole-output
answer on tag 2. Input and answer corpora each store schema version, identity,
and ordered cases on tags 1 through 3. No label, command, rank, or expected text
enters the input corpus.

The checked import config stores its schema, input schema, answer schema,
parser-definition version, normalization-definition version, and exactly two
ordered partitions on tags 1 through 6. Each partition stores role, source
identity, expected parsed-row count, expected `Conversion Expected` count,
expected rank-zero count, expected duplicate-pair count, and expected final
case count on tags 1 through 7. The importer reads both checked raw buffers once,
hashes before parsing, calls the production TSV parser, enforces every count and
the single declared duplicate, rejects cross-partition key or answer overlap,
builds all four corpora fully in memory, validates them, then writes. The
development runner has no flag for the development answer corpus or either
holdout corpus.

Two fresh importer runs produced byte-identical artifacts. The development
input binary is 7,804 bytes with SHA256
`7f06025eacc5d43d7a56aa8b92f79a982ed54699960a73b764fb322c0d84272f`;
its 16,063-byte review text has SHA256
`c4add91fe4327abb76fa62072f39971af6c63c06334a34d5df0966894c9daa71`.
The development answer binary is 6,176 bytes with SHA256
`8d76595da3ccaa609903fc48a84e1ef47ed179c3936ebad06e6beaf242bd0d79`;
its 17,939-byte review text has SHA256
`7a144a0e9779f3908c0a7930b721ec0536f86b82bf7c035eb840beea882b9fb0`.
The holdout input binary is 2,811 bytes with SHA256
`97e5b7eb6a6c7f831f5306725a552eee6d398def178fb5c38601e962b5084087`;
its 5,750-byte review text has SHA256
`a6de8f5ea2c43ff74b7c85603367bb3124e68bd5ddec36b0ef46fc4cdf49196a`.
The holdout answer binary is 2,490 bytes with SHA256
`62e661495adadf9913cd0135dcf3969fd6ccf576c1a7cb78402c1c1362a39f3c`;
its 6,581-byte review text has SHA256
`380209b6e7be202e2b54881db3125438a3bc1a6f89614e457c09529220f7f159`.
All four corpus identities bind the 2,422-byte checked import config at SHA256
`3cfb88ca988d9c46c3bca3095c9e8ee3b61cb43fce6df3c5be0d4ce100abbba5`.

`quality_regression_frozen_corpus.proto` schema version 1 defines a freezer
config containing schema versions, expected corpus identity, input-corpus hash,
case count, and exact Mozc identity on tags 1 through 7. A frozen case stores
source line, conversion reading, Mozc baseline output, and the generic frozen
ranker request on tags 1 through 4. A frozen corpus stores schema version,
corpus identity, input hash, Mozc identity, ordered cases, and freezer-config
hash on tags 1 through 6. The config hash is the SHA256 of the retained raw
config buffer, not a reserialization. Its Mozc field uses the shared evaluation
Mozc identity and records the actual data, request, config, and clock identities
owned by the freezer.
`ValidateQualityRegressionFrozenCorpusStructure` performs the complete
config-independent schema, recursive unknown-field, public identity, hash
format, Mozc identity, case order, request, segment, candidate, reading, and
candidate-zero baseline validation. The freezer-specific validator calls that
shared structural validator first and then compares the exact checked config,
input hash, case count, Mozc identity, and raw freezer-config hash. Later
evaluation code therefore validates the accepted frozen bytes without retaining
or reopening the freezer config.

The reusable freezer has two build boundaries. The lightweight
`evaluation_freezer.h` and `evaluation_freezer.cc` target owns the canonical UTC
clock parser, deterministic protobuf message hashing, exact request copying,
conversion setup, and the fixed-clock conversion session. It depends on the
generic frozen ranker request proto but on no AJIMEE or quality-regression corpus
proto. `EvaluationFreezerSession::Create` takes the canonical clock string, the
desktop `Request`, desktop `Config`, and a borrowed const `ConverterInterface`.
It
rejects any present candidate-ranking config, computes the deterministic request
and config hashes, and installs one nested `ScopedClockMock` for the complete
session lifetime. `EvaluationFreezeCaseInput` contains the positive request
sequence, composer preedit text, preceding text, following text, and an explicit
`FrozenReadingSource` of either `PREEDIT_TEXT` or `CONVERSION_QUERY`.
`EvaluationFreezeCaseResult` contains the composer conversion query, history
reconstruction result, candidate-zero baseline output, and the complete generic
frozen ranker request. Unknown reading sources and candidate-ranker modes fail;
there is no unspecified-mode serialization.

The separate heavy `evaluation_freezer_runtime.h` and
`evaluation_freezer_runtime.cc` target owns only the real Mozc environment.
`EvaluationFreezerRuntime::Create` takes the data path, expected data SHA256,
data type, and canonical clock string. It reads `mozc.data` once, hashes that
retained buffer, installs an outer fixed clock before Engine construction,
creates a unique temporary profile while preserving the previous global profile,
constructs `DataManager` from the same retained bytes, and owns the resulting
`Engine`. It exposes only the borrowed converter. Member and scope order destroy
the Engine before restoring and removing the profile, restore the preceding
clock after the Engine is gone, and keep the data buffer alive longest. Only the
freezer command binaries depend on this heavy target; corpus freezer libraries
and their mock-converter tests depend on the lightweight target.

The conversion session resets history when preceding text is empty and otherwise
calls `ReconstructHistory` exactly once, preserving a false result without reset
or retry. It creates a fresh composer and table, enables only deferred final
candidate limits, runs `StartConversion`, requires a conversion segment and
candidate zero in every conversion segment, and never commits or finishes. It
concatenates candidate zero for the baseline, builds one conversion-mode,
focused-segment-zero request with token `(0, 0, request_sequence)`, and copies
every token, context, reading, segment, candidate, cost, attribute, consumed-key,
and protection field in existing order without sorting. The existing
`SerializeDeterministically`, `ReviewTextproto`, and `Sha256Bytes` implementations
remain the sole artifact serialization primitives used by both adapters.

AJIMEE retains its public freezer API as a thin adapter. It alone validates
AJIMEE source and case semantics, converts Katakana to Hiragana, records the
history result, and selects `PREEDIT_TEXT`, which reproduces its current request
reading. Its default request remains the empty desktop request and its config
remains the exact default config. The accepted 1,335,914-byte frozen binary must
remain SHA256
`9b1656a48ed3cfbf1577b7a404688affb49b793f9c556ba2d2fb46b41238c61c`,
and its 6,900,264-byte review textproto must remain SHA256
`6b148ac3658844a885edb8491a31e020c451ad3a3b6bec9f08ce70e65de73930`.
The quality-regression adapter uses the same empty request, copies the default
config and explicitly sets `use_typing_correction=true`, requires the ranking
section to remain absent, supplies empty external context, selects
`CONVERSION_QUERY`, and stores that same query in both the frozen case and frozen
request. It reads inputs only, constructs no ranking backend, and two fresh
processes must produce byte-identical development and holdout freezes.

That gate passes. The deterministic empty desktop Request remains SHA256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
and the desktop Config with typing correction explicitly enabled is SHA256
`747675393bfacbb3542de2a1b80b9273be46c7987b01956811b5783298958f31`.
The 1,779-byte development freezer config is SHA256
`b9818bc4fb205b76c2e7555e2a37556cf35084e367b08d2e0b7a88914c463ce3`;
the 1,724-byte holdout freezer config is SHA256
`570fd92630dfb07d54c6ed5af456ea7bd2bbebc4cb331a74490c3fda2746fd12`.
Two fresh input-only processes produced the same 219-case development frozen
binary, 328,883 bytes with SHA256
`44c9a08d748f2cef5c8428b1101f490c2719db34e533ba1cda3cee9b12b41c59`,
and the same 1,569,302-byte review text with SHA256
`de22b355a72a2ffd63232e0b2447ca4b60632917835d68dca7fe5e4237318866`.
Two fresh input-only processes likewise produced the same 72-case holdout
frozen binary, 166,803 bytes with SHA256
`aec5be85738ada4fad68943fff3a0e556cdf61b587f93cfa9ca380164e61e812`,
and the same 831,819-byte review text with SHA256
`6097618e7bbab44757a61bcc7079ab1dd133392ce6889676058556a7e546be04`.
The command gate uses only the checked input review artifacts and rejects an
`--answer_corpus` argument before Mozc initialization without writing output.

Focused lightweight tests pin exact candidate field and repeated-field order,
both reading sources, the reset and reconstruction branches, a preserved false
reconstruction result, conversion options and context, candidate-zero baseline,
invalid mode rejection, missing segment and candidate rejection, no commit or
finish calls, the nested clock and preceding-clock restoration, rejection of a
present ranking section, and the exact default request and config hashes. Runtime
tests pin data-hash rejection before Engine creation and profile and clock
restoration. AJIMEE adapter tests keep their existing expectations and add one
fixed synthetic binary hash. The retained-artifact check pins the accepted
AJIMEE binary and review hashes above after the proto move and again after the
freezer extraction. Quality-regression command tests assert that neither binary
accepts an answer path and that two fresh processes emit identical complete
artifacts for each corpus role.

`quality_regression_objective_evaluation.proto` schema version 1 defines the
development execution and artifact boundary. `DevelopmentObjectiveSpec` stores
objective, manifest filename, raw manifest SHA256, and selection eligibility on
tags 1 through 4. `DevelopmentExecutionConfig` stores schema, frozen schema,
coverage schema, frozen SHA256, case count, manifest schema, coverage-definition
version, permutation-definition version, and exactly three ordered objective
specifications on tags 1 through 9. The structured objective is reference-only.
Every persisted objective field uses the imported
`CandidateRankerModelManifest.ScoringTemplate.RecordLayout` enum with its
existing values `RECORD_LAYOUT_UNSPECIFIED=0`, `STRUCTURED_WITH_TARGET=1`,
`STRUCTURED_WITHOUT_TARGET=2`, and `NATURAL_TEXT_ONLY=3`.
`DevelopmentNativeCoverageDisposition` has `UNSPECIFIED=0`, `SELECTED=1`, and
`OMITTED_CAPACITY=2`. `DevelopmentNumericProfile` has `UNSPECIFIED=0` and
`CONFIGURED=1`. `DevelopmentEvaluationLayout` has `UNSPECIFIED=0` and
`SHARED_TRIE=1`. Every unspecified or unknown enum value fails validation.
The fixed permutation reverses only the natively selected candidate objects
among their existing unprotected slots. Protected candidates and unselected
unprotected candidates remain in their original slots, so the native window and
every omitted candidate are unchanged.

`NativeCoverageSegment` stores uint64 segment ID, typed disposition, and ordered
uint64 selected candidate IDs on tags 1 through 3. `NativeCoverageCase` stores source
line and ordered segments on tags 1 and 2; `NativeCoverageMap` stores ordered
cases on tag 1. `ObjectiveNativeCoverage` stores objective and map on tags 1 and
2. `DevelopmentNativeCoverageSuite` stores schema, raw execution-config hash,
frozen hash, coverage-definition version, and exactly three ordered objective
coverages on tags 1 through 5. Run the vocabulary-only capacity
auditor natively for each complete request and each manifest. For every
rankable segment, reconstruct the selected candidate IDs from its typed
disposition and selected unprotected-prefix count. The complete ordered maps of
`(source line, segment ID, disposition, candidate IDs)` must be byte-identical
across all three objectives. Any difference fails the comparison. Never take an
intersection, shorten a prefix, alter a request, or substitute an evaluator-only
candidate plan to manufacture equal coverage.
The accepted coverage artifact is always the original-request ordered map. For
the fixed permutation, the ordered rankable-segment count, disposition,
selected-ID count, and selected-ID membership must match that accepted map,
while the permuted selected-ID order must equal the prescribed reversal. The
unprotected-candidate count is recomputed from the frozen original request and
must remain unchanged in the permuted request. Transient window and omission
counts are validated against each production capacity result while native
coverage is constructed, but they are not persisted and no later artifact
claims to attest them. Permutation validation therefore does not require the
ordered maps themselves to be byte-identical. The complete permuted-capacity
validation pass for every objective and case must finish before the first
scoring operation.

This gate passes. The 1,001-byte checked development execution config is
SHA256 `c0bca67765f3ad26bc16495d4dde60daa07cd1d8cd37d3c12b46999e59f24a9f`.
Two fresh vocabulary-only processes produced byte-identical native-coverage
artifacts. The binary is 40,099 bytes with SHA256
`706bbe8c4c15e1cfb5fbec781afe28a3661ad3659c60d3dcabd03ba3912aad29`;
the review text is 660,366 bytes with SHA256
`8d03a674290ed9f5e5e3e6d6e8dc77de811de8ef4d68e3894bb9530db6823815`.
Each of the three byte-identical maps contains 219 cases, 353 selected
rankable segments, 5,012 selected candidate IDs, and zero segments omitted by
capacity. No tensor was loaded and no decode was performed.

The answer-free development runner passes each original frozen request to its
native F16 backend and emits both semantic orders and exact configured
shared-trie scores. After the native coverage suite is accepted, a separate
checked `DevelopmentScoreRunnerConfig` binds its own schema, score-artifact
schema, raw execution-config hash, frozen hash, accepted coverage-suite hash,
score-definition version, configured numeric profile, and shared-trie layout on
tags 1 through 8. It prevents inference from starting against an unaccepted or
replaced coverage map. The 456-byte checked score-runner config is SHA256
`d6406c042f93a9c451e2ae60d6fe99625fa00d8f70909949fafcdb7212c63192`.
The score command hashes, parses, recursively validates, and joins the raw score
config, execution config, frozen corpus, and accepted coverage suite before it
creates any ranker or opens a model. Because it is a fresh process, it also
rereads and retains all three raw manifests and computes all three raw hashes
before parsing any manifest. It then parses and recursively validates all three,
requires their configured hashes and record layouts, and requires them to be
identical outside record layout before any ranker creation or model open.
Before `InitMozc`, its raw-argument allowlist requires exactly one nonempty
`--name=value` occurrence of each of `--score_config`, `--execution_config`,
`--frozen_corpus`, `--native_coverage`, `--manifest_directory`,
`--model_directory`, `--output_binary`, and `--output_textproto`, and rejects
every other, duplicate, empty, or split-form flag.

`DevelopmentObjectiveScores` schema version 1 binds the raw score config,
execution config, frozen corpus, accepted coverage suite, score definition,
configured numeric profile, and shared-trie layout in its identity. Ordered
objectives contain ordered source lines and segments; candidate messages are
stored in the exact production semantic rank order, so there is no separate
semantic artifact. A
candidate stores uint64 ID on tag 1, uint64 zero-based original Mozc index on
tag 2, `int32` Mozc cost on tag 3, and the `std::bit_cast<uint64_t>` of its finite double score
in a fixed64 tag 4. The original Mozc index is the candidate's zero-based index
in the full frozen original segment, including protected slots, and is never an
index in the fixed-permutation request. Reject NaN and infinities and
canonicalize either signed zero to positive zero before score-bit comparison or
storage. `DevelopmentSegmentScores` stores segment ID and ranked
candidates on tags 1 and 2; `DevelopmentCaseScores` stores source line and
ordered selected segments on tags 1 and 2; and
`DevelopmentObjectiveScoreSet` stores objective and ordered cases on tags 1 and
2. `DevelopmentObjectiveScoresIdentity` stores raw score-config,
execution-config, frozen-corpus, and coverage-suite hashes on tags 1 through 4,
then score definition, numeric profile, and layout on tags 5 through 7. The top
level artifact stores schema, identity, and exactly three ordered objectives on
tags 1 through 3. The deterministic binary is authoritative; review text prints
the fixed64 integer without decimal floating-point conversion. The artifact
contains no answers or answer hashes, candidate text, keys, readings, context,
records, tokens, edges, host data, or time and is never packaged.

The fixed-permutation gate compares the complete candidate-ID-to-canonical-score
map exactly. It then derives and validates each response independently with the
production stable ordering and validates each complete merge against its own
request. Exact score ties intentionally retain that request's current Mozc
order, so reversing selected candidate objects may reverse IDs inside a tied
score group. The ordered canonical score-bit sequence must still be identical.
Candidate-ID order and the complete merge must match across the two requests
when all selected scores in that segment are distinct, and the winner ID must
match when the maximum score is unique. A different ID order or winner inside
an exact tied group is not a scoring failure and is not persisted from the
permuted diagnostic run.

The score artifact permits later development-only calibration of a Mozc cost or
order prior and replacement threshold without rerunning the unchanged F16
model. Any record-layout, score-normalization, terminal, or model change still
requires new scores. Offline calibration is the first component allowed to read
development answers and emits an immutable selection artifact binding the
chosen manifest, score artifact, calibration config, exact policy parameters,
and development report. The holdout execution config names exactly one selected
objective and binds that selection-artifact hash. Its runner accepts no answer
or answer-hash input and emits no calibration score artifact; only the final
holdout reporter opens holdout answers.

Natural-text-only scoring plus a manifest-owned Mozc cost or order prior and a
conservative manifest-owned replacement threshold is the candidate production
direction, not yet the accepted design. The initial three-layout development
run changes record layout only under the checked F16 model and does not select a
Mozc prior, score normalization, terminal policy, replacement threshold,
multiplicity policy, replacement model, or quantization. If neither corrected
layout passes, stop without selecting an objective. Before reusing the retained
development scores, amend the narrative with a finite calibration grid, exact
feature definitions, acceptance gate, and deterministic tie-break for the Mozc
prior and threshold. A normalization, terminal, layout, or model change requires
new model scores. Replacement-model comparison requires its own predeclared
development cells. Q4_K_M remains exclusively in the later fixed F16-versus-Q4
gate. The archived schema-version-4 structured continuation objective is
rejected evidence only and is not executable by current code or eligible for
production wiring.

The interface, deterministic merger, candidate-limit extraction, persisted
configuration, bounded worker service, live invalidation, ordered shutdown,
privacy gates, and client diagnostics are complete. The pinned runtime builds
and links, the rejected label baseline is covered by focused tests, and its
immutable source weights, converted BF16 GGUF, and Q4_K_M GGUF are present in
the ignored cache with exact hash and metadata verification. Stage 9 completed
the deterministic semantic and answer-aware quality evaluation for the first
continuation-scoring Q4_K_M candidate. That candidate failed the quality gate
and is not eligible for production Engine wiring or selection as a packaged
default. Stage 9 remains open for selecting and evaluating a corrected scoring
objective and, if needed, a replacement model.

On the local Intel i9-11900H, both corrected artifacts selected the same correct
semantic winner in all three smoke cases. Their exploratory timing runs are not
performance evidence because concurrent host CPU load changed between runs.
Q4_K_M warm times ranged from about 95 to 748 ms while the host was also running
build, quantization, communication, synchronization, and VPN processes. A CPU
sample during the investigation measured 79--90% total use while the ranker was
idle. The measurements are discarded rather than choosing the favorable sample.
The required benchmark runs on an otherwise idle host, records its environment
and exact request artifacts, and reports distributions for 5, 10, and 26
candidates before any interactive-latency claim or model promotion.

The latency gate consumes deterministic binary requests frozen after Mozc's
full rewriter chain; command-line candidate files remain smoke-only. A checked
derivation produces 5, 10, and 26 rankable-candidate requests from authentic
snapshots while retaining complete candidate metadata, immutable IDs, order,
context, and protected candidates. It records both source and derived request
hashes and never changes backend selection behavior. A versioned benchmark
configuration owns the three candidate counts, paired decode and batch thread
counts 1, 2, 4, 8, and 16, ten warmups, 100 measured calls per cell, the fixed
schedule, a 5% background CPU ceiling, and power requirements.

Before model loading and again after warmup, the runner requires ten consecutive
one-second system CPU windows at or below the configured ceiling. Every timed
call is bracketed by Win32 system and process CPU times, QPC ticks, power state,
affinity, and priority. Background CPU is system busy time minus benchmark
process CPU time. Any counter failure or regression, background use above the
ceiling, power transition, affinity or priority change, ranking-order change,
or backend error invalidates the entire run. It never retries, drops, or
replaces a contaminated sample. An invalid artifact retains the raw evidence
and contains no percentiles. A valid artifact stores integer QPC ticks for all
100 calls and reports nearest-rank p50, p95, and p99. Model loading and warmup
are excluded. Provenance includes source and effective manifest hashes, GGUF
and verifier hashes, frozen corpus, configuration and request hashes, benchmark
executable hash, source revisions, numeric profile, host identity, power state,
affinity, priority, and QPC frequency.

## Pseudocode

```text
function build_candidates(conversion_request):
    config = session.config.candidate_ranking
    private_request = (
        conversion_request.incognito_mode
        or composer.GetInputFieldType() is PASSWORD)

    if not config.enabled:
        result = mozc_converter.convert_and_rewrite(conversion_request)
        do not retain ranker context or call ranking_service
        return result

    if private_request:
        session.record_ranking_diagnostic(SKIPPED, PRIVACY)
        result = mozc_converter.convert_and_rewrite(conversion_request)
        return result

    if not config.has_positive_max_wait_millisec:
        session.record_ranking_diagnostic(FAILED, INVALID_CONFIG)
        result = mozc_converter.convert_and_rewrite(conversion_request)
        return result

    if not ranking_service.has_backend:
        session.record_ranking_diagnostic(UNAVAILABLE, BACKEND_MISSING)
        result = mozc_converter.convert_and_rewrite(conversion_request)
        return result

    conversion_request.defer_candidate_limits = true
    result = mozc_converter.convert_and_rewrite(conversion_request)

    ranking_request = snapshot(
        token = session.begin_ranking_generation(),
        complete_reading = composer.GetQueryForConversion(),
        preceding_text = session.current_context.preceding_text_or_empty,
        following_text = session.current_context.following_text_or_empty,
        segments = result.segments)

    require ranking_request.token is not older than
        ranking_service.latest_live_token_for_session
    ranking_response = ranking_service.rank_until(
        ranking_request,
        config.max_wait_millisec)

    if ranking_request.token does not equal session.current_live_token:
        return without mutating or trimming the current session state

    if ranking_response is an error:
        session.record_stable_ranking_diagnostic(ranking_response.status)
        return apply_candidate_limits(result)

    require ranking_response.token equals ranking_request.token
    require ranking_request.token equals session.current_live_token
    require result still matches ranking_request snapshot
    require every referenced candidate and segment exists
    require protected candidates are not referenced

    validate every segment ranking before mutation
    keep protected candidates in their original slots
    fill other slots from the returned order
    append omitted original candidates in original order

    result = merge_deterministically(result, ranking_response)
    session.record_ranking_diagnostic(APPLIED)
    return apply_candidate_limits(result)

function attach_ranking_diagnostic(output):
    clear output.candidate_ranking_diagnostic
    if session has a current ranking diagnostic:
        output.candidate_ranking_diagnostic = session.current_diagnostic

function update_field_privacy(context, persistent_field_type):
    if context explicitly names a field type:
        session.password_field = (
            context.field_type is PASSWORD
            or persistent_field_type is PASSWORD)
    else if persistent_field_type is PASSWORD:
        session.password_field = true
    # An omitted field type never clears a previously observed password field.

function select_candidate_from_suggestion(candidate_id):
    invalidate_ranking_work()
    predict_internal(rank_candidates = false)
    move_focus_to(candidate_id)

function clone_converter_for_undo():
    clone conversion state and password privacy classification
    assign a fresh session generation
    reset state revision and request sequence
    clear surrounding text, diagnostics, and pending model work

function rank_with_local_model(request, manifest, cancellation):
    response.token = request.token
    require loaded model is decoder-only and causal
    require model context capacities equal manifest per-decode limits
    scan request without copying it and require segment count, candidate count,
        and total string bytes fit the manifest hard request limits
    context = build immutable full-request record context
    selected_segments = empty

    for each request segment in segment-ID order:
        window = first manifest maximum unprotected candidates in Mozc order
        record raw request counts and candidates omitted by the window
        if window has fewer than two candidates:
            continue
        feasible = empty strict window prefix
        for each candidate in window order:
            require cancellation is not requested
            prepare one full-context record and tokenize it once
            if the record byte or token capacity is exceeded:
                record the first limiter and stop this segment
            trial = feasible plus this candidate
            if trial has at least two candidates:
                trial_plan = build exact one-segment trie
                trial_usage = per-decode usage with reserved rows equal to
                    max(trial output rows, manifest sequence capacity)
                if trial_usage exceeds a per-decode capacity:
                    record the first limiter and stop this segment
            feasible = trial
        if feasible has fewer than two candidates:
            record the segment as omitted by capacity
            continue
        retain feasible and its exact one-segment plan

    batches = empty
    current_segments = empty
    for each retained segment in segment-ID order:
        require cancellation is not requested
        trial = exact combined plan for current_segments plus this segment
        if current_segments is nonempty and trial exceeds a per-decode limit:
            append exact plan for current_segments to batches
            current_segments = this unchanged whole segment
            require its already checked one-segment plan fits
        else:
            append segment to current_segments
    if current_segments is nonempty:
        append its exact plan to batches

    if batches are empty:
        return response

    orders = request-local empty aggregate
    for each batch plan in order:
        require cancellation is not requested
        enter cleanup scope that clears abort callback and model memory on exit
        clear model memory before the decode
        create a unified-KV batch with dense local sequence IDs
        emit every trie node once, parent before child
        mark only predecessors with scored outgoing edges for logits
        decode with cancellation connected to the abort callback
        if decode reports cancellation:
            discard orders and return cancellation through the cleanup scope
        if decode reports any other failure:
            discard orders and return backend error through the cleanup scope
        for each marked original batch-token index:
            require cancellation is not requested
            compute finite full-vocabulary double log-sum-exp
            accumulate normalized edge log probabilities by local sequence ID
        stable sort each complete segment by descending finite score and its
            original Mozc order, then append its ID order to local orders
        leave the cleanup scope before the next batch

    require exactly one valid local order for every selected segment
    require cancellation is not requested
    append local orders to response in segment-ID order only after all batches
        succeed
    return response

function windows_manual_evaluation_main(wide_argv):
    utf8_argv = convert every complete UTF-16 argument to UTF-8
    initialize Mozc and parse flags only from utf8_argv
    run the manual smoke or score diagnostic

function verify_shared_trie_scores(request, checked_model):
    print the loaded manifest SHA-256, GGUF identity, source revision,
        runtime revision, and each numeric profile used by the report
    shared = rank_with_local_model_and_return_edge_scores(
        request, checked_model, shared_trie, configured_numeric_profile)
    require shared uses the exact production windows, reductions, and batches
    require production Rank returns the same aggregate selected-segment orders
    for each selected segment:
        complete_sequences = build the same selected candidate records
        common_prefix_length = token longest common prefix of complete_sequences
        for each candidate in original order:
            clear model memory
            decode its complete sequence independently with one sequence ID and
                the same configured numeric profile
            independent_score = sum normalized target log probabilities from
                common_prefix_length through the terminal token
            prefix_tokens = tokenize the same normalized record ending
                immediately after the candidate
            classify scored targets before or crossing that token boundary as
                candidate-or-boundary, later nonterminal targets as following
                context, and the terminal target separately
            record every target position, target token, normalized log
                probability, and contribution class without changing total
            require shared and independent edge identities are equal
            require every shared edge score matches its independent edge score
                within the precision tolerance selected for the checked artifact
    repeat both layouts with F32 K/V and flash attention disabled to distinguish
        runtime graph behavior from the configured F16 K/V and automatic flash
    permute every candidate position and require mapped shared scores are equal
    require packing a whole segment alone or with other segments preserves its
        mapped records, common prefix, edges, scores, and order
    return a deterministic diagnostic report

function locate_transformers_gguf_score_mismatch(request, checked_f32_model):
    records = build the same normalized complete candidate records
    for each candidate:
        require Transformers and GGUF input token IDs are identical
        require the segment common-prefix length and every scored target position
            and target token are identical
        decode every scored predecessor row with the strict F32 numeric profile
        compare complete full-vocabulary rows and stop at the first mismatch
    only after all rows agree may candidate, following-context, and terminal
        contributions be interpreted as model-quality evidence

function verify_f32_model_import(checked_records):
    hf_model = load immutable Transformers model in F32
    gguf_model = load checked F32 GGUF on the portable CPU runtime with
        F32 KV storage and flash attention disabled
    for each record in checked_records:
        tokens = [BOS] + tokenize record without automatic special tokens
        require Hugging Face and GGUF token IDs are identical
        positions = unique(first, middle, final input positions)
        hf_rows = full-vocabulary F32 logits at positions
        gguf_rows = full-vocabulary F32 logits at positions
        require every logit is finite
        require allclose(hf_rows, gguf_rows, rtol = 1e-4, atol = 1e-4)
        require each row has the same top token

function verify_q4_artifact(checked_f16, retained_q4, pinned_quantizer):
    verify checked_f16 identity and pinned quantizer source and patch hashes
    hash the verifier entry point and every imported project verification
        library that controls the accepted artifact or report
    require the quantizer link uses MSVC /Brepro and two independent Bazel
        output bases produce byte-identical executables
    verify the exact quantizer executable SHA256 before executing it
    on Windows, convert complete UTF-16 quantizer arguments to UTF-8 first
    regenerate Q4_K_M from checked_f16 in a private temporary directory
    require regenerated bytes are exactly retained_q4 bytes
    require exact shared metadata, tokenizer, tensor names, logical shapes,
        parameter count, declared Q4_K and Q6_K packing, and F32 values
    reject tensor-type overrides for nonquantized artifacts and require each
        quantized-artifact override to affect one expected tensor type
    record every packed tensor storage hash and the exact byte comparison
    deterministically reproduce the retained verification report
    run model-free tests for schema, branching, overrides, packing, and reports
    expose one manual local target that checks all three real artifact reports

function import_ajimee(source):
    verify external corpus revision and hash
    require the schema-version-one field roles and complete checked attribution
    for each case:
        input_corpus += source ID, complete Katakana input, allowed context,
            context slice, and whether source split data exists
        answer_corpus += source ID and accepted whole outputs
        never serialize original_text or split chunk strings
    copy source, creator, license, upstream, and modification notice metadata
        into both corpora
    write input and answer corpora separately

function create_evaluation_freezer_runtime(
        data_path, expected_data_sha256, data_type, clock_text):
    parse clock_text as the exact canonical UTC RFC3339 representation
    read data_path once and retain the complete byte buffer
    require SHA256(retained bytes) equals expected_data_sha256
    install an outer scoped clock before constructing Engine
    create a unique temporary profile, save the preceding profile, and select it
    construct DataManager from the retained bytes and the declared data type
    construct Engine and expose only its borrowed ConverterInterface
    own resources so destruction order is Engine, profile restoration and
        removal, clock restoration, then retained data bytes
    never construct or expose a candidate-ranking backend

function create_evaluation_freezer_session(
        clock_text, desktop_request, desktop_config, borrowed_converter):
    parse and require the same canonical UTC clock
    require desktop_config has no candidate-ranking section
    compute request_sha256 and config_sha256 only by deterministic protobuf
        serialization followed by SHA256
    install a nested scoped clock for the complete corpus conversion loop
    return a noncopyable session owning request and config copies and borrowing
        only the converter

function freeze_evaluation_conversion_case(session, case_input):
    require positive case_input.request_sequence and a supported reading source
    if case_input.preceding_text is empty:
        ResetConversion once and set history_reconstructed to false
    else:
        history_reconstructed = ReconstructHistory once
        do not reset or retry when reconstruction returns false
    create a fresh Composer and Table from the session request and config
    set its preedit to case_input.preedit_text
    set exact preceding and following Context fields
    build a conversion request with only defer_candidate_limits set true
    run StartConversion once without Commit or FinishConversion
    require at least one conversion segment and candidate zero in every segment
    conversion_query = conversion_request.key
    if reading source is PREEDIT_TEXT:
        frozen_reading = case_input.preedit_text
    else if reading source is CONVERSION_QUERY:
        frozen_reading = conversion_query
    else:
        fail invalid reading source
    baseline = concatenate candidate zero from every conversion segment
    token = session_generation zero, state_revision zero,
        request_sequence case_input.request_sequence
    ranker_request = BuildCandidateRankerRequest with conversion mode,
        frozen_reading, exact external context, focused segment zero, and the
        rewritten and suppressed conversion segments
    copy every ranker request and candidate field in existing repeated order
    fail an unknown candidate-ranker mode instead of storing MODE_UNSPECIFIED
    return conversion_query, history_reconstructed, baseline, and frozen request

function freeze_external_evaluation_case(input_case, ordinal, session):
    normalized_reading = katakana_to_hiragana(input_case.complete_reading)
    result = freeze_evaluation_conversion_case(session, {
        request_sequence: ordinal + 1,
        preedit_text: normalized_reading,
        preceding_text: input_case.published_preceding_context,
        following_text: empty,
        reading_source: PREEDIT_TEXT,
    })
    store source index, normalized reading, result.history_reconstructed,
        result.baseline, and result.frozen_request in the AJIMEE case
    retain the existing AJIMEE source, Mozc, input, and case validators
    never open the answer corpus or source dataset

function audit_frozen_corpus(corpus, manifest):
    verify corpus schema, source hashes, Mozc revision, and manifest
    reject any malformed case before constructing an audit artifact
    open the checked Q4 GGUF through the protected hash-verified path
    typed_metadata = parse the same verified file handle with the pinned GGUF
        reader in no-allocation mode
    architecture = require typed STRING metadata `general.architecture`
    model_context = require typed UINT32 metadata in 1 through INT32_MAX at
        key architecture plus `.context_length`
    free typed_metadata and rewind the same verified handle
    load vocabulary metadata only, without tensors or a decode context
    require manifest per-record token limit minus the un-fed terminal token
        is at most model_context
    audit every complete request with the production preparation path
    record raw request counts, bounded coverage, batch count, and per-decode
        maxima for every request
    if any auditor call fails or an emitted batch violates a cap:
        write a typed audit-error result and do not run inference
    treat explicit window and capacity omissions as measured passing coverage
    validate and deterministically serialize the complete capacity audit
    destroy the vocabulary-only auditor

function run_frozen_semantics(
        corpus, capacity_config, capacity_audit, config, manifest):
    verify config schema and exact capacity-config and capacity-audit hashes
    validate the actual checked capacity_config without reconstructing it
    verify the exact frozen hash and case count owned by capacity_config
    recursively reject unknown frozen-corpus fields, then validate the complete
        frozen corpus before creating the backend or invoking any callback
    validate capacity_audit with capacity_config and require every case passed
    require capacity_config, capacity_audit, and manifest identities match
    expected_identity = capacity_config manifest hash, GGUF name and hash,
        tokenizer hash, quantization, model-source revision, and runtime revision
    backend = full Create with expected_identity checked before GGUF open
    require the backend exposes exactly expected_identity before any Rank call
    require the backend reads the same metadata model_context and
        llama_model_n_ctx_train equals it
    for each frozen case in stable case order:
        request = exhaustively convert the exact frozen request without sorting
        backend_result = call production Rank exactly once with no cancellation
        if backend_result is not OK:
            record BACKEND_ERROR and canonical code without its message or any
                response field
            continue
        response = backend_result value
        record whether response token equals request token, without the token
        record exact response segment-order and candidate-ID counts
        record whether both counts are within their frozen request counts
        if the response is within bounds:
            preserve response segment and candidate order exactly as returned
        if the response token does not equal the request token:
            record INVALID_RESPONSE and canonical ABORTED without output
            continue
        if the response is outside request bounds:
            record INVALID_RESPONSE and canonical INVALID_ARGUMENT without raw
                orders or output
            continue
        merge = BuildCandidateRankerMergeOrder(request, response, request.token)
        if merge is not OK:
            record INVALID_RESPONSE and exact canonical code without output
            continue
        merged_top = concatenate the request candidate value at position zero
            of every complete merge order in request-segment order
        record SUCCESS, canonical OK, exact raw orders, and merged_top
    validate response-field presence, exact counts, bounds, token precedence,
        pure merges, and recomputed tops
    reject unknown fields recursively and never sort malformed artifact input
    deterministically serialize semantic results without timing or host fields
    write binary and review text only after complete in-memory validation
    return failure after writing if any case is not SUCCESS
    accept only the 284-byte config with SHA256
        982b1c8a6c8748a6b30bd3f60be1850328454522c20916c307dfd6d091bea223
    require two fresh sequential processes produce byte-identical outputs
    accept only 200 AJIMEE_SEMANTIC_SUCCESS cases with zero backend errors and
        zero invalid responses
    pin the 78,338-byte binary SHA256
        7b1aa411809ca6e74bad9f5f96bcb9706a735833c98c94b4d4a9d39be9ac38c8
    pin the 642,545-byte review-text SHA256
        9012fcf4eed6afe2a9b035904ea424fe0104eca06b81008fe96556883259896e

function report_ajimee_quality(
        reporter_config, capacity_config, frozen, answers,
        capacity_audit, semantic_results, output_binary, output_json):
    invoke target report_ajimee_quality from report_ajimee_quality_main.py
    use checked config ajimee_quality_report_config.textproto
    require its accepted 1,675 raw bytes have SHA256
        c8675a28f645a2edccf843894261204bef5c6aa9843189d456394c2048128f6e
    accept exactly --config, --capacity_audit_config, --frozen_corpus,
        --answer_corpus, --capacity_audit, --semantic_results,
        --output_binary, and --output_json as required path flags
    read each input file once and hash the retained bytes before parsing
    require every raw hash and schema equals the checked reporter config
    reject recursive unknown fields and validate the complete expected source
    validate the actual capacity_config instead of reconstructing its identity
    validate frozen with the shared validator and require 200 ordered cases
    require answers have the same source identity and exact ordered source IDs
    require every accepted list and string is nonempty and valid UTF-8
    preserve answer lists but form a metric view by stable byte-exact deduplication
    validate capacity_audit with capacity_config and require every result passed
    validate semantic_results and require its frozen, capacity-config,
        capacity-audit, backend, and numeric-profile identities all match
    require its accepted 78,338 raw bytes have SHA256
        7b1aa411809ca6e74bad9f5f96bcb9706a735833c98c94b4d4a9d39be9ac38c8
    for each joined case in ordinal order:
        context_slice = frozen request preceding text is nonempty
        baseline_correct = frozen baseline exactly equals an accepted output
        oracle_covered = false
        for each distinct accepted output:
            reachable_byte_offsets = set containing zero
            for each frozen request segment in order:
                advance an offset by every candidate value that is an exact
                    prefix of the accepted output at that offset
                replace reachable offsets with the bounded advanced set
            oracle_covered |= complete accepted-output length is reachable
        if semantic outcome is SUCCESS:
            defensively replay the C++ pure merger contract from the exact raw
                response orders, using fixtures cross-checked with C++ outputs
            recompute merged_top from position zero of every complete segment
            require recomputed merged_top equals stored merged_top_output
            effective_model_output = recomputed merged_top
        else:
            require the typed backend or invalid-response invariants
            effective_model_output = frozen baseline
        model_correct = effective_model_output exactly equals an accepted output
        require baseline_correct and model_correct imply oracle_covered
        classify retained correct, win, regression, or unchanged incorrect
        baseline_reference = first accepted ordinal attaining minimum unit-cost
            Levenshtein distance over Unicode scalar values from baseline
        model_reference = first accepted ordinal attaining the same minimum
            from effective_model_output
        accumulate integer distances and selected-reference scalar lengths
        add the exact capacity usage and diagnostics to overall and context slice
    require exactly 100 context and 100 no-context joined cases
    componentwise sum and maximize all 22 capacity-usage fields per slice
    build ascending decode-batch histograms and enum-ordered limiter summaries
    for overall, context, and no-context slices:
        derive exact accuracy, gain, oracle, conditional, and retention ratios
        require all count algebra and correctness-at-most-oracle invariants
        W, R, T = win, regression, and remaining tie counts
        mass[0] = one arbitrary-precision integer
        repeat the slice case count times:
            next[s - 1] += mass[s] multiplied by R
            next[s] += mass[s] multiplied by T
            next[s + 1] += mass[s] multiplied by W
            mass = next
        total_mass = case count to the power case count
        lower = first sum with 40 times cumulative mass at least total_mass
        upper = first sum with 40 times cumulative mass at least 39 times total_mass
        store lower and upper as signed gain-case numerators over case count
    for the checked real frozen and answer hashes:
        require Mozc correct and oracle counts are 103 and 180 overall,
            46 and 88 with context, and 57 and 92 without context
        require Mozc CER is 251/4246, 158/2039, and 93/2207 respectively
    build the report identity only from checked hashes and validated identities
    emit exactly three ordered metric and capacity-coverage slices
    validate the complete report and serialize deterministic binary in memory
    serialize canonical UTF-8 JSON with tag-order keys, decimal-string integers,
        LF line endings, one final newline, and no floating-point numbers
    require the report schema copies no accepted, baseline, or model-output field
    write both outputs only after complete in-memory validation
    never open a model, manifest, source JSON, input corpus, or latency artifact
    require two fresh reporter processes produce byte-identical outputs
    pin the 2,513-byte binary SHA256
        c3cd5b9819873647f00f7d2a8b4e110358ac0b86b83d06b19d650f174f6987e4
    pin the 12,974-byte JSON SHA256
        0460c0d5e4f684c5f0ec57f66e3ec8e3da42532c4c71895dc4cf6dcc061df4c5
    require 200 semantic successes, 200 capacity passes, 174 window-omission
        cases, three candidate-capacity-omission cases, and no
        segment-capacity-omission case
    require overall Mozc and model correctness are 103 and 17 of 200,
        retained correctness 16, one win, 87 regressions, Mozc CER 251/4246,
        model CER 1837/4254, oracle coverage 180/200, and exact paired-bootstrap
        gain interval -100 through -72 over 200 cases
    require context Mozc and model correctness are 46 and 14 of 100
    require no-context Mozc and model correctness are 57 and three of 100
    mark the Q4_K_M quality promotion gate failed decisively
    block this Q4_K_M artifact from promotion and packaging as the default model

function build_candidate_record_v5(context, target, include_following, manifest):
    require manifest schema version 5 and an explicit supported record layout
    natural_body = context.preceding_text
    append each frozen baseline value before target in segment-ID order
    append target candidate value
    if include_following:
        append each frozen baseline value after target in segment-ID order
        append context.following_text

    switch manifest.scoring_template.record_layout:
        STRUCTURED_WITH_TARGET:
            record = mode field + separator + full-reading field + separator
                + target-key field + separator + text prefix + natural_body
        STRUCTURED_WITHOUT_TARGET:
            record = mode field + separator + full-reading field + separator
                + text prefix + natural_body
        NATURAL_TEXT_ONLY:
            record = natural_body
        otherwise:
            fail invalid manifest
    overflow-check and enforce the raw byte limit
    if the raw record fits, validate UTF-8 only for serialized fields
    normalize the complete record once and enforce normalized byte limit
    return the normalized record

function prepare_qwen_development_model():
    read the prepare config
    read artifact identities from all three checked manifests
    require the three manifests to agree on source model, revision, weight hash,
        tokenizer hash, GGUF name and hash, F16 quantization, runtime name,
        converter revision, and converter archive hash
    require the config model id, revision, GGUF name, and F16 outtype to equal
        those manifest values
    require every required hash field to be present
    require the model id to contain neither Instruct nor sarashina
    destination = cache model directory / GGUF name
    if destination exists:
        require its SHA256 equals the manifest GGUF hash
        return
    download the pinned llama.cpp archive when it is absent
    require the archive SHA256 equals the manifest converter archive hash
    extract the archive
    download each configured source file for only the pinned revision
    require the source weight SHA256 and tokenizer SHA256
    install the pinned converter Python requirements into the cache venv
    convert the source directory with convert_hf_to_gguf.py to F16
    require the written GGUF SHA256 equals the manifest GGUF hash
    write the GGUF only to the cache destination

function require_named_command(name):
    if the command is absent:
        fail with that command name

function llama_cpp_overlay_host_platform():
    require cpu x86_64
    if os is windows:
        use MSVC /bigobj and Advapi32
        define _CRT_SECURE_NO_WARNINGS
    else if os is linux:
        define _GNU_SOURCE for every llama.cpp translation unit
        link dl, pthread, and m
    else:
        fail incompatible

function install_pinned_hpc4_bazelisk():
    read url, sha256, and cache path from the HPC4 config
    if the cache path exists:
        require its SHA256
    else:
        download the pinned Linux amd64 binary to a partial path
        require the partial SHA256
        mark the partial file executable
        move it to the cache path
        require the cache-path SHA256
    return the cache path

function run_hpc4_qwen_setup():
    require_named_command python
    require_named_command curl
    require_named_command sha256sum
    validate the HPC4 config
    load the pinned compiler module
    require_named_command python
    require_named_command gcc
    require_named_command g++
    call llama_cpp_overlay_host_platform()
    bazelisk = install_pinned_hpc4_bazelisk()
    call prepare_qwen_development_model()
    bazelisk build the opt score target with the setup CPU count
    require the score executable

function prepare_quality_regression_objective_suite():
    read and hash checked oss.tsv and regression.tsv bytes before parsing
    parse with QualityRegressionUtil TestItem semantics on Windows
    require exact parsed, command, rank-zero, duplicate, and final counts
    preserve parsed keys; require every production-parser expected value equals
        the raw expected field under the explicit pinned kAll rule
    retain the production-parser expected value after that equality check
    stable-deduplicate by parsed key and normalized expected value
    reject development and holdout key or answer overlap
    emit separate deterministic development and holdout input and answer corpora

    for each input corpus without opening its answer corpus:
        read and hash the retained freezer-config and input buffers once
        require config schema, corpus identity, input hash, and case count match
        runtime = create_evaluation_freezer_runtime with the checked data path,
            data hash, data type, and clock
        desktop_request = the empty default desktop Request
        desktop_config = an exact copy of DefaultConfig
        set desktop_config.use_typing_correction to true
        require desktop_config has no candidate-ranking section
        session = create_evaluation_freezer_session with the same clock,
            desktop request, desktop config, and runtime converter
        require the session request and config hashes equal the checked Mozc
            identity before freezing the first case
        for each ordered input case at zero-based ordinal:
            result = freeze_evaluation_conversion_case(session, {
                request_sequence: ordinal + 1,
                preedit_text: input_case.reading,
                preceding_text: empty,
                following_text: empty,
                reading_source: CONVERSION_QUERY,
            })
            require result.history_reconstructed is false
            store input source line, result.conversion_query, result.baseline,
                and result.frozen_request
        store the retained raw freezer-config hash and actual runtime and session
            Mozc identities
        validate the complete frozen corpus in memory
        deterministically serialize binary and review text only after validation
        freeze again in a separate fresh process
        require byte-identical frozen ranker requests, baselines, complete binary,
            and review text for development and holdout independently
    before any development model output is opened:
        read and hash the raw checked baseline-report config before parsing
        require config schema, frozen schema, answer schema, report schema,
            complete DEVELOPMENT identity, frozen hash, answer hash,
            expected case count 219, and expected correct count 164
        read each accepted development frozen and answer buffer exactly once
        hash each raw buffer before parsing and require the configured hashes
        recursively reject unknown fields and structurally validate both corpora
        require identical complete identities, DEVELOPMENT role, 219 cases each,
            and identical source lines at every ordinal
        compare Mozc baseline and normalized answer by exact UTF-8 bytes only
        require exactly 164 matches and construct exact ratio 164 over 219
        construct and recursively validate one aggregate-only report containing
            only raw hashes, public identity, counts, and the exact ratio
        deterministically serialize binary and review text fully in memory
            before either write
        run the reporter in two fresh processes and require byte-identical
            binary and review artifacts; pin their hashes and sizes

    load the three checked schema-version-5 F16 manifests
    require their raw hashes and all fields except record layout are identical
    run the vocabulary-only capacity auditor for every development request
    reconstruct each objective's native selected IDs from the frozen request
        and its typed per-segment selected-prefix count, validating transient
        unprotected, window, and omission counts against the production result
        while persisting only segment ID, disposition, and selected IDs
    require all three complete native coverage maps are byte-identical
    never intersect, truncate, or replace native coverage
    build and fully validate the complete native coverage suite
    deterministically serialize its binary and review text fully in memory,
        then write both outputs
    run the coverage command again in a fresh process, applying the same
        build-validation-serialization-before-write order
    require byte-identical binary and review outputs and pin their exact hashes
    require the accepted execution config is 1,001 bytes with SHA256
        c0bca67765f3ad26bc16495d4dde60daa07cd1d8cd37d3c12b46999e59f24a9f
    require the accepted coverage binary is 40,099 bytes with SHA256
        706bbe8c4c15e1cfb5fbec781afe28a3661ad3659c60d3dcabd03ba3912aad29
    require the accepted review text is 660,366 bytes with SHA256
        8d03a674290ed9f5e5e3e6d6e8dc77de811de8ef4d68e3894bb9530db6823815
    require each of the three byte-identical maps contains 219 cases, 353
        selected segments, 5,012 selected candidate IDs, and zero
        omitted-capacity segments
    create the checked score-runner config only afterward, binding that accepted
        coverage-suite hash plus execution, frozen, definition, profile, and
        layout identities
    before InitMozc, require the score command raw argv to contain exactly one
        nonempty --name=value occurrence of --score_config, --execution_config,
        --frozen_corpus, --native_coverage, --manifest_directory,
        --model_directory, --output_binary, and --output_textproto, and reject
        every other, duplicate, empty, or split-form flag
    before ranker creation or model open, hash, parse, recursively validate, and
        join the raw score config, execution config, frozen corpus, and accepted
        coverage suite; in this fresh process, also reread and retain all three
        raw manifests and compute all three raw hashes before parsing any
        manifest, then parse and recursively validate all three, require their
        configured hashes and record layouts, and require equality outside
        record layout

    define the fixed permutation by reversing only natively selected candidate
        objects among their existing unprotected slots; leave every protected
        and unselected candidate in its original slot
    for each objective and original frozen request:
        construct its fixed-permutation request without changing native coverage
        audit the permuted request and require the accepted ordered rankable-
            segment count, disposition, selected-ID count, and selected-ID
            membership; recompute the unprotected count from the original and
            permuted requests and require equality; require selected-ID order
            equals the exact prescribed reversal of the accepted original
            order; do not claim persisted evidence for transient window or
            omission counts
    require the complete permuted-capacity validation pass for all objectives
        and cases to finish before invoking any scoring operation
    for model in [rinna_37m_f16_control, qwen2_5_0_5b_f16]:
        verify source revision, license, source weight, tokenizer, F16 GGUF,
            tensor identity, tokenizer identity, and reference logit parity
        for each objective and original frozen request:
            reconstruct the same validated fixed-permutation request
            score the original and permuted native candidates with configured
                shared-trie F16 and require finite totals and edge scores
            derive native orders with the production sort and merge both through
                the shared pure merger
            for each candidate, compute:
                full_score = sum every varying continuation, following-context,
                    and terminal edge
                continuation_count = count those scored edges
                require continuation_count > 0
            canonicalize either signed zero to positive zero and require complete
                candidate-ID maps of full score and continuation count are
                identical under the fixed permutation
            require each order independently equals the production stable order
                for its own request and each merge is complete and valid
            store the original result once; persist only source line, segment and
                candidate IDs, full original index, signed cost, canonical
                fixed64 full score bits, and continuation count
        build and fully validate the complete score artifact, then serialize
            binary and review text fully in memory before writing either
        repeat the complete score command in a fresh process
        require binary and review outputs are byte-identical
        give none of these runners an answer or answer-hash flag

    report the two models and three layouts with development answers
    for aggregation in [full_sum, continuation_mean]:
        for order_prior in [0, 0.25, 0.5, 1.0]:
            for exact_key_penalty in [0, 100]:
                for each selected candidate:
                    base = full_score if aggregation == full_sum
                        else full_score / continuation_count
                    calibrated = base
                        - order_prior * full_original_index
                        - exact_key_penalty if the Mozc top differs from the
                            segment key and candidate value equals that key
                rank by calibrated score and original index
                compute exact metrics and paired clean key copy counts
                apply the unchanged development gates
    require the report config SHA256 is
        7b65a859ddcaf4c4d12febec30cc559a883a20002cc7f4494c1accc96a085a66
    repeat the report with the independently repeated score inputs
    require byte-identical binary and canonical JSON reports
    require binary SHA256
        0863da2dc27c581934490f028e293658aadd8944911299d2cf38f5f4c6405d95
    require canonical JSON SHA256
        c9dce5deb448477e6c743aa805d8d6f5c2f049e7592d14143dcc6ed4b2fa4d45
    require the report contains 96 cells and no passing cell
    require selected is false and every selection field is absent
    require the strongest rinna cell is natural text only, full sum, order
        prior 1.0, exact key penalty 100, 21 wins, 11 regressions, net gain 10,
        and bootstrap lower gain -1
    require the strongest Qwen cell uses the same policy with 17 wins, 25
        regressions, net gain -8, and bootstrap lower gain -21
    record the verified failure, close development answers, and return

function diagnose_rejected_scoring_and_select_objective():
    verify the retained frozen, capacity, semantic, and quality artifact hashes
    recompute from retained requests and orders:
        579 changed segment tops
        440 newly promoted values exactly equal their target segment key
        85 of 87 Mozc-correct regressions contain an exact-key promotion
        538 of 579 substitutions have higher Mozc cost
        all 150 cases with at least two substitutions are wrong
    require merge replay, capacity coverage, and both deterministic runs agree
    label exact-key and rank-cutoff counterfactuals as attribution only
    forbid converting either counterfactual into a production rule

    source125_candidates = ["満たせば", "みたせば"]
    hold complete reading, baseline-right sentence, mode, and candidates fixed
    actual = score with serialized target key "みたせば"
    counterbalanced = score with serialized target key "満たせば"
    margin(result) = score("みたせば") - score("満たせば")
    under the checked F32 weights, F32 KV, and disabled flash attention:
        require shared and independent edge and total differences <= 1e-4
        require actual margin approximately +7.79768
        require counterbalanced margin approximately +0.04521
        record the approximately 7.75247 reduction and unchanged winner
        record dominant target-field contribution for source 125
        record the counterbalanced result as a fixed kana-candidate preference
    repeat both interventions under F16 and Q4_K_M configured profiles
    record scoring-graph-specific margins and identical token and edge metadata
    require both scoring graphs preserve the intervention direction
    compare matched F32 and Q4_K_M configured profiles
    record that the Q4_K_M model/runtime cell amplifies the counterbalanced
        fixed candidate preference without assigning that effect to quantization
    do not relax or reinterpret the strict F32 equivalence gate
    do not use this probe to choose coefficients or a promotion policy

    call prepare_quality_regression_objective_suite()

function benchmark_latency(frozen_requests, checked_model, config):
    verify artifact, corpus, config, request, and executable hashes
    derive checked 5, 10, and 26 candidate requests from full snapshots
    build effective manifests that change only paired thread counts
    require ten idle one-second windows before model loading
    load rankers and execute the fixed ten-call warmup schedule
    require ten idle one-second windows after warmup
    for each of 100 deterministic rotating rounds:
        for each candidate-count and thread-count cell:
            capture system, process, power, affinity, priority, and QPC state
            call production Rank exactly once
            capture the ending state
            require the ranking order remains identical
            compute background CPU as system busy minus benchmark process CPU
            if any state or contamination check fails:
                write raw invalid evidence with no percentiles and stop
            append integer elapsed QPC ticks
    write all raw samples once
    compute nearest-rank p50, p95, and p99 without rerunning inference
```

## Implementation stages

1. Import the pinned Mozc source and preserve all upstream licenses.
2. Run the smallest official Windows x64 build target that proves the toolchain.
3. Locate the session-owned post-generation integration points and record them
   here.
4. Add the model-neutral request, response, ranker interface, and merger.
5. Add focused tests for ranking, protected candidates, duplicates, invalid
   references, and stale request identifiers.
6. Extract candidate limiting without changing default converter behavior.
7. Inject a synchronous fake service and verify one ranking call per completed
   session generation path.
8. Add persisted opt-in configuration, live mutation invalidation, a persistent
   process-owned worker, per-session queue replacement, bounded waits,
   cooperative cancellation, structured diagnostics, and ordered shutdown.
9. Select a local model and runtime from measured latency and Japanese ranking
   quality, then implement and benchmark the backend.
10. Package under independent product branding and add end-to-end TSF tests.

## Verification policy

Use focused Bazel targets during development. Do not run Mozc's full test suite
or build the installer merely to verify a localized candidate-ranking change.
