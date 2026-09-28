# LLM日本語入力 (LLM Japanese Input)

LLM日本語入力 is a Windows Japanese input method based on
[Mozc](https://github.com/google/mozc). It keeps Mozc's composition,
dictionaries, learning, and candidate window, and reorders Mozc's conversion
candidates with a small kana-kanji conversion language model that runs locally
on the CPU. No text is sent over the network.

LLM日本語入力は [Mozc](https://github.com/google/mozc) をもとにした Windows 用の
日本語入力です。Mozc の変換候補を、PC 上で動く小さなかな漢字変換モデルで並べ替えます。
入力した文章が外部に送信されることはありません。

This project is not affiliated with or endorsed by Google. Google Japanese
Input is a trademark of Google LLC.

## How it works

Mozc produces its normal candidates. For each conversion segment, the ranker
scores Mozc's top five candidates with
[zenz-v3.2-small](https://huggingface.co/Miwa-Keita/zenz-v3.2-small-gguf), a
kana-kanji conversion model from the
[Zenzai](https://github.com/azooKey/AzooKeyKanaKanjiConverter) engine, given
the reading and the surrounding text. The score is adjusted by a penalty for
candidates that copy the reading and by a prior on Mozc's original order, and
the segment is reordered. Inference uses
[llama.cpp](https://github.com/ggml-org/llama.cpp) on the CPU. If ranking does
not finish within 250 milliseconds, Mozc's original order is shown.

Top-one accuracy of the whole conversion, compared with Mozc:

| Corpus | Mozc | LLM日本語入力 |
|---|---|---|
| Mozc quality regression, development (219 cases) | 164 | 186 |
| Mozc quality regression, holdout (72 cases) | 42 | 53 |
| AJIMEE-Bench (200 cases) | 103 | 145 |

The method, the predeclared evaluation, and the full results are in
[.cursor/plans/local-reranker-zenz.md](.cursor/plans/local-reranker-zenz.md).

## Requirements and limitations

* Windows 10 1809 or later on an x64 processor with AVX2.
* Ranking takes a median of about 90 milliseconds per short conversion on an
  eight-core laptop processor. Long sentences often exceed the 250 millisecond
  limit and show Mozc's order.
* The installer is about 100 MB because it includes the model.

## Install

Download `LLMJapaneseInput64-<version>.msi` from the
[Releases page](https://github.com/kohei-kawaguchi/llm-japanese-input/releases)
and open it.

## Build

The Windows build follows Mozc's
[Windows build instructions](docs/build_mozc_in_windows.md). The details of
the host setup are recorded in
[.cursor/plans/daily-use-ime-build.md](.cursor/plans/daily-use-ime-build.md).

### Prerequisites

* 64-bit Windows 10 or later.
* Visual Studio 2022 Community with the Windows 11 SDK, the MSVC v143 x64/x86
  build tools, and C++ ATL for the v143 build tools (x86 and x64).
* Python 3.12 or later.
* Git for Windows, whose Git Bash runs the build command and provides `curl`.
* The .NET 8 SDK. A .NET runtime alone is not sufficient, because the
  dependency update restores WiX with `dotnet tool restore`. Install it with
  `winget install Microsoft.DotNet.SDK.8` and confirm that
  `dotnet --list-sdks` lists an 8.0 version.
* nuget.org as a NuGet package source. When `dotnet nuget list source` shows
  no source, add it with
  `dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org`.

Bazel records the MSVC include directories when it first configures the C++
toolchain. When a Visual Studio component such as ATL is added after a build
has run, run `../.tools/bazelisk.exe fetch --configure --force` in `src` so that
the new include directories are recorded.

Bazelisk is not a prerequisite. The build downloads it to `.tools/` and checks
its SHA256.

### Build steps

Python, Bazelisk, and the Qt build PATH differ by device. They are read from
`scripts/config/windows_build.json`, which stays on the device and is not
committed. Copy `scripts/config/windows_build.example.json` to that path and
replace `python` with the absolute path of the local Python interpreter. Paths
use forward slashes. Change `qt_path_directories` only when Windows is not
installed in `C:/WINDOWS`. From the repository root:

```
bash scripts/run.sh windows-build
```

The command changes to `src` and runs the dependency update, the Qt build, and
the package build.

The command copies the installer to `dist/LLMJapaneseInput64-<version>.msi`,
where the version is read from the version file of the same build. Install it
by opening that file. The copy is needed because Bazel places the installer
behind the `src/bazel-bin` junction, and Windows Installer does not open a
package through that junction.

The model is downloaded by Bazel at a pinned revision and SHA256 and is
installed next to the conversion server. Raise `BUILD_OSS` in `src/version.bzl`
for every installer release, because Windows Installer does not replace
executables whose version is unchanged.

## Turning ranking off

Ranking is on by default. Open the properties dialog and clear
「LLMで変換候補を並べ替える」 to use Mozc's original order.

## License

Source code from Mozc is licensed under the BSD 3-Clause license by Google.
Files added or modified by this project are licensed under the BSD 3-Clause
license by the LLM Japanese Input Authors. Dictionaries and other third-party
data keep their own licenses. See [LICENSE](LICENSE).

Components that are downloaded at build time and included in the installer:

* llama.cpp, MIT license.
* zenz-v3.2-small by Keita Miwa, Apache License 2.0.

Their notices are included in
[src/data/installer/credits_en.html](src/data/installer/credits_en.html).
Evaluation outputs derived from
[AJIMEE-Bench](https://github.com/azooKey/AJIMEE-Bench) are distributed under
CC BY-SA 3.0 as noted in [LICENSE](LICENSE).

## Upstream

This repository is based on Mozc commit
`851c3fe33060d2a6090363e4d7ec44fafde2c03d`. See [UPSTREAM.md](UPSTREAM.md).
Mozc's own documentation remains under [docs/](docs/).
