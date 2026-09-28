# Build and install the IME for daily use with candidate ranking disabled

## Problem

Stages 1 to 8 are done, but nothing in this tree has been installed and used as
a Windows IME. Stage 9 is stalled on the local ranker, and stage 10 has not
started. Model ranking is opt-in: an absent `candidate_ranking_config` keeps
Mozc ordering, and the production `Engine` creates its ranking service with no
backend (`src/engine/engine.cc`). The product can therefore be built and used
now, and a ranker can be switched on later.

## Host prerequisites

The build host is Windows 11 x64 with Visual Studio 2022 Community and Python
3.13. The .NET 8 SDK is required because `update_deps.py` restores WiX 5 with
`dotnet tool restore`; a machine with only .NET runtimes fails with "No .NET
SDKs were found". The user NuGet configuration must list
`https://api.nuget.org/v3/index.json` as a package source; an empty
`packageSources` list fails with "No NuGet sources are defined or enabled".

## Build environment

The host system PATH contains MSYS2 MinGW and Miniconda library directories.
Qt's CMake searches the prefixes of PATH entries, finds their zstd and zlib
packages, and passes `C:\msys64\mingw64\include` to MSVC, whose headers then
conflict with the MinGW headers. The Qt build therefore runs with a PATH that
contains only the directories in `qt_path_directories` and the directory of
the configured Python interpreter. The host PATH setting itself is not changed.

## Build steps

Python, Bazelisk, and the Qt PATH are device settings. They are stored in
`scripts/config/windows_build.json` and are not committed. `python` is the
absolute path of the interpreter. `bazelisk` gives the download URL, the
SHA256, and the path of the binary. `qt_path_directories` lists the Windows
system directories used while Qt is built. The interpreter directory is
appended to that list.

Run from the repository root.

1. If `src/third_party/qt_src` or `src/third_party/qt_host` was produced by a
   build that used the host PATH, remove those directories. Their CMake cache
   keeps the MinGW packages.
2. `bash scripts/run.sh windows-build`
3. Install `dist/LLMJapaneseInput64-<version>.msi`.

The command changes to `src` and runs `build_tools/update_deps.py`,
`build_tools/build_qt.py --release --confirm_license`, and
`bazelisk build package --config release_build`. Bazelisk is downloaded to
the configured path when that file is absent, and its SHA256 is checked.
After the package build, the installer is copied from
`src/bazel-bin/win32/installer/LLMJapaneseInput64.msi` to `dist/` under a name
that carries the version in `src/bazel-bin/base/mozc_version.txt`. Windows
Installer does not open a package through the `src/bazel-bin` junction.

## Release versions

Every installer given to a user raises `BUILD_OSS` in `src/version.bzl`.
Windows Installer keeps an installed executable whose file version equals the
incoming one, so an upgrade with an unchanged version replaces only files
that are new and leaves the old `mozc_server.exe` and `mozc_tool.exe` in
place. Version 3.34.6240.100 is the first release with the zenz ranker.

## Product identity

The product name is LLM日本語入力, with the English name LLM Japanese Input,
the identifier prefix LLMJapaneseInput, and the manufacturer Kohei Kawaguchi.
The identity is applied by rewriting the values of the existing OSS branding
(`BRANDING = "Mozc"`) rather than adding a third branding, which would add a
branch to every file that already switches between Mozc and Google Japanese
Input. Display names, version resources, the installer package, the install
directory, the registry key, IPC and window class names, the cache service
name, and every OSS TSF GUID and the installer UpgradeCode are new, so the
product can coexist with an official Mozc install. Internal executable names
such as `mozc_server.exe` stay unchanged because users do not see them. The
Mozc icons are reused until the product has its own icons. The installer is
`dist/LLMJapaneseInput64-<version>.msi`. The previous Mozc install
is removed before this installer is installed, and its learning data is not
migrated.

## Acceptance

Check composition, conversion, prediction, learning, and the candidate window
in Notepad and a browser. Stop after this check and report before changing
product branding.
