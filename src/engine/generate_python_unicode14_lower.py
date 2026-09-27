#!/usr/bin/env python3

# Copyright 2026 LLM Japanese Input Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the distribution.
#     * Neither the name of LLM Japanese Input Authors nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Generates the Unicode 14 tables used by PythonUnicode14Lower."""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import unicodedata
import urllib.request
from pathlib import Path


UNICODE_VERSION = "14.0.0"
PYTHON_VERSION = (3, 11)
SOURCES = {
    "UnicodeData.txt": (
        "https://www.unicode.org/Public/14.0.0/ucd/UnicodeData.txt",
        "36018e68657fdcb3485f636630ffe8c8532e01c977703d2803f5b89d6c5feafb",
    ),
    "SpecialCasing.txt": (
        "https://www.unicode.org/Public/14.0.0/ucd/SpecialCasing.txt",
        "c667b45908fd269af25fd55d2fc5bbc157fb1b77675936e25c513ce32e080334",
    ),
}


def download_sources() -> dict[str, str]:
    sources = {}
    for name, (url, expected_sha256) in SOURCES.items():
        with urllib.request.urlopen(url) as response:
            data = response.read()
        actual_sha256 = hashlib.sha256(data).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{name} SHA-256 is {actual_sha256}, expected {expected_sha256}"
            )
        sources[name] = data.decode("utf-8")
    return sources


def parse_code_points(field: str) -> tuple[int, ...]:
    return tuple(int(value, 16) for value in field.split())


def parse_lower_mappings(
    unicode_data: str, special_casing: str
) -> dict[int, tuple[int, ...]]:
    mappings: dict[int, tuple[int, ...]] = {}
    for line in unicode_data.splitlines():
        fields = line.split(";")
        code_point = int(fields[0], 16)
        if fields[13]:
            mappings[code_point] = (int(fields[13], 16),)

    for line in special_casing.splitlines():
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        fields = [field.strip() for field in content.split(";")]
        code_point = int(fields[0], 16)
        lower = parse_code_points(fields[1])
        condition = fields[4]
        if not condition:
            mappings[code_point] = lower
    return mappings


def verify_against_python(mappings: dict[int, tuple[int, ...]]) -> None:
    if sys.version_info[:2] != PYTHON_VERSION:
        raise RuntimeError(
            f"generator requires Python {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}"
        )
    if unicodedata.unidata_version != UNICODE_VERSION:
        raise RuntimeError(
            f"Python Unicode version is {unicodedata.unidata_version}, "
            f"expected {UNICODE_VERSION}"
        )

    for code_point in range(0x110000):
        expected = tuple(ord(value) for value in chr(code_point).lower())
        actual = mappings.get(code_point, (code_point,))
        if actual != expected:
            raise ValueError(
                f"U+{code_point:04X} maps to {actual}, Python maps to {expected}"
            )

def logical_table_sha256(
    mappings: list[tuple[int, tuple[int, ...]]],
) -> str:
    digest = hashlib.sha256()
    for source, targets in mappings:
        digest.update(struct.pack(">IB", source, len(targets)))
        for target in targets:
            digest.update(struct.pack(">I", target))
    return digest.hexdigest()


def format_code_point(code_point: int) -> str:
    return f"0x{code_point:06X}"


def render(
    mappings_by_code_point: dict[int, tuple[int, ...]],
) -> str:
    mappings = sorted(
        (source, targets)
        for source, targets in mappings_by_code_point.items()
        if targets != (source,)
    )
    targets: list[int] = []
    records: list[tuple[int, int, int]] = []
    for source, mapped in mappings:
        records.append((source, len(targets), len(mapped)))
        targets.extend(mapped)

    lines = [
        "// Generated by engine/generate_python_unicode14_lower.py.",
        "// Unicode version: 14.0.0; Python reference: 3.11.",
        "// See engine/python_unicode14_lower_unicode_notice.txt.",
    ]
    for name, (_, source_sha256) in SOURCES.items():
        lines.append(f"// {name} SHA-256: {source_sha256}")
    lines.extend(
        (
            "// Logical table SHA-256: "
            + logical_table_sha256(mappings),
            "",
            "constexpr LowerMapping kLowerMappings[] = {",
        )
    )
    lines.extend(
        f"    {{{format_code_point(source)}, {offset}, {length}}},"
        for source, offset, length in records
    )
    lines.extend(("};", "", "constexpr char32_t kLowerMappingData[] = {"))
    for start in range(0, len(targets), 8):
        values = ", ".join(
            format_code_point(value) for value in targets[start : start + 8]
        )
        lines.append(f"    {values},")
    lines.extend(("};", ""))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("python_unicode14_lower_data.inc"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sources = download_sources()
    mappings = parse_lower_mappings(
        sources["UnicodeData.txt"], sources["SpecialCasing.txt"]
    )
    verify_against_python(mappings)
    generated = render(mappings)

    if args.check:
        if args.output.read_text(encoding="utf-8") != generated:
            raise ValueError(f"{args.output} is not up to date")
    else:
        args.output.write_text(generated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
