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
#     * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
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

"""Command-line entry point for the deterministic AJIMEE quality reporter."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from engine.evaluation import ajimee_quality_report


_PATH_FLAGS = (
    "config",
    "capacity_audit_config",
    "frozen_corpus",
    "answer_corpus",
    "capacity_audit",
    "semantic_results",
    "output_binary",
    "output_json",
)


def _build_argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
  for name in _PATH_FLAGS:
    parser.add_argument(f"--{name}", required=True, type=Path)
  return parser


def main(argv: Sequence[str] | None = None) -> None:
  args = _build_argument_parser().parse_args(argv)
  inputs = ajimee_quality_report.AjimeeQualityReportInputs(
      report_config_textproto=args.config.read_bytes(),
      capacity_audit_config_textproto=args.capacity_audit_config.read_bytes(),
      frozen_corpus_binary=args.frozen_corpus.read_bytes(),
      answer_corpus_binary=args.answer_corpus.read_bytes(),
      capacity_audit_binary=args.capacity_audit.read_bytes(),
      semantic_results_binary=args.semantic_results.read_bytes(),
  )
  report = ajimee_quality_report.build_quality_report(inputs)
  artifacts = ajimee_quality_report.serialize_quality_report(report)
  args.output_binary.write_bytes(artifacts.binary)
  args.output_json.write_bytes(artifacts.canonical_json)


if __name__ == "__main__":
  main()
