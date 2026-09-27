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

"""Runs retained rinna GGUF checks against explicit local artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--llama-cpp-dir", required=True, type=Path)
  parser.add_argument("--model-dir", required=True, type=Path)
  parser.add_argument("--python", required=True, type=Path)
  parser.add_argument("--quantizer", required=True, type=Path)
  args = parser.parse_args()

  package_dir = Path(__file__).resolve().parent
  workspace_dir = Path(os.environ["BUILD_WORKSPACE_DIRECTORY"])
  python_path = (workspace_dir / args.python).resolve()
  llama_cpp_dir = (workspace_dir / args.llama_cpp_dir).resolve()
  model_dir = (workspace_dir / args.model_dir).resolve()
  quantizer_path = (workspace_dir / args.quantizer).resolve()
  verifier = package_dir / "verify_rinna_gguf.py"
  config = package_dir / "rinna_gguf_verification_config.json"
  reports = {
      "f32": package_dir / "rinna_f32_gguf_verification_report.json",
      "f16": package_dir / "rinna_f16_gguf_verification_report.json",
      "q4_k_m": package_dir / "rinna_q4_k_m_gguf_verification_report.json",
  }
  verifier_environment = dict(os.environ)
  verifier_environment["PYTHONPATH"] = str(package_dir)
  for artifact, report in reports.items():
    command = [
        str(python_path),
        str(verifier),
        "--artifact",
        artifact,
        "--config",
        str(config),
        "--llama-cpp-dir",
        str(llama_cpp_dir),
        "--model-dir",
        str(model_dir),
        "--report",
        str(report),
        "--check-report",
    ]
    if artifact == "q4_k_m":
      command.extend(["--quantizer", str(quantizer_path)])
    subprocess.run(command, check=True, env=verifier_environment)


if __name__ == "__main__":
  main()
