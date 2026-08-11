#!/usr/bin/env python3
"""Public-interface and known-case checks for the LMP2 launcher."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    if len(sys.argv) != 4:
        raise RuntimeError("expected launcher, engine, and source directory")
    launcher = Path(sys.argv[1]).resolve()
    engine = Path(sys.argv[2]).resolve()
    source = Path(sys.argv[3]).resolve()

    with tempfile.TemporaryDirectory(prefix="lmp2-cli-test-") as temporary:
        root = Path(temporary)
        output = root / "known-case"
        known_case = run(
            [
                sys.executable,
                str(launcher),
                str(source / "examples" / "water_monomer.xyz"),
                "--engine",
                str(engine),
                "--threads",
                "2",
                "--output",
                str(output),
            ]
        )
        require(
            known_case.returncode == 0,
            "known-case calculation failed:\n" + known_case.stdout + known_case.stderr,
        )
        require(
            "LMP2 completed\n  Basis:        aug-cc-pVDZ" in known_case.stdout,
            "completed summary does not start with the full basis name",
        )
        with (output / "energy.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 1, "known-case energy.csv has the wrong row count")
        row = rows[0]
        require(
            abs(float(row["rhf_energy_hartree"]) - (-76.040761392138137)) < 5.0e-10,
            "known-case RHF energy changed",
        )
        require(
            abs(float(row["lmp2_correlation_hartree"]) - (-0.220148353960764))
            < 5.0e-10,
            "known-case LMP2 correlation energy changed",
        )
        require(row["energy_backend"] == "localized_cached_ovov", "wrong backend")
        require(int(row["canonical_ovov_bytes"]) == 0, "canonical OVOV was stored")
        require(int(row["low_confidence_occ"]) == 0, "low-confidence occupied LMO")
        require(int(row["low_confidence_vir"]) == 0, "low-confidence virtual LMO")
        require(
            int(row["localized_ovov_bytes"])
            == int(row["localized_ovov_stored_bytes_global"]),
            "localized OVOV ownership audit failed",
        )

        truncated_output = root / "truncated"
        truncated = run(
            [
                sys.executable,
                str(launcher),
                str(source / "examples" / "water_monomer.xyz"),
                "--engine",
                str(engine),
                "--threads",
                "2",
                "--mode",
                "1m2m",
                "--checkpoint",
                str(output / "rhf.h5"),
                "--output",
                str(truncated_output),
            ]
        )
        require(
            truncated.returncode == 0,
            "truncated calculation failed:\n" + truncated.stdout + truncated.stderr,
        )
        require("  1M:" in truncated.stdout, "truncated summary omitted 1M")
        require("  2M:" in truncated.stdout, "truncated summary omitted 2M")
        require("  1M+2M:" in truncated.stdout, "truncated summary omitted its sum")
        require("  3M:" not in truncated.stdout, "truncated summary printed 3M")
        require("  4M:" not in truncated.stdout, "truncated summary printed 4M")

        malformed = root / "malformed.xyz"
        malformed.write_text(
            "3\ninvalid record order\nH 0 0 0\nO 0 0 1\nH 0 0 2\n",
            encoding="utf-8",
        )
        rejected = run(
            [
                sys.executable,
                str(launcher),
                str(malformed),
                "--engine",
                str(engine),
                "--dry-run",
            ]
        )
        require(rejected.returncode != 0, "malformed O-H-H ordering was accepted")

        overwrite = run(
            [
                sys.executable,
                str(launcher),
                str(source / "examples" / "water_monomer.xyz"),
                "--engine",
                str(engine),
                "--output",
                str(output),
                "--dry-run",
            ]
        )
        require(overwrite.returncode != 0, "nonempty output directory was accepted")

    print("LMP2 public CLI checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
