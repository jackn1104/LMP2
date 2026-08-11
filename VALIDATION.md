# Validation record

## 0.1.0 local validation

Date: 2026-08-10

An arm64 macOS release build with AppleClang 16 completed the included
W1/aVDZ full calculation through the public `LMP2` command. The seven-point
minimax result was:

```text
RHF energy          -76.040761392138137 Eh
MP2 correlation     -0.220148353960764 Eh
Total MP2 energy    -76.260909746098903 Eh
```

The calculation used `localized_cached_ovov`, reported zero canonical OVOV
bytes, stored exactly one logical localized OVOV tensor globally, and had
zero low-confidence occupied and virtual orbitals.

The standalone Debug test configuration passed all four CTest gates:

```text
lmp2_resource_tests
lmp2_block_symmetry_tests
lmp2_fullspace_energy_tests
lmp2_public_cli
```

The same four gates passed with undefined-behavior sanitization. The
four-center block-symmetry sanitizer test required 362 seconds, so its test
timeout is 600 seconds. No undefined-behavior diagnostic was reported.

## Larger external reference

The W20/aVDZ seven-point values documented in the README come from a completed
four-node calculation outside this package build. They are retained as an
external comparison and are not presented as a locally reproduced result.

## Checks not completed for this package snapshot

- The standalone package was not rerun on a multi-node cluster after
  extraction from its development repository.
- The Docker build recipe was not executed locally.
- The hosted Ubuntu workflow was not run because the public repository does
  not yet exist.
- Address sanitization was not completed; undefined-behavior sanitization was
  completed instead.

