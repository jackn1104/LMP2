# Contributing

Changes to scientific algorithms require:

1. a stated mathematical and numerical contract;
2. a small deterministic unit test;
3. comparison with an independent result or the unchanged reference path;
4. Debug and undefined-behavior-sanitizer builds;
5. serial and MPI/OpenMP agreement where the change affects parallel code;
6. documentation of thresholds, tolerances, units, and untested limits.

Never weaken a tolerance merely to pass a test. Do not commit calculation
outputs, checkpoints, credentials, machine-specific paths, or personal data.
Preserve complete occupied and virtual spaces unless a truncation is explicit
in the method name and output metadata.

