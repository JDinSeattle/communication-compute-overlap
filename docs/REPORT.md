# Local validation report

## Executed on real hardware

The MSCCL++ core library and extension compiled with CUDA 13.2 for `sm_89`. The extension's `--self-test` executed 72 combinations on an RTX 4090: three element counts (including tails), four compute intensities, two rank patterns and three sequence numbers.

Each case compared every produced word against an independent host affine-composition reference and checked the untouched tail of the allocation. NVIDIA Compute Sanitizer memcheck reported `ERROR SUMMARY: 0 errors`. This establishes local compute/reference and checked memory-boundary behavior for these cases; it does not test a remote channel.

## Executed on CPU

The original seven regression tests passed. The September maintenance suite now has 20 tests; see [the maintenance report](MAINTENANCE_2026-09-07.md). They include finite protocol exploration, failing counterexamples when acknowledgement/publication ordering is removed, rejecting missing/duplicate/incorrect rank records, and selecting the slower rank's completion time. The correct three-generation model explored 25 states.

## Hardware gate

The real compiled binary reports one CUDA device, unavailable two-device peer capability, and a blocked status. The supervisor returns 2 and writes an empty performance result. There are no simulated GPU bandwidth or overlap numbers in this project.

## Remaining work

Two-GPU channel correctness, device synchronization stress, actual serial/chunk timings, proxy/GPU profiles and performance regions remain unverified. The original 70–130-hour project scope must not be represented as fully completed on the basis of this local checkpoint.
