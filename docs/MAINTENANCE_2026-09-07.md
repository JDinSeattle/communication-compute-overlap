# Maintenance verification — 2026-09-07

## Problem and implemented behavior

The original analysis labeled a median of three paired trials as beneficial/regressed using a 5% margin without uncertainty. `analysis.py` now uses an exact binomial/order-statistic interval for the population median of paired process-repeat ratios. Each ratio compares serial and chunked run medians, each based on the slower rank. Iterations inside a process are not resampled as independent trials.

The tightest available interval with coverage at least 95% is selected. Fewer than six repeats cannot produce a finite interval by this method; the default is seven. Benefit requires the entire interval above 1.05, regression below 0.95, and within-margin requires the interval inside [0.95, 1.05]. Other results are inconclusive. Coverage assumes independent representative pairs, is pointwise, and does not address multiple comparisons, thermal drift or biased ordering.

`run.py` also requires two complete rank logs, strict metadata types, sequence/checksum fields, a final successful record, and a valid two-direction hardware preflight. It records executable-launch failures, kills/reaps the peer after errors, and withholds timings after failed evidence. Checksum presence is a diagnostic contract; full payload correctness remains the native extension's independent host-reference check.

## Actual validation

- Baseline: seven Python tests. Current: 20 tests. Four real local process-failure tests cover launch failure, peer exit, timeout and exit-zero/missing evidence. A fifth real CLI test uses explicitly synthetic CPU rank workers to exercise serialization and comparison assembly; it is not a GPU run.
- Interval coverage is independently checked by exhaustively enumerating all sign patterns for 6–12 pairs. Positive but noisy medians remain inconclusive, and three apparently fast repeats remain insufficient.
- The unchanged CUDA extension reran 72 real single-GPU compute/reference cases. NVIDIA Compute Sanitizer memcheck reports 0 errors.
- The native two-GPU preflight and runner correctly block on the one-GPU host, exit 2, with no performance records. No real dual-rank transfer, overlap benefit or production performance is claimed.

[Verification and source/binary hashes](../evidence/maintenance-20260907/verification.json), [raw logs and receipts](../evidence/maintenance-20260907/), and [the hardware runbook](TWO_GPU_RUNBOOK.md) define the reproducible boundary. A final admission-only source check is distinct from the earlier GPU execution snapshot.

```bash
python3 -m unittest discover -s tests -v
build/overlap --self-test
compute-sanitizer --tool memcheck --error-exitcode=9 build/overlap --self-test
python3 run.py --binary build/overlap --out evidence/local/new-two-gpu-run
```

The correct finite protocol model is retained; it cannot prove CUDA memory ordering. Actual CudaIpc/PortChannel correctness, fault behavior during GPU communication, Nsight profiling and target-host repeated timing remain required before qualification. The Python supervisor is not a production cluster scheduler.
