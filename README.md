# Communication–Compute Overlap

A generation-scoped buffer ownership model and MSCCL++ PortChannel study. Paired trials use the slower rank, complete two-rank evidence, an order-statistic interval, and a predeclared materiality threshold.

## Confirmed qualification results

These are the user-confirmed results from a separate cloud test run, recorded in the experience bank. The device, workload, timing round, and counting boundaries below remain part of each result. They are distinct from the CPU checks performed in this checkout; cloud-hosted testing does not imply production deployment.

- Implemented a generation-scoped source-ownership protocol: a single-CTA producer writes independent source slices per generation, the consumer ACKs after finishing, and reuse happens only after the matching ACK, because an initiated transfer is not proof that the consumer finished reading.

- Replayed the 72 single-GPU compute/reference cases (3 counts 257/4096/65536 × 4 strengths 1/4/16/64 rounds × 2 rank modes × 3 generations) under seed=2026 alongside 20 CPU protocol regressions, and all payload and untouched-tail values matched an independent CPU affine-composition reference.

- Confirmed two faulty protocol models fail as designed: removing the ACK constraint overwrites the old buffer at generation 2, and removing the generation match accepts a stale ACK.

- Admitted two-rank outputs strictly: six fixtures (missing rank, duplicate rank, different payload, NaN time, wrong generation, timeout) were each rejected, and rejection terminates and reaps both child processes.

- Built a conservative paired-trial decision over 31 clearly labeled synthetic pairs, with speedup defined as serial slower-rank time divided by chunked slower-rank time, randomized order and the two ranks never treated as independent samples: assuming a paired median of 1.06 and order statistics 10 and 22 at [0.99, 1.12], the distribution-free median interval covers about 97%, but the lower bound does not clear the pre-set 1.03 materiality threshold, so the benefit stays uncertain.

- Kept the two-GPU path unqualified while pinning down the intended design — two peer-accessible devices on one machine, serial and chunked paths, chunk=16 KiB, slower rank as trial time — because the gate exits 2 and publishes no performance results.

## Implementation and reproduction

| Contract | Implementation |
|---|---|
| Ownership model and mutations | [model.py](model.py) |
| Rank acceptance and cleanup | [run.py](run.py) |
| Paired decision procedure | [analysis.py](analysis.py) |
| Native compute and intended transport | [src](src) |

Run each experiment into a fresh output directory to preserve earlier evidence.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/export_evidence.py --verify evidence/maintenance-20260907
python3 run.py --help
```

Regression entry points: [tests/test_protocol.py](tests/test_protocol.py), [tests/test_acceptance.py](tests/test_acceptance.py).

## Scope and evidence

The two-GPU transport path remains unqualified. Synthetic 31-pair statistics validate the decision procedure and are not measured transport performance.

- There is no two-GPU measurement: the gate exits 2 and publishes no performance results, and overlap is the research object rather than a verified communication-overlap result.

- The 1.06 median and the [0.99, 1.12] order-statistic interval come from 31 clearly labeled synthetic pairs and test only the decision procedure; they are not a two-GPU measurement.

- The lower bound does not clear the 1.03 materiality threshold and the interval includes no benefit, so no victory is claimed; the interval is pointwise and is not a simultaneous confidence guarantee over the entire parameter grid.

- The 25-state abstract protocol exhausts only a finite model with three generations, fixed participants and a given message order; it can surface missing-ACK counterexamples but does not prove the real system deadlock-free, nor CUDA memory ordering, DMA/RDMA visibility or proxy-thread progress.

- A source slice can be reused only after the corresponding generation ACK; initiation alone is not safety, and single-GPU tests cannot prove real DMA/RDMA visibility.

- Environment scope: Ubuntu 24.04 and a single RTX 4090, C++20, CUDA 13.2, Python 3.14; the real PortChannel two-GPU path remains unqualified, and the pinned MSCCL++ 0.10.1.post1.dev6+g626734ed7 package is the study's fixed revision, not a rewrite of the historical pin.

The [previous README](README.historical.md) preserves earlier setup details, design discussion, and historical measurements. Its older counts, splits, versions, and timing cohorts must not be mixed with the confirmed round above. [Result provenance](docs/experience-bank-results.json) retains the confirmed bullet text; [checkout validation](docs/checkout-validation.md) records what was actually rerun here.
