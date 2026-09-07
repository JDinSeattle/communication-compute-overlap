# Communication–Compute Overlap

A small **MSCCL++ PortChannel** experiment for determining when chunked GPU production and asynchronous data movement improve end-to-end completion time.

**Status:** the real CUDA extension compiles; 72 single-GPU compute/reference cases pass NVIDIA Compute Sanitizer; CPU protocol and analysis tests pass. **Two-GPU communication correctness and overlap performance remain unvalidated.**

## The experiment

Each rank computes a generation-dependent `uint32` payload and sends it to the other GPU using the pinned MSCCL++ CudaIpc connection and host proxy. Three configurations share the same compute workload:

1. **Serial:** compute the full message, then issue the full transfer.
2. **16 KiB chunks:** publish each completed source slice while producing the next.
3. **256 KiB chunks:** the same protocol with larger chunks.

The single-CTA producer uses system fences before publishing to the proxy. Source slices remain immutable until local completion; a host bootstrap barrier acknowledges consumption before the destination can be overwritten in the next iteration. No delay is used to make a race disappear.

The independent host reference composes the payload's affine transform by exponentiation, rather than repeating the GPU's compute loop. Every received element is checked against its rank and iteration, with a checksum retained in each record. This is a controlled integer workload, not a claim about training or Tensor Core throughput.

## Evidence

| Check | Local result |
|---|---|
| Pinned MSCCL++ library and CUDA extension | Built with CUDA 13.2, `sm_89`, CudaIpc; IB/GDRCopy disabled |
| CPU regression suite | 7 tests passed |
| Finite protocol model | Ordered 3-generation model explored 25 states; missing acknowledgement and early publication each produced a counterexample |
| Real GPU compute/reference test | 72 cases passed; Compute Sanitizer memcheck reported 0 errors |
| Two-GPU eligibility | Correctly blocked on the one-GPU workstation |
| Communication correctness / actual overlap / speedup | **Pending target hardware** |

See [design and happens-before reasoning](docs/DESIGN.md), [the local report](docs/REPORT.md), [the target runbook](docs/TWO_GPU_RUNBOOK.md), [original scope](PROJECT_SPEC.md), and [the evidence snapshot](evidence/snapshot/).

## Build and check

Requirements: Linux, Git, CMake ≥ 3.24, a C++20-capable CUDA toolchain and NUMA development headers/library. The tested toolkit is CUDA 13.2. Use the architecture of the actual GPU.

```bash
python3 -m unittest discover -s tests -v
python3 model.py
python3 scripts/build.py --arch 89
build/overlap --self-test
compute-sanitizer --tool memcheck --error-exitcode=9 build/overlap --self-test
build/overlap --preflight
```

The builder downloads exact commits from `upstream.lock.json`, builds MSCCL++ without IB, GDRCopy, Python bindings or external collectives, and links this extension. If NUMA is installed outside standard paths, pass `--numa-include` and `--numa-library`.

On a machine with two peer-accessible GPUs:

```bash
python3 run.py --binary build/overlap --out evidence/local/two-gpu-run
```

The supervisor reaps both ranks on failure or timeout. It accepts performance samples only after both ranks report correct, complete sequences. Paired repetitions compare the slower-rank completion time, with a descriptive ±5% no-material-gain region. Profiles and repeated-trial uncertainty still require review before qualification.

```bash
python3 scripts/export_evidence.py --out evidence/new-snapshot
python3 scripts/export_evidence.py --verify evidence/new-snapshot
```

## Role relevance

This project demonstrates GPU/host communication interfaces, source-buffer ownership, release/acquire reasoning, counterexample-driven protocol checks and performance gating. [Interview notes](docs/INTERVIEW.md) contain evidence-safe resume wording. CPU models and a single-GPU compute check cannot establish remote device memory ordering.
