# Target-hardware acceptance runbook

**Not executed here.** This workstation has one RTX 4090. The following steps are pending on a host with two mutually peer-accessible GPUs. The CudaIpc build does not test NIC, RDMA or inter-node paths.

1. Build for the actual GPU architecture. Record driver/toolkit, `nvidia-smi topo -m`, peer-access capability and the pinned MSCCL++ commit. Run `build/overlap --preflight`; stop on nonzero exit.
2. Run the 72-case single-GPU reference/memory check first. Then execute a short two-GPU sweep with at least two generations and tail messages such as 4112 bytes. Correctness must pass before interpreting any speedup.
3. Run the complete size/work matrix with the serial baseline and both chunk sizes. Repeat with changed rank CPU-affinity masks to vary scheduling. Inspect both ranks and every sequence; a peer failure or timeout invalidates the corresponding comparison.
4. Wrap each explicitly launched rank with NVIDIA Compute Sanitizer for a short exchange. Select an unused loopback bootstrap port and identical byte/chunk/work/iteration settings. The library/rank processes must be supervised with a wall-time deadline. Device memory checking complements, but does not replace, the synchronization argument.
5. Capture Nsight Systems CUDA and OS-runtime traces for one instance of each configuration. The rank command format is `build/overlap rank gpu lo:127.0.0.1:PORT bytes chunk_bytes work iterations warmup mode`. Prefix each rank with `nsys profile --trace=cuda,osrt --output=rankN ...`; retain the two trace files and matching manifests. Never interpret two overlapping rectangles alone as an application speedup.
6. Compare full exchange completion, CPU time, copy/compute activity and run-to-run dispersion. Report beneficial, no-material-gain and regressed regions only if the actual measurements contain them. An absent region is a result, not a reason to invent a speedup or slowdown.

## Minimal supervised run

```bash
python3 run.py --out evidence/local/two-gpu-small \
  --sizes 4112 262144 --work 0 32 --iterations 10 --warmup 3 --repeats 3
```

The repository intentionally does not claim this recipe has passed on an unobserved target host. The final qualification remains incomplete until the device-channel correctness, timeout behavior, profiles and region analysis are all backed by target-host evidence.
