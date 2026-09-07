# Channel and buffer contract

The implementation adapts the setup pattern from [MSCCL++ tutorial 04 at the pinned revision](https://github.com/microsoft/mscclpp/blob/de2e486304276b59c9ab4e23e4e2c911eca6b1cf/examples/tutorials/04-port-channel/bidir_port_channel.cu). It uses an actual `PortChannel`, not a CPU replacement for the device path.

## Data and resource layout

Each rank owns one registered allocation with two regions: local outbound data `[0,N)` and inbound data `[N,2N)`. A rank writes only its outbound region and sends it to the peer's inbound region. Registration and channel objects outlive all kernels and proxy activity. The proxy stops before buffers are released.

Every generation has a distinct initial value determined by index, sequence and source rank. The GPU applies an unsigned 32-bit affine recurrence. The host reference composes that transform in logarithmic time; unsigned overflow has defined modulo-2³² semantics. Full-element comparison detects stale or partial payloads, while checksums make raw records easy to cross-reference.

## Happens-before chain

```mermaid
sequenceDiagram
  participant G as Producer GPU
  participant P as Host proxy / CudaIpc copy
  participant R as Peer GPU
  participant H as Consumer host
  G->>G: Compute chunk; every writer system-fences
  G->>G: CTA barrier
  G->>P: put(chunk)
  G->>G: Compute next immutable source chunk
  G->>P: signal after the final put
  G->>P: flush
  P->>R: Complete ordered copies and publish signal
  R->>R: wait acquires publication
  R->>H: Kernel/event completion
  H->>H: Copy inbound payload; validate every element
  H->>G: Bootstrap barrier acknowledges consumption
  G->>G: Next generation may reuse regions
```

`put` enqueues a transfer; return does not mean the DMA engine has released the source. `signal` publishes through the ordered connection after prior transfers. `wait` consumes peer publication. `flush` establishes local completion. The consumer acknowledgement is separately required to prevent a future transfer overwriting a destination that is still being checked.

The release edge from all GPU writers to the thread that enqueues work is explicit: each writer calls `__threadfence_system`, followed by `__syncthreads`. One CTA avoids a non-cooperative grid-wide barrier that could deadlock under limited occupancy. No detached compute stream or speculative buffer reuse is used.

These claims are based on the pinned [PortChannel device interface](https://github.com/microsoft/mscclpp/blob/de2e486304276b59c9ab4e23e4e2c911eca6b1cf/include/mscclpp/port_channel_device.hpp) and require target-device validation. They are not a new portability guarantee for arbitrary transports.

## The finite model

`model.py` enumerates logical compute, enqueue, copy-completion, signal, flush and consume/ack events for a reusable slot. The correct 3-generation model reaches 25 states and one terminal state. Removing consumer acknowledgement allows generation 2 to overwrite generation 1 before consumption. Allowing signal before copy completion permits a read of generation 0.

This is a counterexample and protocol-ordering aid. It abstracts away CUDA caches, memory scopes, proxy implementation details and hardware scheduling. Passing it cannot prove the CUDA channel is race-free.

## Measurement definition

CUDA events bracket the full exchange kernel, including local completion and waiting for the peer. Host monotonic timing brackets launch through event completion. The per-process CPU interval includes proxy activity; it is CPU time, not an instantaneous occupancy measurement. Full host validation and the between-iteration acknowledgement are outside the measured kernel interval in all configurations.

The runner pairs ranks by sequence and uses the slower rank for each critical-path sample. It alternates serial/chunk case order across repeats. Region labels describe observed paired ratios; a profile is required to verify actual concurrent copy/compute activity, and wider uncertainty analysis is required before a performance claim.
