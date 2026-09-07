#!/usr/bin/env python3
"""Exhaustive logical protocol model. It does not model CUDA's memory system."""
from dataclasses import dataclass, replace
from collections import deque
import json


@dataclass(frozen=True)
class State:
    produced: int = 0
    enqueued: int = 0
    copied: int = 0
    signalled: int = 0
    consumed: int = 0
    flushed: int = 0
    source: int = 0
    destination: int = 0


def explore(iterations=3, require_ack=True, ordered_signal=True):
    initial = State()
    queue = deque([(initial, [])])
    seen = {initial}
    terminal = 0
    while queue:
        s, trace = queue.popleft()
        transitions = []
        if s.produced < iterations and s.enqueued == s.produced and s.flushed == s.produced and (not require_ack or s.consumed == s.produced):
            n = s.produced+1
            transitions.append((f"compute({n})", replace(s, produced=n, source=n)))
        if s.enqueued < s.produced:
            transitions.append((f"put({s.produced})", replace(s, enqueued=s.produced)))
        if s.copied < s.enqueued:
            n = s.copied+1
            if s.source != n:
                return {"safe": False, "states": len(seen), "witness": trace+[f"DMA({n}) reads overwritten source {s.source}"]}
            transitions.append((f"DMA-complete({n})", replace(s, copied=n, destination=n)))
        signal_limit = s.copied if ordered_signal else s.enqueued
        if s.signalled < signal_limit:
            transitions.append((f"signal({s.signalled+1})", replace(s, signalled=s.signalled+1)))
        if s.flushed < s.copied:
            transitions.append((f"flush({s.copied})", replace(s, flushed=s.copied)))
        if s.consumed < s.signalled:
            n = s.consumed+1
            if s.destination != n:
                return {"safe": False, "states": len(seen), "witness": trace+[f"consume({n}) sees generation {s.destination}"]}
            transitions.append((f"consume-and-ack({n})", replace(s, consumed=n)))
        if not transitions:
            if s.consumed != iterations or s.flushed != iterations:
                return {"safe": False, "states": len(seen), "witness": trace+["deadlock"]}
            terminal += 1
        for action, nxt in transitions:
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt,trace+[action]))
    return {"safe": True, "states": len(seen), "terminal_states": terminal, "witness": None}


if __name__ == "__main__":
    print(json.dumps({"scope": "finite abstract state exploration; no device memory-order proof",
                      "correct": explore(), "missing_consumer_ack": explore(require_ack=False),
                      "signal_before_completion": explore(ordered_signal=False)}, indent=2))
