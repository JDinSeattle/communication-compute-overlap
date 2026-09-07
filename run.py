#!/usr/bin/env python3
"""Run a bounded, correctness-gated two-GPU PortChannel experiment."""
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import time


def save(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def capture(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return {"command": cmd, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": cmd, "returncode": None, "error": str(e), "stdout": ""}


def terminate(proc):
    if proc.poll() is None:
        with contextlib.suppress(ProcessLookupError): os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def validate_samples(logs, iterations, warmup, mode, size, chunk, work):
    ranks = []
    for rank, text in enumerate(logs):
        entries = [json.loads(s) for s in text.splitlines() if s.startswith("{")]
        samples = [r for r in entries if r.get("kind") == "sample"]
        finals = [r for r in entries if r.get("kind") == "result"]
        if len(finals) != 1 or finals[0].get("status") != "pass" or finals[0].get("rank") != rank:
            raise ValueError("missing successful rank result")
        if finals[0].get("transport") != "CudaIpc" or finals[0].get("channel") != "PortChannel":
            raise ValueError("unrecognized transport/channel")
        if len(samples) != iterations or [s.get("sequence") for s in samples] != list(range(warmup+1,warmup+iterations+1)):
            raise ValueError("missing, duplicate, or reordered sequence")
        for s in samples:
            if any(s.get(k) != v for k,v in {"rank":rank,"mode":mode,"bytes":size,"chunk_bytes":chunk,"work":work,"correct":True}.items()):
                raise ValueError("sample metadata/correctness mismatch")
            for key in ("wall_us", "gpu_us", "process_cpu_us"):
                if not isinstance(s.get(key),(int,float)) or not math.isfinite(s[key]) or s[key]<0:
                    raise ValueError("invalid timing")
            if s["wall_us"] <= 0 or s["gpu_us"] <= 0:
                raise ValueError("zero duration")
        ranks.append(samples)
    # The critical path is the slower rank, not an average of independently completing ranks.
    return [max(a["wall_us"],b["wall_us"]) for a,b in zip(*ranks)]


def pair(binary, out, size, chunk, work, mode, iterations, warmup, timeout):
    out.mkdir()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1",0)); port=listener.getsockname()[1]
    commands=[];procs=[];streams=[];status="pass"
    try:
        for rank in range(2):
            cmd=[str(binary),str(rank),str(rank),f"lo:127.0.0.1:{port}",str(size),str(chunk),str(work),str(iterations),str(warmup),mode]
            commands.append(cmd)
            out_file=(out/f"rank{rank}.stdout").open("w");err_file=(out/f"rank{rank}.stderr").open("w")
            streams.extend([out_file,err_file])
            env={k:os.environ[k] for k in ("PATH","LD_LIBRARY_PATH","CUDA_VISIBLE_DEVICES") if k in os.environ}
            procs.append(subprocess.Popen(cmd,stdout=out_file,stderr=err_file,env=env,start_new_session=True))
        deadline=time.monotonic()+timeout
        while any(p.poll() is None for p in procs):
            if any(p.returncode not in (None,0) for p in procs): status="runtime_failure";break
            if time.monotonic()>=deadline:status="timeout";break
            time.sleep(.02)
    finally:
        for proc in procs:terminate(proc)
        for stream in streams:stream.close()
    samples=[];error=None
    if any(p.returncode!=0 for p in procs):status="runtime_failure" if status=="pass" else status
    if status=="pass":
        try: samples=validate_samples([(out/f"rank{r}.stdout").read_text() for r in range(2)],iterations,warmup,mode,size,chunk,work)
        except (ValueError,KeyError,TypeError) as e:status="invalid_evidence";error=str(e)
    result={"status":status,"error":error,"commands":commands,"returncodes":[p.returncode for p in procs],
            "bytes":size,"chunk_bytes":chunk,"work":work,"mode":mode,"critical_path_us":samples,
            "median_us":statistics.median(samples) if samples else None,
            "sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("rank*")}}
    save(out/"result.json",result);return result


def region(speedup, margin=.05):
    if not math.isfinite(speedup) or speedup <= 0: raise ValueError("invalid speedup")
    return "beneficial" if speedup>1+margin else "regressed" if speedup<1-margin else "no_material_gain"


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,default=Path("build/overlap"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sizes",nargs="+",type=int,default=[4096,262144,4194304])
    p.add_argument("--work",nargs="+",type=int,default=[0,32,256])
    p.add_argument("--iterations",type=int,default=20)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--timeout",type=int,default=60)
    a=p.parse_args()
    if not all(16<=n<=128*1024*1024 and n%16==0 for n in a.sizes) or not all(0<=n<=4096 for n in a.work): p.error("invalid size or work")
    if not(1<=a.iterations<=10000 and 0<=a.warmup<=1000 and 1<=a.repeats<=100 and 1<=a.timeout<=3600): p.error("invalid experiment bounds")
    a.out.mkdir(parents=True,exist_ok=False);binary=a.binary.resolve()
    preflight=capture([str(binary),"--preflight"])
    manifest={"timestamp_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"preflight":preflight,
              "binary_sha256":hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None,
              "upstream":json.loads((Path(__file__).parent/"upstream.lock.json").read_text()),
              "topology":capture(["nvidia-smi","topo","-m"]),
              "gpus":capture(["nvidia-smi","--query-gpu=name,uuid,pci.bus_id,driver_version","--format=csv"]),
              "cpu_affinity":sorted(os.sched_getaffinity(0)),
              "profile":None,"profile_missing_reason":"collect with Nsight Systems on the target two-GPU host before performance sign-off"}
    save(a.out/"manifest.json",manifest)
    if preflight["returncode"]!=0:
        save(a.out/"summary.json",{"status":"blocked_hardware","qualification":False,"performance":[]})
        print("Blocked: two peer-accessible GPUs required. No overlap measurements recorded.");return 2
    records=[]
    for rep in range(a.repeats):
        cases=[("serial",262144),("overlap",16384),("overlap",262144)]
        if rep%2:cases.reverse()
        for size in a.sizes:
            for work in a.work:
                for mode,chunk in cases:
                    tag=f"r{rep}-n{size}-w{work}-{mode}-c{chunk}"
                    r=pair(binary,a.out/tag,size,chunk,work,mode,a.iterations,a.warmup,a.timeout)
                    r["repeat"]=rep;records.append(r);save(a.out/"records.json",records)
                    if r["status"]!="pass":
                        save(a.out/"summary.json",{"status":r["status"],"qualification":False,"performance":[]});return 1
    performance=[]
    for size in a.sizes:
        for work in a.work:
            group=[r for r in records if r["bytes"]==size and r["work"]==work]
            for chunk in (16384,262144):
                ratios=[]
                for rep in range(a.repeats):
                    baseline=next(r for r in group if r["mode"]=="serial" and r["repeat"]==rep)
                    candidate=next(r for r in group if r["mode"]=="overlap" and r["chunk_bytes"]==chunk and r["repeat"]==rep)
                    ratios.append(baseline["median_us"]/candidate["median_us"])
                speedup=statistics.median(ratios)
                performance.append({"bytes":size,"work":work,"chunk_bytes":chunk,"median_paired_speedup":speedup,
                                    "paired_ratios":ratios,"region":region(speedup),"classification_margin":.05})
    save(a.out/"summary.json",{"status":"pass","qualification":False,"performance":performance,
                                "remaining":"review target-host profiles, run compute-sanitizer, and inspect repeated-trial uncertainty before sign-off"})
    return 0


if __name__=="__main__":raise SystemExit(main())
