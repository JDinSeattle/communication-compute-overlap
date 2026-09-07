#!/usr/bin/env python3
"""Build pinned MSCCL++ (CudaIpc only) and the PortChannel extension."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir",type=Path,default=Path(tempfile.gettempdir())/("overlap-lab-"+hashlib.sha256(str(ROOT).encode()).hexdigest()[:8]))
    p.add_argument("--arch",choices=["80","86","89","90","100","120"],default="89")
    p.add_argument("--numa-include",type=Path)
    p.add_argument("--numa-library",type=Path)
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args();work=a.work_dir.resolve();work.mkdir(parents=True,exist_ok=True)
    out=ROOT/"evidence/local";out.mkdir(parents=True,exist_ok=True)
    with (out/"build.log").open("w") as log:
        def run(cmd,cwd=work):
            cmd=list(map(str,cmd));log.write("$ "+repr(cmd)+"\n");log.flush()
            subprocess.run(cmd,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
        lock=json.loads((ROOT/"upstream.lock.json").read_text())
        for name,item in lock.items():
            src=work/name
            if not src.exists():
                run(["git","init",src]);run(["git","fetch","--depth=1",item["repository"],item["revision"]],src)
                run(["git","checkout","--detach","FETCH_HEAD"],src)
            actual=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
            if actual!=item["revision"]:raise RuntimeError("revision mismatch: "+name)
        prefix=work/"install"
        cmd=["cmake","-S",work/"mscclpp","-B",work/"build","-DCMAKE_BUILD_TYPE=Release",f"-DCMAKE_INSTALL_PREFIX={prefix}",
             "-DMSCCLPP_USE_CUDA=ON","-DMSCCLPP_BYPASS_GPU_CHECK=ON",f"-DMSCCLPP_GPU_ARCHS={a.arch}",
             "-DMSCCLPP_USE_IB=OFF","-DMSCCLPP_USE_GDRCOPY=OFF","-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF",
             "-DMSCCLPP_BUILD_EXT_NCCL=OFF","-DMSCCLPP_BUILD_EXT_COLLECTIVES=OFF"]
        if "json" in lock:cmd.append(f"-DFETCHCONTENT_SOURCE_DIR_JSON={work/'json'}")
        if a.numa_include:cmd.append(f"-DNUMA_INCLUDE_DIR={a.numa_include.resolve()}")
        if a.numa_library:cmd.append(f"-DNUMA_LIBRARIES={a.numa_library.resolve()}")
        run(cmd);run(["cmake","--build",work/"build",f"-j{a.jobs}"]);run(["cmake","--install",work/"build"])
        run(["cmake","-S",ROOT,"-B",ROOT/"build",f"-DMSCCLPP_ROOT={prefix}",f"-DCMAKE_CUDA_ARCHITECTURES={a.arch}","-DCMAKE_BUILD_TYPE=Release"])
        run(["cmake","--build",ROOT/"build",f"-j{a.jobs}"])
    (out/"build.json").write_text(json.dumps({"prefix":str(prefix),"arch":a.arch,"lock":lock},indent=2)+"\n")
    print(ROOT/"build/overlap")


if __name__=="__main__":main()
