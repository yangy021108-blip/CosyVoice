#!/usr/bin/env python3
"""Persistent-worker PD interference benchmark with event-level boundaries."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def wait_paths(paths,procs,timeout):
    deadline=time.monotonic()+timeout
    while not all(p.exists() for p in paths):
        for proc in procs:
            if proc.poll() not in (None,0):
                raise RuntimeError("worker exited early: %s" % proc.returncode)
        if time.monotonic()>=deadline: raise TimeoutError("timed out waiting for files")
        time.sleep(.1)

def pct(vals,q):
    if not vals: return None
    x=sorted(vals); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1)
    return x[lo]*(hi-pos)+x[hi]*(pos-lo)

def summary(vals):
    return {"count":len(vals),"mean":(sum(vals)/len(vals) if vals else None),
            "p50":pct(vals,.5),"p95":pct(vals,.95),"p99":pct(vals,.99),
            "max":max(vals,default=None)}

def trace_records(path, prefix):
    out=[]
    if not path.exists(): return out
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try:
            x=json.loads(line); rid=str(x.get("request_id",""))
            if rid==prefix or rid.startswith(prefix+"-"): out.append(x)
        except Exception: pass
    return out

def make_phase(metrics,started,bstart,bend):
    ts=[int(started+float(x)*1e9) for x in metrics.get("token_timestamps_seconds",[])]
    vals=[float(x) for x in metrics.get("token_intervals_seconds",[])]
    out={"before":[],"during":[],"after":[]}
    for i,v in enumerate(vals,1):
        end=ts[i] if i<len(ts) else None
        if end is None or bstart is None or end<bstart: out["before"].append(v)
        elif bend is not None and end>bend: out["after"].append(v)
        else: out["during"].append(v)
    return {k:summary(v) for k,v in out.items()}

def run_length(a,length):
    work=a.results_root/"pd"/("len_%d"%length)
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    base_socket="/tmp/cosy_pd_%s_%d"%(os.getpid(),length)
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"third_party/Matcha-TTS"),env.get("PYTHONPATH","")])
    env["VLLM_USE_FLASHINFER_SAMPLER"]="0"; env["COSY_PD_CONNECTOR_MODULE"]="pd_exp.prompt_nixl_connector_0251"
    env.setdefault("UCX_TLS","tcp,cuda_ipc,cuda_copy,sm,self"); env.setdefault("UCX_NET_DEVICES","lo")
    p_cmd=[a.python,str(ROOT/"pd_exp/pd_interference_prefill_persistent.py"),
        "--gpu",str(a.p_gpu),"--model-dir",str(a.model_dir),"--input",str(a.prompt_payload),
        "--work",str(work),"--b-prompt-length",str(length),"--repeats",str(a.repeats),
        "--barrier-socket",base_socket,"--side-channel-port",str(a.port),"--timeout",str(a.timeout),"--standard-optimized"]
    d_cmd=[a.python,str(ROOT/"pd_exp/pd_interference_decode_persistent.py"),
        "--gpu",str(a.d_gpu),"--model-dir",str(a.model_dir),"--input",str(a.prompt_payload),
        "--work",str(work),"--repeats",str(a.repeats),"--barrier-socket",base_socket,
        "--side-channel-port",str(a.port+1),"--timeout",str(a.timeout),"--standard-optimized","--graph-safe-remote-prefill"]
    if a.instrument_nixl: p_cmd.append("--instrument-nixl"); d_cmd.append("--instrument-nixl")
    plog=(work/"prefill.log").open("w",encoding="utf-8"); dlog=(work/"decode.log").open("w",encoding="utf-8")
    pp=dd=None
    try:
        dd=subprocess.Popen(d_cmd,env=env,stdout=dlog,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        pp=subprocess.Popen(p_cmd,env=env,stdout=plog,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        procs=[pp,dd]; files=[work/"p_ready.json",work/"d_ready.json",work/"warm_prefill.json",work/"warm_decode.json"]
        for r in range(a.repeats):
            tag="r%02d"%r
            files += [work/f"control_{tag}_prefill.json",work/f"control_{tag}_decode.json",
                      work/f"measured_{tag}_prefill.json",work/f"measured_{tag}_decode.json",
                      work/f"barrier_{tag}.json",work/f"b_{tag}_prefill.json"]
        wait_paths(files,procs,a.timeout)
        for proc in procs: proc.wait(timeout=90)
        baseline=json.loads(a.baseline_tokens.read_text())
        records=[]
        for r in range(a.repeats):
            tag="r%02d"%r
            c=json.loads((work/f"control_{tag}_decode.json").read_text())
            m=json.loads((work/f"measured_{tag}_decode.json").read_text())
            b=json.loads((work/f"b_{tag}_prefill.json").read_text())
            barrier=json.loads((work/f"barrier_{tag}.json").read_text())
            if c.get("status")!="ok" or m.get("status")!="ok" or b.get("status")!="ok": raise RuntimeError("bad worker result")
            phases=make_phase(m["metrics"],int(m["started_monotonic_ns"]),
                              int(b["started_monotonic_ns"]),int(b["finished_monotonic_ns"]))
            tr=trace_records(work/"decode_trace.jsonl","pd-d-measured-"+tag)
            done=[x for x in tr if x.get("event")=="kv_transfer_complete" and x.get("duration_ns") is not None]
            records.append({"repeat":r,"b_prompt_length":length,
                "a":{"raw_speech_tokens":m["raw_speech_tokens"],"token_count":len(m["raw_speech_tokens"]),
                     "correctness":{"token_count":len(m["raw_speech_tokens"]),"baseline_token_count":len(baseline["raw_speech_tokens"])},
                     "metrics":m["metrics"],"phase_tpot":phases,
                     "started_monotonic_ns":m["started_monotonic_ns"],
                     "b_started_monotonic_ns":b["started_monotonic_ns"],
                     "b_finished_monotonic_ns":b["finished_monotonic_ns"]},
                "control":{"metrics":c["metrics"],"token_count":len(c["raw_speech_tokens"]),
                          "tpot":summary([float(x) for x in c["metrics"].get("token_intervals_seconds",[])])},
                "b":{"prefill_seconds":b["metrics"].get("total_seconds"),"gpu_utilization":b.get("gpu_utilization"),"driver_timing":b["metrics"].get("driver_timing"),"engine_metrics":b["metrics"].get("engine_metrics")},
                "kv_transfer":{"latency_seconds":(done[0]["duration_ns"]/1e9 if done else None),"events":done},
                "barrier":{"d_token40_monotonic_ns":barrier.get("d_token40_monotonic_ns"),
                           "p_received_monotonic_ns":barrier.get("p_received_monotonic_ns")}})
        return {"mode":"pd","engine_mode":"FULL_DECODE_ONLY","length":length,"repeats":a.repeats,
                "p_gpu":str(a.p_gpu),"d_gpu":str(a.d_gpu),"records":records}
    finally:
        for proc in (pp,dd):
            if proc is not None and proc.poll() is None: proc.terminate()
        for proc in (pp,dd):
            if proc is not None:
                try: proc.wait(timeout=15)
                except subprocess.TimeoutExpired: proc.kill()
        for proc in (pp, dd):
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        plog.close(); dlog.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--python",default=sys.executable); p.add_argument("--p-gpu",default="0"); p.add_argument("--d-gpu",default="7")
    p.add_argument("--model-dir",type=Path,required=True); p.add_argument("--prompt-payload",type=Path,required=True)
    p.add_argument("--baseline-tokens",type=Path,required=True); p.add_argument("--results-root",type=Path,required=True)
    p.add_argument("--lengths",nargs="+",type=int,default=[512,1024,2048,4096,8192]); p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--timeout",type=float,default=1800); p.add_argument("--port",type=int,default=6100); p.add_argument("--instrument-nixl",action="store_true")
    a=p.parse_args(); a.results_root.mkdir(parents=True,exist_ok=True); allrec=[]
    for i,L in enumerate(a.lengths):
        a.port += i*2
        result=run_length(a,L); allrec.append(result)
        (a.results_root/f"pd_len_{L}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({"length":L,"repeats":a.repeats}),flush=True)
    (a.results_root/"pd_interference_summary.json").write_text(json.dumps({"mode":"pd","engine_mode":"FULL_DECODE_ONLY","lengths":a.lengths,"repeats":a.repeats,"records":allrec},ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__": raise SystemExit(main())
