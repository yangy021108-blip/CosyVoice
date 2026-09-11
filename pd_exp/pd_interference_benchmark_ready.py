#!/usr/bin/env python3
"""Orchestrate ready 1P1D PD interference cases."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def wait_paths(paths,procs,timeout):
    deadline=time.monotonic()+timeout
    while True:
        if all(p.exists() for p in paths): return
        for proc in procs:
            if proc.poll() not in (None,0):
                raise RuntimeError("worker exited early: %s" % proc.returncode)
        if time.monotonic()>=deadline: raise TimeoutError("timed out waiting for %s" % paths)
        time.sleep(.2)

def pct(vals,q):
    if not vals: return None
    a=sorted(vals); pos=(len(a)-1)*q; lo=int(pos); hi=min(lo+1,len(a)-1)
    return a[lo]*(hi-pos)+a[hi]*(pos-lo)

def summary(vals):
    return {"count":len(vals),"mean":(sum(vals)/len(vals) if vals else None),
            "p50":pct(vals,.5),"p95":pct(vals,.95),"p99":pct(vals,.99),
            "max":max(vals,default=None)}

def read_trace(path,request_id):
    if not path.exists(): return []
    out=[]
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try:
            value=json.loads(line)
            if value.get("request_id")==request_id or str(value.get("request_id", "")).startswith(request_id+"-"): out.append(value)
        except Exception: pass
    return out

def phase_metrics(decode,work):
    metrics=decode["metrics"]; start=int(decode["started_monotonic_ns"])
    barrier=json.loads((work/"barrier.json").read_text()) if (work/"barrier.json").exists() else {}
    b=json.loads((work/"b_prefill.json").read_text()) if (work/"b_prefill.json").exists() else {}
    inject=barrier.get("monotonic_ns"); bstart=b.get("started_monotonic_ns"); bend=b.get("finished_monotonic_ns")
    times=[start+int(float(x)*1e9) for x in metrics.get("token_timestamps_seconds",[])]
    vals=[float(x) for x in metrics.get("token_intervals_seconds",[])]
    groups={"before":[],"during":[],"after":[]}
    for i,v in enumerate(vals,1):
        end=times[i] if i<len(times) else None
        if end is None or inject is None or end<inject: groups["before"].append(v)
        elif bend is not None and end>bend: groups["after"].append(v)
        else: groups["during"].append(v)
    kv=read_trace(work/"decode_trace.jsonl","pd-d-measured")
    complete=[x for x in kv if x.get("event")=="kv_transfer_complete" and x.get("duration_ns") is not None]
    return groups,summary,{"before":summary(groups["before"]),"during":summary(groups["during"]),"after":summary(groups["after"])},b,complete

def run_case(a,length,repeat,index):
    work=a.results_root/"pd"/("len_%d_rep_%02d"%(length,repeat))
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True)
    port=5900+index*2
    env=os.environ.copy()
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"third_party/Matcha-TTS"),env.get("PYTHONPATH","")])
    env["VLLM_USE_FLASHINFER_SAMPLER"]="0"
    env["COSY_PD_CONNECTOR_MODULE"]="pd_exp.prompt_nixl_connector_0251"
    env.setdefault("UCX_TLS","tcp,cuda_ipc,cuda_copy,sm,self")
    env.setdefault("UCX_NET_DEVICES","lo")
    common=[a.python,str(ROOT/"pd_exp/pd_interference_prefill_ready.py")]
    p_cmd=common+["--gpu",str(a.p_gpu),"--model-dir",str(a.model_dir),
        "--input",str(a.prompt_payload),"--work",str(work),
        "--b-prompt-length",str(length),"--side-channel-port",str(port),
        "--timeout",str(a.timeout),"--standard-optimized"]
    d_cmd=[a.python,str(ROOT/"pd_exp/pd_interference_decode_ready.py"),
        "--gpu",str(a.d_gpu),"--model-dir",str(a.model_dir),
        "--input",str(a.prompt_payload),"--work",str(work),
        "--side-channel-port",str(port+1),"--timeout",str(a.timeout),
        "--standard-optimized","--graph-safe-remote-prefill"]
    if a.instrument_nixl: p_cmd.append("--instrument-nixl"); d_cmd.append("--instrument-nixl")
    plog=(work/"prefill.log").open("w",encoding="utf-8"); dlog=(work/"decode.log").open("w",encoding="utf-8")
    pproc=dproc=None
    try:
        dproc=subprocess.Popen(d_cmd,env=env,stdout=dlog,stderr=subprocess.STDOUT,text=True)
        pproc=subprocess.Popen(p_cmd,env=env,stdout=plog,stderr=subprocess.STDOUT,text=True)
        procs=[pproc,dproc]
        wait_paths([work/"p_ready.json",work/"d_ready.json"],procs,a.timeout)
        wait_paths([work/"warm_prefill.json",work/"warm_decode.json",
                    work/"control_prefill.json",work/"control_decode.json",
                    work/"measured_prefill.json",work/"measured_decode.json",
                    work/"barrier.json",work/"b_prefill.json"],procs,a.timeout)
        for proc in procs: proc.wait(timeout=60)
        measured=json.loads((work/"measured_decode.json").read_text())
        control=json.loads((work/"control_decode.json").read_text())
        b=json.loads((work/"b_prefill.json").read_text())
        if measured.get("status")!="ok" or control.get("status")!="ok" or b.get("status")!="ok":
            raise RuntimeError("invalid worker result")
        _,_,phases,b,complete=phase_metrics(measured,work)
        from pd_exp.compare_tokens import compare
        baseline=json.loads(a.baseline_tokens.read_text())
        rec={"mode":"pd","engine_mode":"FULL_DECODE_ONLY","p_gpu":str(a.p_gpu),"d_gpu":str(a.d_gpu),
             "b_prompt_length":length,"repeat":repeat,"inject_after_tokens":40,
             "a":{"raw_speech_tokens":measured["raw_speech_tokens"],
                   "token_count":len(measured["raw_speech_tokens"]),
                   "correctness":compare(baseline["raw_speech_tokens"],measured["raw_speech_tokens"]),
                   "metrics":measured["metrics"],"started_monotonic_ns":measured["started_monotonic_ns"],
                   "phase_tpot_seconds":phases},
             "control":{"raw_speech_tokens":control["raw_speech_tokens"],
                       "token_count":len(control["raw_speech_tokens"]),
                       "correctness":compare(baseline["raw_speech_tokens"],control["raw_speech_tokens"]),
                       "metrics":control["metrics"],
                       "tpot_summary":summary([float(x) for x in control["metrics"].get("token_intervals_seconds",[])])},
             "b":{"prompt_tokens":length,"prefill_seconds":b.get("metrics",{}).get("total_seconds"),
                  "gpu_utilization":b.get("gpu_utilization"),"started_monotonic_ns":b.get("started_monotonic_ns"),
                  "finished_monotonic_ns":b.get("finished_monotonic_ns")},
             "kv_transfer":{"events":complete,
                           "latency_seconds":(complete[0]["duration_ns"]/1e9 if complete else None)}}
        (work/"summary.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2)+"\n")
        return rec
    finally:
        for proc in (pproc,dproc):
            if proc is not None and proc.poll() is None: proc.terminate()
        for proc in (pproc,dproc):
            if proc is not None:
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill()
        plog.close(); dlog.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--python",default=sys.executable); p.add_argument("--p-gpu",default="0"); p.add_argument("--d-gpu",default="7")
    p.add_argument("--model-dir",type=Path,required=True); p.add_argument("--prompt-payload",type=Path,required=True)
    p.add_argument("--baseline-tokens",type=Path,required=True); p.add_argument("--results-root",type=Path,required=True)
    p.add_argument("--lengths",nargs="+",type=int,default=[512,1024,2048,4096,8192]); p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--timeout",type=float,default=900); p.add_argument("--instrument-nixl",action="store_true")
    a=p.parse_args(); a.results_root.mkdir(parents=True,exist_ok=True)
    records=[]; started=time.perf_counter()
    for i,(length,repeat) in enumerate(( (l,r) for l in a.lengths for r in range(a.repeats))):
        rec=run_case(a,length,repeat,i); records.append(rec)
        print(json.dumps({"length":length,"repeat":repeat,
            "control_tpot_ms":(rec["control"]["tpot_summary"]["mean"] or 0)*1000,
            "before_p99_ms":(rec["a"]["phase_tpot_seconds"]["before"]["p99"] or 0)*1000,
            "during_p99_ms":(rec["a"]["phase_tpot_seconds"]["during"]["p99"] or 0)*1000,
            "during_max_ms":(rec["a"]["phase_tpot_seconds"]["during"]["max"] or 0)*1000,
            "b_prefill_ms":(rec["b"]["prefill_seconds"] or 0)*1000,
            "kv_ms":(rec["kv_transfer"]["latency_seconds"] or 0)*1000}, ensure_ascii=False), flush=True)
        (a.results_root/"pd_interference_records.json").write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
    (a.results_root/"pd_interference_summary.json").write_text(json.dumps({"mode":"pd","engine_mode":"FULL_DECODE_ONLY",
        "lengths":a.lengths,"repeats":a.repeats,"p_gpu":a.p_gpu,"d_gpu":a.d_gpu,
        "script_seconds":time.perf_counter()-started,"records":records},ensure_ascii=False,indent=2)+"\n")
if __name__=="__main__": raise SystemExit(main())
