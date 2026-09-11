#!/usr/bin/env python3
"""Unified FULL_DECODE_ONLY interference benchmark."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def pct(values,q):
    if not values: return None
    vals=sorted(values); pos=(len(vals)-1)*q; lo=int(pos); hi=min(lo+1,len(vals)-1)
    return vals[lo]*(hi-pos)+vals[hi]*(pos-lo)

class GpuSampler:
    def __init__(self,gpu):
        self.gpu=str(gpu); self.samples=[]; self.stop_evt=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        while not self.stop_evt.is_set():
            now=time.perf_counter_ns()
            try:
                out=subprocess.check_output(["nvidia-smi","-i",self.gpu,
                    "--query-gpu=utilization.gpu,memory.used","--format=csv,noheader,nounits"],
                    text=True,stderr=subprocess.DEVNULL,timeout=2)
                f=[x.strip() for x in out.strip().split(",")]
                self.samples.append({"monotonic_ns":now,"utilization_gpu_percent":float(f[0]),
                                     "memory_used_mib":float(f[1])})
            except Exception: pass
            self.stop_evt.wait(0.1)
    def start(self): self.thread.start()
    def stop(self): self.stop_evt.set(); self.thread.join(timeout=3)
    def summary(self,start,end):
        s=[x for x in self.samples if start<=x["monotonic_ns"]<=end]
        v=[x["utilization_gpu_percent"] for x in s]
        return {"sample_count":len(s),"mean_percent":(sum(v)/len(v) if v else None),
                "max_percent":(max(v) if v else None),"samples":s}

def exact_prompt(prompt,n):
    return prompt.repeat(((n+prompt.shape[0]-1)//prompt.shape[0],1))[:n]

def run_case(engine,prompt,metadata,baseline,length,repeat,timeout,inject_after):
    from pd_exp.common import make_sampling_params,run_engine_request
    from pd_exp.compare_tokens import compare
    control_id=f"unified-control-{length}-{repeat}"
    control_tokens,control_metrics,_=run_engine_request(engine,prompt,
        make_sampling_params(metadata),request_id=control_id,timeout_seconds=timeout,
        nvtx_label="COSY_UNIFIED_CONTROL")
    a_id=f"unified-a-{length}-{repeat}"; b_id=f"unified-b-{length}-{repeat}"
    dev_prompt=prompt.to("cuda",dtype=__import__("torch").bfloat16)
    engine.add_request(a_id,{"prompt_embeds":dev_prompt},make_sampling_params(metadata))
    a_prev=0; a_tokens=[]; a_times=[]; b_tokens=[]; a_finished=False; b_finished=False
    injected=False; inject_ns=None; b_started_ns=None; b_finished_ns=None
    started_ns=time.perf_counter_ns(); deadline=time.monotonic()+timeout
    sampler=GpuSampler(os.environ["CUDA_VISIBLE_DEVICES"]); sampler.start()
    try:
        while time.monotonic()<deadline and not (a_finished and b_finished):
            outs=engine.step(); now=time.perf_counter_ns()
            for output in outs:
                if output.request_id==a_id:
                    vals=list(output.outputs[0].token_ids); added=len(vals)-a_prev
                    if added>0:
                        a_times.extend([now/1e9]*added); a_prev=len(vals); a_tokens=[int(x) for x in vals]
                    a_finished=bool(output.finished)
                elif output.request_id==b_id:
                    b_tokens=[int(x) for x in output.outputs[0].token_ids]
                    if output.finished and b_finished_ns is None: b_finished_ns=now
                    b_finished=bool(output.finished)
            if not injected and len(a_tokens)>=inject_after:
                b_started_ns=time.perf_counter_ns()
                engine.add_request(b_id,{"prompt_embeds":exact_prompt(prompt,length).to("cuda",dtype=__import__("torch").bfloat16)},
                    make_sampling_params(metadata,prefill_only=True))
                inject_ns=time.perf_counter_ns(); injected=True
            if not outs: time.sleep(0.001)
    finally:
        sampler.stop()
    if not (a_finished and b_finished): raise RuntimeError("unified case timed out")
    stop_ids=set(int(x) for x in metadata["stop_token_ids"])
    stop_index=next((i for i,x in enumerate(a_tokens) if x in stop_ids),None)
    if stop_index is not None: a_tokens=a_tokens[:stop_index]; a_times=a_times[:stop_index]
    abs_ns=[int(x*1e9) for x in a_times]
    intervals=[(abs_ns[i]-abs_ns[i-1])/1e9 for i in range(1,len(abs_ns))]
    pre=[]; during=[]; after=[]
    for i,val in enumerate(intervals,1):
        end=abs_ns[i]
        if inject_ns is None or end<inject_ns: pre.append(val)
        elif b_finished_ns is not None and end>b_finished_ns: after.append(val)
        else: during.append(val)
    b_end=b_finished_ns or inject_ns
    b_seconds=None if b_started_ns is None or b_end is None else (b_end-b_started_ns)/1e9
    return {"mode":"unified","engine_mode":"FULL_DECODE_ONLY","gpu":os.environ["CUDA_VISIBLE_DEVICES"],
      "b_prompt_length":length,"repeat":repeat,"inject_after_tokens":inject_after,
      "a":{"raw_speech_tokens":a_tokens,"token_count":len(a_tokens),
           "correctness":compare(baseline["raw_speech_tokens"],a_tokens),
           "started_monotonic_ns":started_ns,"inject_monotonic_ns":inject_ns,
           "b_finished_monotonic_ns":b_finished_ns,
           "tpot_seconds":{"pre":pre,"during":during,"after":after,
             "pre_summary":{"mean":(sum(pre)/len(pre) if pre else None),"p50":pct(pre,.5),"p95":pct(pre,.95),"p99":pct(pre,.99),"max":max(pre,default=None)},
             "during_summary":{"mean":(sum(during)/len(during) if during else None),"p50":pct(during,.5),"p95":pct(during,.95),"p99":pct(during,.99),"max":max(during,default=None)},
             "after_summary":{"mean":(sum(after)/len(after) if after else None),"p50":pct(after,.5),"p95":pct(after,.95),"p99":pct(after,.99),"max":max(after,default=None)}}},
      "control":{"raw_speech_tokens":control_tokens,"token_count":len(control_tokens),
                 "correctness":compare(baseline["raw_speech_tokens"],control_tokens),
                 "metrics":control_metrics,
                 "tpot_summary":{"mean":control_metrics["mean_tpot_seconds"],
                    "p50":pct(control_metrics["token_intervals_seconds"],.5),
                    "p95":pct(control_metrics["token_intervals_seconds"],.95),
                    "p99":pct(control_metrics["token_intervals_seconds"],.99),
                    "max":max(control_metrics["token_intervals_seconds"],default=None)}},
      "b":{"token_count":len(b_tokens),"prefill_seconds":b_seconds,
           "gpu_utilization":sampler.summary(b_started_ns or started_ns,b_end or time.perf_counter_ns())}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--gpu",default="7"); p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--prompt-payload",type=Path,required=True); p.add_argument("--baseline-tokens",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--lengths",nargs="+",type=int,default=[512,1024,2048,4096,8192])
    p.add_argument("--repeats",type=int,default=3); p.add_argument("--timeout",type=float,default=600); p.add_argument("--inject-after",type=int,default=40)
    a=p.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    from pd_exp.common import build_engine,load_prompt_payload,read_json,write_json
    prompt,metadata=load_prompt_payload(a.prompt_payload); baseline=read_json(a.baseline_tokens)
    engine=build_engine(a.model_dir,eager=False,gpu_memory_utilization=.2,max_num_seqs=2)
    records=[]; started=time.perf_counter()
    try:
        for length in a.lengths:
            for repeat in range(a.repeats):
                rec=run_case(engine,prompt,metadata,baseline,length,repeat,a.timeout,a.inject_after)
                records.append(rec); write_json(a.output,{"status":"running","records":records})
                print(json.dumps({"length":length,"repeat":repeat,
                    "control_tpot_ms":(rec["control"]["tpot_summary"]["mean"] or 0)*1000,
                    "during_p99_ms":(rec["a"]["tpot_seconds"]["during_summary"]["p99"] or 0)*1000,
                    "during_max_ms":(rec["a"]["tpot_seconds"]["during_summary"]["max"] or 0)*1000,
                    "b_prefill_s":rec["b"]["prefill_seconds"]}),flush=True)
    finally:
        write_json(a.output,{"status":"ok","mode":"unified","engine_mode":"FULL_DECODE_ONLY","gpu":a.gpu,
            "lengths":a.lengths,"repeats":a.repeats,"script_seconds":time.perf_counter()-started,"records":records})
if __name__=="__main__": raise SystemExit(main())
