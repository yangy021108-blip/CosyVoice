#!/usr/bin/env python3
"""Persistent PD prefill worker for low-latency interference measurements."""
from __future__ import annotations
import argparse
import json
import os
import pynvml
import socket
import threading
import time
import traceback
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",required=True); p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True); p.add_argument("--work",type=Path,required=True)
    p.add_argument("--b-prompt-length",type=int,required=True); p.add_argument("--repeats",type=int,default=30)
    p.add_argument("--barrier-socket",required=True)
    p.add_argument("--side-channel-port",type=int,required=True); p.add_argument("--timeout",type=float,default=900)
    p.add_argument("--instrument-nixl",action="store_true"); p.add_argument("--standard-optimized",action="store_true")
    return p.parse_args()

class GpuSampler:
    def __init__(self,gpu):
        self.gpu=str(gpu); self.samples=[]; self.stop_event=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
    def _run(self):
        handle=None
        try:
            pynvml.nvmlInit(); handle=pynvml.nvmlDeviceGetHandleByIndex(int(self.gpu))
        except Exception: handle=None
        while not self.stop_event.is_set():
            now=time.perf_counter_ns()
            if handle is not None:
                try:
                    util=pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem=pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.samples.append({"monotonic_ns":now,"utilization_gpu_percent":float(util.gpu),
                                         "memory_used_mib":float(mem.used)/(1024*1024)})
                except Exception: pass
            self.stop_event.wait(0.02)
        if handle is not None:
            try: pynvml.nvmlShutdown()
            except Exception: pass
    def start(self): self.thread.start()
    def stop(self): self.stop_event.set(); self.thread.join(timeout=3)
    def summary(self,start,end):
        s=[x for x in self.samples if start<=x["monotonic_ns"]<=end]
        v=[x["utilization_gpu_percent"] for x in s]
        return {"sample_count":len(s),"mean_percent":(sum(v)/len(v) if v else None),
                "max_percent":(max(v) if v else None),"samples":s}

def main():
    a=parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"]="127.0.0.1"; os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"]=str(a.side_channel_port)
    os.environ.setdefault("UCX_TLS","tcp,cuda_ipc,cuda_copy,sm,self"); os.environ.setdefault("UCX_NET_DEVICES","lo")
    os.environ["COSY_PD_CONNECTOR_MODULE"]="pd_exp.prompt_nixl_connector_0251"
    if a.instrument_nixl:
        os.environ["COSY_PD_INSTRUMENT_NIXL"]="1"; os.environ["COSY_PD_TRACE_FILE"]=str(a.work/"prefill_trace.jsonl")
    from pd_exp.common import build_engine,load_prompt_payload,make_sampling_params,run_engine_request,wait_for_file,write_json
    engine=None
    try:
        prompt,metadata=load_prompt_payload(a.input)
        engine=build_engine(a.model_dir,kv_role="kv_producer",eager=not a.standard_optimized,
                            gpu_memory_utilization=.2,max_num_seqs=1)
        write_json(a.work/"p_ready.json",{"status":"ready","pid":os.getpid(),"gpu":a.gpu,
                   "graph_mode":"FULL_DECODE_ONLY" if a.standard_optimized else "eager",
                   "ready_monotonic_ns":time.perf_counter_ns(),"repeats":a.repeats})
        def run_prefill(name,req_id,p):
            params={"do_remote_decode":True,"do_remote_prefill":False,"remote_engine_id":None,
                    "remote_block_ids":None,"remote_host":None,"remote_port":None}
            started=time.perf_counter_ns()
            tokens,metrics,transfer=run_engine_request(
                engine,p,make_sampling_params(metadata,kv_transfer_params=params,prefill_only=True),
                request_id=req_id,timeout_seconds=a.timeout,nvtx_label="COSY_PD_"+name.upper())
            finished=time.perf_counter_ns()
            if not transfer: raise RuntimeError("missing KV transfer metadata")
            write_json(a.work/(name+"_prefill.json"),{"status":"ok","request_id":req_id,
                "started_monotonic_ns":started,"finished_monotonic_ns":finished,
                "prompt_tokens":int(p.shape[0]),"auxiliary_tokens":tokens,"metrics":metrics,
                "kv_transfer_params":transfer})
        run_prefill("warm","pd-p-warm",prompt); wait_for_file(a.work/"warm_release.json",a.timeout)
        for r in range(a.repeats):
            tag=f"r{r:02d}"
            run_prefill("control_"+tag,"pd-p-control-"+tag,prompt)
            wait_for_file(a.work/f"control_{tag}_release.json",a.timeout)
            sock_path=Path(a.barrier_socket+"_"+tag+".sock")
            try: sock_path.unlink()
            except FileNotFoundError: pass
            listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            print("BARRIER_SOCKET",sock_path,flush=True)
            listener.bind(str(sock_path)); listener.listen(1); listener.settimeout(a.timeout)
            run_prefill("measured_"+tag,"pd-p-measured-"+tag,prompt)
            conn,_=listener.accept()
            chunks=[]
            while True:
                part=conn.recv(4096)
                if not part: break
                chunks.append(part)
            conn.close(); listener.close()
            event=json.loads(b"".join(chunks).decode("utf-8"))
            bprompt=prompt.repeat(((a.b_prompt_length+prompt.shape[0]-1)//prompt.shape[0],1))[:a.b_prompt_length]
            os.environ["COSY_PD_TIMING_STEPS"] = "1"
            sampler=GpuSampler(a.gpu); sampler.start()
            b_started=time.perf_counter_ns()
            btokens,bmetrics,_=run_engine_request(
                engine,bprompt,make_sampling_params(metadata,prefill_only=True),request_id="pd-p-b-"+tag,
                timeout_seconds=a.timeout,nvtx_label="COSY_PD_INTERFERENCE_B_PREFILL")
            b_finished=time.perf_counter_ns(); sampler.stop()
            write_json(a.work/f"barrier_{tag}.json",{"status":"ok","d_event":event,
                "d_token40_monotonic_ns":event.get("monotonic_ns"),
                "p_received_monotonic_ns":time.perf_counter_ns(),
                "b_started_monotonic_ns":b_started,"b_finished_monotonic_ns":b_finished})
            write_json(a.work/f"b_{tag}_prefill.json",{"status":"ok","request_id":"pd-p-b-"+tag,
                "prompt_tokens":a.b_prompt_length,"started_monotonic_ns":b_started,
                "finished_monotonic_ns":b_finished,"auxiliary_tokens":btokens,"metrics":bmetrics,
                "gpu_utilization":sampler.summary(b_started,b_finished)})
            wait_for_file(a.work/f"measured_{tag}_release.json",a.timeout)
        return 0
    except Exception as exc:
        err={"status":"error","stage":"prefill","exception":f"{type(exc).__name__}: {exc}",
             "traceback":traceback.format_exc()}
        for path in [a.work/"warm_prefill.json"]+[a.work/f"{n}_{r:02d}_prefill.json" for r in range(a.repeats) for n in ("control","measured","b")]:
            if not path.exists(): write_json(path,err)
        return 1
    finally:
        if engine is not None and callable(getattr(engine,"shutdown",None)): engine.shutdown()
if __name__=="__main__": raise SystemExit(main())
