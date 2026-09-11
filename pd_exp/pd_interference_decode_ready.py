#!/usr/bin/env python3
"""Ready PD decode worker for interference measurements."""
from __future__ import annotations
import argparse
import os
import time
import traceback
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--gpu",required=True); p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--input",type=Path,required=True); p.add_argument("--work",type=Path,required=True)
    p.add_argument("--side-channel-port",type=int,required=True); p.add_argument("--timeout",type=float,default=600)
    p.add_argument("--instrument-nixl",action="store_true"); p.add_argument("--standard-optimized",action="store_true")
    p.add_argument("--graph-safe-remote-prefill",action="store_true")
    return p.parse_args()

def main():
    a=parse_args(); a.work.mkdir(parents=True,exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"]=str(a.gpu)
    os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"]="127.0.0.1"
    os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"]=str(a.side_channel_port)
    os.environ.setdefault("UCX_TLS","tcp,cuda_ipc,cuda_copy,sm,self")
    os.environ.setdefault("UCX_NET_DEVICES","lo")
    os.environ["COSY_PD_CONNECTOR_MODULE"]="pd_exp.prompt_nixl_connector_0251"
    if a.instrument_nixl:
        os.environ["COSY_PD_INSTRUMENT_NIXL"]="1"
        os.environ["COSY_PD_TRACE_FILE"]=str(a.work/"decode_trace.jsonl")
    if a.graph_safe_remote_prefill:
        os.environ["COSY_PD_GRAPH_SAFE_REMOTE_PREFILL"]="1"
    from pd_exp.common import build_engine, load_prompt_payload, make_sampling_params, read_json, run_engine_request, run_engine_requests, wait_for_file, write_json
    engine=None
    try:
        prompt, metadata=load_prompt_payload(a.input)
        engine=build_engine(a.model_dir,kv_role="kv_consumer",eager=not a.standard_optimized,
                            gpu_memory_utilization=0.2,max_num_seqs=1)
        write_json(a.work/"d_ready.json",{"status":"ready","pid":os.getpid(),"gpu":a.gpu,
                     "graph_mode":"FULL_DECODE_ONLY" if a.standard_optimized else "eager",
                     "ready_monotonic_ns":time.perf_counter_ns()})
        def run_decode(name, req_id, prefill_name, measured=False):
            wait_for_file(a.work/f"{prefill_name}_prefill.json",a.timeout)
            pre=read_json(a.work/f"{prefill_name}_prefill.json")
            if pre.get("status")!="ok": raise RuntimeError(f"prefill failed: {pre}")
            sampling=make_sampling_params(metadata,kv_transfer_params=pre["kv_transfer_params"],prefill_only=False)
            started=time.perf_counter_ns()
            if measured:
                os.environ["COSY_PD_PROGRESS_FILE"]=str(a.work/"barrier.json")
                os.environ["COSY_PD_PROGRESS_REQUEST_ID"]=req_id
                os.environ["COSY_PD_PROGRESS_TOKENS"]="40"
                result=run_engine_requests(engine,[{"request_id":req_id,"prompt_embeds":prompt,
                                                    "sampling_params":sampling}],
                                            timeout_seconds=a.timeout,nvtx_label="COSY_PD_INTERFERENCE_A_DECODE")
                tokens,metrics,transfer=result[req_id]
            else:
                tokens,metrics,transfer=run_engine_request(engine,prompt,sampling,request_id=req_id,
                                                            timeout_seconds=a.timeout,nvtx_label="COSY_PD_INTERFERENCE_"+name.upper())
            finished=time.perf_counter_ns()
            write_json(a.work/f"{name}_decode.json",{"status":"ok","request_id":req_id,
                "started_monotonic_ns":started,"finished_monotonic_ns":finished,
                "raw_speech_tokens":tokens,"metrics":metrics,
                "kv_transfer_params":transfer})
            write_json(a.work/f"{name}_release.json",{"status":"decode_complete","finished_monotonic_ns":finished})
        run_decode("warm","pd-d-warm","warm",False)
        run_decode("control","pd-d-control","control",False)
        run_decode("measured","pd-d-measured","measured",True)
        return 0
    except Exception as exc:
        err={"status":"error","stage":"decode","exception":f"{type(exc).__name__}: {exc}",
             "traceback":traceback.format_exc()}
        for name in ("warm_decode","control_decode","measured_decode"):
            if not (a.work/f"{name}.json").exists(): write_json(a.work/f"{name}.json",err)
        for name in ("warm","control","measured"):
            write_json(a.work/f"{name}_release.json",{"status":"decode_failed"})
        return 1
    finally:
        if engine is not None and callable(getattr(engine,"shutdown",None)): engine.shutdown()
if __name__=="__main__":
    raise SystemExit(main())
