#!/usr/bin/env python3
"""Persistent full CosyVoice pipeline benchmark for service-scenario research."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def pct(values,q):
    if not values: return None
    x=sorted(values); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1)
    return x[lo]*(hi-pos)+x[hi]*(pos-lo)

def summary(values):
    vals=[float(v) for v in values if v is not None]
    return {"count":len(vals),"mean":sum(vals)/len(vals) if vals else None,
            "p50":pct(vals,.5),"p95":pct(vals,.95),"p99":pct(vals,.99),
            "max":max(vals) if vals else None}

def args_parse():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='0'); p.add_argument('--model-dir',type=Path,default=ROOT/'pretrained_models/Fun-CosyVoice3-0.5B')
    p.add_argument('--voices-json',type=Path,default=ROOT/'pd_exp/voices.json'); p.add_argument('--voice',default='default')
    p.add_argument('--text',default=None); p.add_argument('--seed',type=int,default=20260726)
    p.add_argument('--iterations',type=int,default=30); p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--prompt-payload',type=Path,default=None); p.add_argument('--flow-seed',type=int,default=0)
    p.add_argument('--matcha-path',type=Path,default=ROOT/'third_party/Matcha-TTS')
    return p.parse_args()

def main():
    a=args_parse(); os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    if str(a.matcha_path) not in sys.path: sys.path.append(str(a.matcha_path))
    import torch
    from pd_exp.common import (DEFAULT_TEXT,build_engine,make_sampling_params,prompt_payload,
        run_engine_request,timing_summary,wrap_nvtx_method,write_json)
    from research.error_pattern.phase2_5_core import (_prepare_lm_input,build_model_inputs,
        decode_fixed_trajectory,filter_silent_tokens,load_research_backend,write_waveform)
    from tools.cosyvoice_phase_timing import PhaseCollector,wrap_method
    a.results_dir.mkdir(parents=True,exist_ok=True); (a.results_dir/'audio').mkdir(exist_ok=True)
    text=a.text or DEFAULT_TEXT
    backend,_=load_research_backend(model_dir=a.model_dir,voices_json=a.voices_json,voice_id=a.voice,
        repository_root=ROOT,load_vllm=False,fp16=False)
    llm=backend.model.llm; llm.vllm=build_engine(a.model_dir/'vllm',eager=False,max_num_seqs=1)
    llm.lock=__import__('threading').Lock()
    try: del llm.llm.model.model.layers
    except Exception: pass
    canonical=None; spans=None; chunks0=None
    if a.prompt_payload is not None:
        canonical=torch.load(a.prompt_payload,map_location='cpu',weights_only=False)
        spans=canonical['metadata']['spans']
    wrap_nvtx_method(backend.model.flow,'inference','COSY_FLOW')
    wrap_nvtx_method(backend.model.hift,'inference','COSY_HIFT')
    collector=PhaseCollector()
    wrap_method(backend.model.flow,'inference','flow_seconds',collector)
    wrap_method(backend.model.hift,'inference','hift_seconds',collector)
    records=[]
    started_all=time.perf_counter()
    try:
        for i in range(a.iterations):
            iter_started=time.perf_counter()
            frontend_started=time.perf_counter()
            chunks,model_inputs=build_model_inputs(backend,text=text,voice_id=a.voice,text_frontend=True)
            frontend_seconds=time.perf_counter()-frontend_started
            if len(model_inputs)!=1: raise RuntimeError('benchmark requires exactly one frontend chunk')
            if canonical is None:
                lm_input,minimum_tokens,maximum_tokens,iter_spans=_prepare_lm_input(llm,model_inputs[0],min_token_text_ratio=2.0,max_token_text_ratio=20.0)
                payload=prompt_payload(lm_input.squeeze(0),spans=iter_spans,minimum_tokens=minimum_tokens,maximum_tokens=maximum_tokens,seed=a.seed,stop_token_ids=list(llm.stop_token_ids))
                spans=iter_spans
            else:
                payload=canonical
            sampling=make_sampling_params(payload['metadata'])
            raw,llm_metrics,_=run_engine_request(llm.vllm,payload['prompt_embeds'],sampling,request_id=f'cosy-unified-service-{i:04d}',nvtx_label='COSY_UNIFIED_SERVICE')
            filtered,dropped=filter_silent_tokens(raw,backend.model.silent_tokens)
            mark=collector.mark()
            waveform,acoustic=decode_fixed_trajectory(backend,trajectory={'text':text,'frontend_chunks':chunks,'chunks':[{'spans':spans,'filtered_speech_tokens':filtered}]},flow_seed=a.flow_seed,voice_id=a.voice,text_frontend=True,speed=1.0)
            phases=collector.since(mark)
            wav=write_waveform(a.results_dir/'audio'/f'request_{i:04d}.wav',waveform,int(backend.sample_rate))
            e2e=time.perf_counter()-iter_started
            records.append({'iteration':i,'frontend_seconds':frontend_seconds,'llm_seconds':llm_metrics['total_seconds'],'ttft_seconds':llm_metrics['first_token_seconds'],'tpot':timing_summary(llm_metrics),'flow_seconds':phases.get('flow_seconds',0.0),'hift_seconds':phases.get('hift_seconds',0.0),'acoustic_total_seconds':acoustic['decode_latency_seconds'],'e2e_seconds':e2e,'audio_duration_seconds':acoustic['audio_duration_seconds'],'rtf':e2e/acoustic['audio_duration_seconds'] if acoustic['audio_duration_seconds'] else None,'raw_token_count':len(raw),'filtered_token_count':len(filtered),'audio':{**acoustic,**wav},'cer':None,'wer':None,'speaker_similarity':None})
            print(json.dumps({'iteration':i,'e2e_seconds':e2e,'raw_tokens':len(raw),'audio_seconds':acoustic['audio_duration_seconds']}),flush=True)
    finally:
        if callable(getattr(llm.vllm,'shutdown',None)): llm.vllm.shutdown()
    def field(name): return summary([r.get(name) for r in records])
    tpot_all=[v for r in records for v in (r['tpot'].get('p99'),)]
    out={'mode':'unified_full_service','engine_mode':'FULL_DECODE_ONLY','gpu':str(a.gpu),'iterations':a.iterations,'text':text,'prompt_payload':None if a.prompt_payload is None else str(a.prompt_payload.resolve()),'e2e':field('e2e_seconds'),'ttft':field('ttft_seconds'),'llm':field('llm_seconds'),'llm_tpot_p50_ms':summary([r['tpot']['p50']*1000 for r in records]),'llm_tpot_p95_ms':summary([r['tpot']['p95']*1000 for r in records]),'llm_tpot_p99_ms':summary([r['tpot']['p99']*1000 for r in records]),'flow':field('flow_seconds'),'hift':field('hift_seconds'),'acoustic':field('acoustic_total_seconds'),'rtf':field('rtf'),'audio_duration':field('audio_duration_seconds'),'raw_token_count':field('raw_token_count'),'filtered_token_count':field('filtered_token_count'),'quality':{'acceptable_rate':sum(bool(r['audio'].get('quality_acceptable')) for r in records)/len(records) if records else None,'clipping_ratio_max':max((r['audio'].get('clipping_ratio',0.0) for r in records),default=None),'cer':None,'wer':None,'speaker_similarity':None},'records':records,'total_script_seconds':time.perf_counter()-started_all}
    write_json(a.results_dir/'summary.json',out); print(json.dumps(out,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
