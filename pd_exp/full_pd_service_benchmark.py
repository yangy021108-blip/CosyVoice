#!/usr/bin/env python3
"""Acoustic completion for a persistent PD LLM run (research-only)."""
from __future__ import annotations
import argparse,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
def pct(v,q):
    if not v:return None
    x=sorted(float(a) for a in v); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1)
    return x[lo]*(hi-pos)+x[hi]*(pos-lo)
def summ(v):
    x=[float(a) for a in v if a is not None]
    return {'count':len(x),'mean':sum(x)/len(x) if x else None,'p50':pct(x,.5),'p95':pct(x,.95),'p99':pct(x,.99),'max':max(x) if x else None}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--llm-results-dir',type=Path,required=True); p.add_argument('--model-dir',type=Path,default=ROOT/'pretrained_models/Fun-CosyVoice3-0.5B'); p.add_argument('--prompt-payload',type=Path,required=True); p.add_argument('--results-dir',type=Path,required=True); p.add_argument('--voice',default='default'); p.add_argument('--voices-json',type=Path,default=ROOT/'pd_exp/voices.json'); p.add_argument('--text',default=None); p.add_argument('--matcha-path',type=Path,default=ROOT/'third_party/Matcha-TTS'); p.add_argument('--flow-seed',type=int,default=0); a=p.parse_args()
    if str(a.matcha_path) not in sys.path: sys.path.append(str(a.matcha_path))
    import torch
    from pd_exp.common import DEFAULT_TEXT,read_json,wrap_nvtx_method,write_json
    from research.error_pattern.phase2_5_core import build_model_inputs,decode_fixed_trajectory,filter_silent_tokens,load_research_backend,write_waveform
    from tools.cosyvoice_phase_timing import PhaseCollector,wrap_method
    a.results_dir.mkdir(parents=True,exist_ok=True); (a.results_dir/'audio').mkdir(exist_ok=True)
    text=a.text or DEFAULT_TEXT; payload=torch.load(a.prompt_payload,map_location='cpu',weights_only=False); spans=payload['metadata']['spans']
    backend,_=load_research_backend(model_dir=a.model_dir,voices_json=a.voices_json,voice_id=a.voice,repository_root=ROOT,load_vllm=False,fp16=False)
    wrap_nvtx_method(backend.model.flow,'inference','COSY_FLOW'); wrap_nvtx_method(backend.model.hift,'inference','COSY_HIFT'); collector=PhaseCollector(); wrap_method(backend.model.flow,'inference','flow_seconds',collector); wrap_method(backend.model.hift,'inference','hift_seconds',collector)
    paths=sorted(a.llm_results_dir.glob('decode_*.json'),key=lambda q:int(re.search(r'(\d+)',q.stem).group(1)))
    records=[]; all_started=time.perf_counter()
    for i,path in enumerate(paths):
        d=read_json(path)
        if d.get('status')!='ok': raise RuntimeError(f'PD decode failed: {path}: {d}')
        begun=time.perf_counter(); f0=time.perf_counter(); chunks,model_inputs=build_model_inputs(backend,text=text,voice_id=a.voice,text_frontend=True); frontend=time.perf_counter()-f0
        raw=[int(x) for x in d['raw_speech_tokens']]; filtered,dropped=filter_silent_tokens(raw,backend.model.silent_tokens); mark=collector.mark(); waveform,acoustic=decode_fixed_trajectory(backend,trajectory={'text':text,'frontend_chunks':chunks,'chunks':[{'spans':spans,'filtered_speech_tokens':filtered}]},flow_seed=a.flow_seed,voice_id=a.voice,text_frontend=True,speed=1.0); phase=collector.since(mark); wav=write_waveform(a.results_dir/'audio'/f'request_{i:04d}.wav',waveform,int(backend.sample_rate)); lm=d['metrics']; e2e=frontend+float(lm['total_seconds'])+float(acoustic['decode_latency_seconds'])
        records.append({'iteration':i,'frontend_seconds':frontend,'llm_seconds':lm['total_seconds'],'ttft_seconds':lm.get('first_token_seconds'),'tpot':{'mean':lm.get('mean_tpot_seconds'),'p50':pct(lm.get('token_intervals_seconds',[]),.5),'p95':pct(lm.get('token_intervals_seconds',[]),.95),'p99':pct(lm.get('token_intervals_seconds',[]),.99)},'flow_seconds':phase.get('flow_seconds',0.0),'hift_seconds':phase.get('hift_seconds',0.0),'acoustic_total_seconds':acoustic['decode_latency_seconds'],'e2e_seconds':e2e,'audio_duration_seconds':acoustic.get('audio_duration_seconds'),'rtf':e2e/acoustic['audio_duration_seconds'] if acoustic.get('audio_duration_seconds') else None,'raw_token_count':len(raw),'filtered_token_count':len(filtered),'audio':{**acoustic,**wav},'cer':None,'wer':None,'speaker_similarity':None})
        print(json.dumps({'iteration':i,'decode_file':str(path),'e2e_estimate_seconds':e2e,'raw_tokens':len(raw),'audio_seconds':acoustic.get('audio_duration_seconds')}),flush=True)
    def fld(k): return summ([r.get(k) for r in records])
    out={'mode':'pd_full_service_posthoc_acoustic','engine_mode':'FULL_DECODE_ONLY','llm_results_dir':str(a.llm_results_dir.resolve()),'iterations':len(records),'startup_seconds':read_json(a.llm_results_dir/'pd_metrics.json').get('worker_startup_to_ready_seconds') if (a.llm_results_dir/'pd_metrics.json').exists() else None,'measurement_note':'PD LLM requests ran persistently; Flow/HiFT were executed as each decode result became available to this posthoc harness. E2E is frontend + per-request LLM + acoustic estimate, not a wall-clock online overlap trace.','e2e':fld('e2e_seconds'),'ttft':fld('ttft_seconds'),'llm':fld('llm_seconds'),'llm_tpot_p50_ms':summ([r['tpot']['p50']*1000 for r in records]),'llm_tpot_p95_ms':summ([r['tpot']['p95']*1000 for r in records]),'llm_tpot_p99_ms':summ([r['tpot']['p99']*1000 for r in records]),'flow':fld('flow_seconds'),'hift':fld('hift_seconds'),'acoustic':fld('acoustic_total_seconds'),'rtf':fld('rtf'),'audio_duration':fld('audio_duration_seconds'),'raw_token_count':fld('raw_token_count'),'filtered_token_count':fld('filtered_token_count'),'quality':{'acceptable_rate':sum(bool(r['audio'].get('quality_acceptable')) for r in records)/len(records) if records else None,'clipping_ratio_max':max((r['audio'].get('clipping_ratio',0.0) for r in records),default=None),'cer':None,'wer':None,'speaker_similarity':None},'records':records,'total_script_seconds':time.perf_counter()-all_started}
    write_json(a.results_dir/'summary.json',out); print(json.dumps(out,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
