import tool_archive
import sys,json,hashlib,struct
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tool_archive import load_rollout,ArchiveError

def mk(tmp):
 root=Path(tmp); d=root/'stage-1';d.mkdir(parents=True); n=2;h=3;v=7
 ids=[11,12]; positions=[4,5];
 (d/'RUN.json').write_text(json.dumps({'model':'ring','passed':True,'source_dirty':False,'source_state_before':{'dirty':False},'generation':{'generated_token_ids':ids,'sampling':{'sampled_logprobs':[-.1,-.2]},'prompt_token_ids':[1,2,3,4,5]}}))
 f=np.arange(n*(11*h+8208),dtype='<f4');l=np.arange(n*v,dtype='<f4');f.tofile(d/'selected-terminal.f32');l.tofile(d/'selected-logits.f32')
 (d/'RETENTION.json').write_text(json.dumps({'retained':{'selected-terminal.f32':{'positions':positions,'sha256':hashlib.sha256((d/'selected-terminal.f32').read_bytes()).hexdigest(),'bytes':(d/'selected-terminal.f32').stat().st_size},'selected-logits.f32':{'positions':positions,'sha256':hashlib.sha256((d/'selected-logits.f32').read_bytes()).hexdigest(),'bytes':(d/'selected-logits.f32').stat().st_size}}}))
 rec={'schema':'strata-identity-tool-rollout-v1','model':'ring','group':'g','family':'arithmetic','split':'train','grade':{'reward':1.0},'policy_adapter_sha256':None,'transcript_ids':[1,2,3,4,5,11,12],'ids':ids,'sampling_logprobs':[-.1,-.2],'action_positions':[5,6],'native_stages':[{'path':'stage-1','run_sha256':'','retention_sha256':''}]}
 (root/'ROLLOUT.json').write_text(json.dumps(rec));
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 # fill bindings
 run=sha(d/'RUN.json');ret=sha(d/'RETENTION.json');rec['native_stages'][0]['run_sha256']=run;rec['native_stages'][0]['retention_sha256']=ret;(root/'ROLLOUT.json').write_text(json.dumps(rec))
 paths=[root/'ROLLOUT.json',d/'RUN.json',d/'RETENTION.json',d/'selected-terminal.f32',d/'selected-logits.f32']; man={str(p.relative_to(root)): {'sha256':sha(p),'size':p.stat().st_size} for p in paths};return root,man

def test_valid_multistage_shape_cpu(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);r=load_rollout(root,man);assert tuple(r['ids'])==(11,12) and tuple(r['action_positions'])==(5,6);assert tuple(r['features'].shape)==(2,8241);assert tuple(r['logits'].shape)==(2,7) and r['features'].device.type=='cpu';assert r['family']=='arithmetic' and r['grade']['reward']==1.0 and r['sampling_logprobs']==(-.1,-.2) and len(r['native_stages'])==1
def test_hash_tamper_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);man['stage-1/RUN.json']['sha256']='0'*64
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError
def test_position_tamper_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);p=root/'stage-1/RETENTION.json';x=json.loads(p.read_text());x['retained']['selected-terminal.f32']['positions']=[3,5];p.write_text(json.dumps(x));man['stage-1/RETENTION.json']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();man['stage-1/RETENTION.json']['size']=p.stat().st_size
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError
def test_escape_and_missing_manifest_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);man['../outside']='0'*64
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError

def test_action_id_tamper_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);p=root/'ROLLOUT.json';x=json.loads(p.read_text());x['ids']=[99,12];p.write_text(json.dumps(x));man['ROLLOUT.json']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();man['ROLLOUT.json']['size']=p.stat().st_size
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError

def test_duplicate_stage_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7)
 root,man=mk(tmp_path);p=root/'ROLLOUT.json';x=json.loads(p.read_text());x['native_stages'].append(x['native_stages'][0]);p.write_text(json.dumps(x));man['ROLLOUT.json']['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();man['ROLLOUT.json']['size']=p.stat().st_size
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError

def test_symlink_root_and_component_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7); root,man=mk(tmp_path); link=tmp_path/"alias";link.symlink_to(root,target_is_directory=True)
 try:load_rollout(link,man)
 except ArchiveError:pass
 else:raise AssertionError("symlink root accepted")
 (root/"escape").symlink_to(tmp_path/"outside",target_is_directory=True);man["escape/ROLLOUT.json"]={"sha256":"0"*64,"size":1}
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError("symlink component accepted")

def test_prompt_transcript_and_model_binding_rejected(tmp_path):
 tool_archive.MODEL_GEOMETRY["ring"]=(3,7); root,man=mk(tmp_path);p=root/"ROLLOUT.json";x=json.loads(p.read_text());x["transcript_ids"][0]=99;p.write_text(json.dumps(x));man["ROLLOUT.json"]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest();man["ROLLOUT.json"]["size"]=p.stat().st_size
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError("altered transcript accepted")
 root,man=mk(tmp_path/"model");p=root/"ROLLOUT.json";x=json.loads(p.read_text());x["model"]="ling";p.write_text(json.dumps(x));man["ROLLOUT.json"]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest();man["ROLLOUT.json"]["size"]=p.stat().st_size
 try:load_rollout(root,man)
 except ArchiveError:pass
 else:raise AssertionError("mismatched model accepted")
