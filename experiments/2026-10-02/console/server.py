#!/usr/bin/env python3
"""A1Z Console: web front end for evaluation, HIL collection and control tests on the Jetson.

Runs on the Jetson (inference venv), listens on 127.0.0.1 only; the Mac reaches it through an SSH tunnel
("A1Z Console.command"). It never talks to the motors itself. Sessions are the same tmux windows eval.sh
creates (A1Z_NO_ATTACH=1), and every action is a key press sent to that window, so the terminal flow, its
safety layer (robot_proc.py) and its files stay exactly as before; `tmux attach -t eval` still works as a fallback.
READY is sent only when the operator clicks it in the page and the client is waiting for it.

Reads: tmux pane text (session state), /dev/shm/a1z_live.jpg (wrist camera, written by a1z_robot.py while a
session runs), /dev/shm/a1z_side.jpg (HIL-SERL side camera, written by whichever process holds it), flight.jsonl of
the newest session (joints, temperatures), trial/run JSON and MP4 files.
Writes: only operator edits of trial labels / exclusion in trial JSON (marked edited_via=console), and the side crop in
../2026-10-09-hilserl/cameras.json. Starts/stops only the side-camera preview (tmux "sidecam"), never the arm.
"""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import json
import math
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse,parse_qs,unquote

ROOT=Path(__file__).resolve().parent.parent          # ~/a1z-experiments/2026-10-02
STATIC=Path(__file__).resolve().parent/'static'
INFER=Path.home()/'a1z-infer'
CKPTS=INFER/'checkpoints'
KINDS={'eval':ROOT/'eval-runs','control':ROOT/'control-runs','hil':ROOT/'hil-runs'}
LIVE='/dev/shm/a1z_live.jpg'
SIDE_LIVE='/dev/shm/a1z_side.jpg';SIDE_INFO='/dev/shm/a1z_side.json'
HILSERL=ROOT.parent/'2026-10-09-hilserl';CAMERAS=HILSERL/'cameras.json'
SIDE_W,SIDE_H=640,480
PORT=int(os.environ.get('A1Z_CONSOLE_PORT','8780'))
def _token():
    """Per-machine token kept across server restarts, so an open page keeps working after a restart."""
    f=Path.home()/'.a1z_console_token'
    if not f.exists():
        f.write_text(secrets.token_hex(16));f.chmod(0o600)
    return f.read_text().strip()
TOKEN=_token()
TMUX='eval'
GRIP_WARN_C=80.0;GRIP_BLOCK_C=88.0
sys.path.insert(0,str(ROOT))

_kin=None
def fk_mm(q):
    global _kin
    if _kin is None:
        from insertion import Kin
        _kin=Kin()
    return [float(v) for v in (_kin.fk(q)[:3,3]*1000)]


# ---------------- tmux session ----------------
def sh(cmd,timeout=10):
    r=subprocess.run(cmd,shell=isinstance(cmd,str),capture_output=True,text=True,timeout=timeout)
    return r.returncode,r.stdout,r.stderr

def tmux_alive():
    return sh(['tmux','has-session','-t',TMUX])[0]==0

def pane_text(lines=400):
    if not tmux_alive():return ''
    code,out,_=sh(['tmux','capture-pane','-p','-J','-S',f'-{lines}','-t',f'{TMUX}:client'])
    return out if code==0 else ''

MARKERS=[  # (regex, state) — the latest match in the pane decides the state; patterns match the client scripts' Chinese output, do not translate
    (r'现场就绪，输入 READY：','await_ready'),
    (r'SPACEMOUSE_STARTING','starting'),
    (r'SPACEMOUSE_HOLD','teleop_hold'),
    (r'SPACEMOUSE_ACTIVE','teleop_active'),
    (r'就绪。空格=开/合夹爪','idle'),
    (r'已回到重力辅助|回到重力辅助。','idle'),
    (r'▶ 模型开始|▶ 交还模型','running'),
    (r'▶ A1 示教点 开始|▶ A2 回放 开始','running'),
    (r'✋ 你接管了','human'),
    (r'● 开始录制','recording'),
    (r'录好了：|录制停止并保存|录制已中止|录制太短','idle'),
    (r'SESSION_SAVED','ended'),
    (r'按回车关闭','ended'),
]

def session_state():
    if not tmux_alive():return {'state':'none','pane':'','mode':None}
    text=pane_text();best=('starting',-1)
    for rx,st in MARKERS:
        for m in re.finditer(rx,text,flags=re.M):
            if m.start()>best[1]:best=(st,m.start())
    mode=next((pr['id'] for pr in sorted(PROGRAMS,key=lambda q:q.get('detect_rank',50)) if any(d in text for d in pr.get('detect',()))),None)  # lowest detect_rank wins ties (teleop, control, hil, eval)
    m=re.search(r'模型：(\S+)',text)
    model=m.group(1).replace('/pretrained_model','') if m else None
    return {'state':best[0],'pane':'\n'.join(text.rstrip().splitlines()[-80:]),'mode':mode,'model':model}

KEYS={'g','s',' ','x','d','i','1','2','3','t','r','p','w','q'}

def send_key(k):
    if k=='READY':
        st=session_state()
        if st['state']!='await_ready':return False,'The client is not waiting for READY'
        sh(['tmux','send-keys','-t',f'{TMUX}:client','-l','READY']);sh(['tmux','send-keys','-t',f'{TMUX}:client','Enter'])
        return True,'READY sent'
    if k=='CANCEL':
        st=session_state()
        if st['state']!='await_ready':return False,'Can only cancel while waiting for READY'
        sh(['tmux','send-keys','-t',f'{TMUX}:client','-l','no']);sh(['tmux','send-keys','-t',f'{TMUX}:client','Enter'])
        return True,'Cancelled'
    if k=='ENTER':
        sh(['tmux','send-keys','-t',f'{TMUX}:client','Enter']);return True,'ok'
    if k not in KEYS:return False,'Unsupported key'
    sh(['tmux','send-keys','-t',f'{TMUX}:client','-l',k]);return True,'ok'

# ---------------- program registry ----------------
# One entry per launchable program; GET /api/programs sends the public fields to the page, which renders the picker and
# only the inputs a program declares. Adding a program = adding one entry here (plus `launch`, and `detect` strings that
# its terminal output prints so session_state() can recognise it, and `detect_rank` to order ties). The POST /api/session/start body keeps its old shape:
# {"mode": <id>, <input name>: value, ...}; the advanced inputs are optional and the defaults below apply when omitted.
INPUTS={  # input name -> how the page renders it; "advanced" ones sit in a closed disclosure and are sent only when filled in
    'model':{'label':'Model','type':'model'},
    'epoch':{'label':'HG-DAgger round (epoch)','type':'int','min':1,'default':3},
    'stiffness':{'label':'Stiffness × vendor kp (legacy gains)','type':'number','step':0.1,'placeholder':'default dimos gains'},
    'ki':{'label':'Integral ki, 1/s','type':'number','step':0.5,'placeholder':'2'},
    'ki_max':{'label':'Integral cap, deg','type':'number','step':0.5,'placeholder':'1'},
    'grip_hold':{'label':'Grip hold between trials, N·m','type':'number','step':0.5,'placeholder':'1.5'},
}
TUNING=['stiffness','ki','ki_max','grip_hold']  # advanced overrides understood by the eval.sh programs
PROGRAMS=[
    {'id':'teleop','label':'SpaceMouse','description':'Move the arm with the SpaceMouse on the Jetson; translation only at startup.',
     'inputs':[],'advanced':[],'start_label':'Prepare SpaceMouse','note':'Next, confirm on-site readiness before enabling the arm.',
     'launch':{'type':'script','script':'spacemouse_teleop.py','args':['--execute']},'detect':['SpaceMouse 遥操','SPACEMOUSE_'],'detect_rank':0},
    {'id':'eval','label':'Eval','description':'Run a trained policy on the arm; label each trial after it ends.',
     'inputs':['model'],'advanced':TUNING,'launch':{'type':'eval_sh','args':[]},'detect':['评估模式'],'detect_rank':3},
    {'id':'hil','label':'HIL collection','description':'HG-DAgger: the policy drives, you take over to correct it.',
     'inputs':['model','epoch'],'advanced':TUNING,'launch':{'type':'eval_sh','args':['--hil']},'detect':['HIL 采集模式'],'detect_rank':2},
    {'id':'control','label':'Control test','description':'Teach-and-return and record-and-replay, no model.',
     'inputs':[],'advanced':TUNING,'launch':{'type':'eval_sh','args':['--control']},'detect':['控制层测试'],'detect_rank':1},
]
PROGRAM_BY_ID={pr['id']:pr for pr in PROGRAMS}

def programs_public():
    """What the page needs: the picker entries and the definition of every input they mention."""
    pub=[{k:v for k,v in pr.items() if k not in ('launch','detect','detect_rank')} for pr in PROGRAMS]
    names={n for pr in pub for n in pr['inputs']+pr['advanced']}
    return {'programs':pub,'inputs':{n:INPUTS[n] for n in names}}

def start_session(p):
    if tmux_alive():return False,'A session is already running (end it first)'
    prog=PROGRAM_BY_ID.get(p.get('mode','eval'));launch=(prog or {}).get('launch') or {}
    if launch.get('type')=='script':
        script=ROOT/launch['script']
        if not script.exists():return False,'SpaceMouse program not installed'
        command='cd '+shlex.quote(str(ROOT))+' && '+shlex.quote(str(INFER/'run.sh'))+' -u '+shlex.quote(str(script))+''.join(' '+shlex.quote(a) for a in launch.get('args',[]))+'; echo; read -r -p "Session ended. Press Enter to close"; tmux kill-session -t eval'
        code,_,err=sh(['tmux','new-session','-d','-s',TMUX,'-n','client',command])
        return code==0, 'SpaceMouse ready for on-site confirmation' if code==0 else err
    try:
        stiff=p.get('stiffness');stiff=None if stiff in (None,'','default') else float(stiff)  # None = default dimos gains (2026-10-09)
        ki=float(p.get('ki',2));kimax=float(p.get('ki_max',1))
        hold=p.get('grip_hold',1.5);hold=None if hold in (None,'','off') else float(hold)  # omitted = 1.5 N·m, the same as the client script's default
    except ValueError:return False,'Invalid parameter format'
    if hold is not None and not 0.5<=hold<=4:return False,'Grip hold between trials must be 0.5–4 N·m'
    if not ((stiff is None or 0.5<=stiff<=3.0) and 0<=ki<=5 and 0<kimax<=6):return False,'Parameter out of range (stiffness 0.5–3 or empty for default gains, ki 0–5, cap ≤6°)'
    env=dict(os.environ,A1Z_NO_ATTACH='1')
    args=(['--stiffness',f'{stiff}'] if stiff is not None else [])+['--ki',f'{ki}','--ki-max',f'{kimax}']+(['--grip-hold',f'{hold}'] if hold is not None else [])
    if launch.get('type')!='eval_sh':return False,'Unknown mode'
    lead=list(launch.get('args',[]))
    if 'model' in prog['inputs']:
        ck=p.get('model','')
        if not re.fullmatch(r'[A-Za-z0-9._-]+',ck) or not (CKPTS/ck/'pretrained_model'/'model.safetensors').exists():return False,'Model not found'
        env['CKPT']=str(CKPTS/ck/'pretrained_model')
    if 'epoch' in prog['inputs']:
        try:ep=int(p.get('epoch',1))
        except ValueError:return False,'Invalid epoch'
        lead+=['--epoch',str(ep)]
    cmd=[str(INFER/'eval.sh')]+lead+args
    def run():subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=60)
    threading.Thread(target=run,daemon=True).start()
    return True,'Session starting (model load takes ~10 s)'

def close_session():
    st=session_state()
    if st['state']=='none':return True,'No session'
    if st['state'] not in ('ended',):return False,'Client still running: click Quit session on the page first'
    sh(['tmux','kill-session','-t',TMUX]);return True,'Session closed'


# ---------------- side camera (HIL-SERL) ----------------
def has_tmux(name):return sh(['tmux','has-session','-t',name])[0]==0

def side_status():
    age=time.time()-os.path.getmtime(SIDE_LIVE) if os.path.exists(SIDE_LIVE) else None
    info=load_json(Path(SIDE_INFO)) if os.path.exists(SIDE_INFO) else None
    cams=(load_json(CAMERAS) if CAMERAS.exists() else None) or {}
    return {'age':None if age is None else round(age,2),'camera':info,'preview':has_tmux('sidecam'),'actor':has_tmux('hil'),
            'size':[SIDE_W,SIDE_H],'crop_side':cams.get('crop_side'),'saved_at':cams.get('crop_side_saved_at'),
            'crop':cams.get('crop'),'crop_saved_at':cams.get('crop_saved_at'),
            'wrist_age':round(time.time()-os.path.getmtime(LIVE),1) if os.path.exists(LIVE) else None,
            'installed':(HILSERL/'side_preview.py').exists()}

def side_preview(on):
    if not on:
        if has_tmux('sidecam'):sh(['tmux','kill-session','-t','sidecam'])
        return True,'Preview stopped'
    if has_tmux('hil'):return False,'The HIL-SERL actor is running and already streams the side camera'
    if has_tmux('sidecam'):return True,'Preview already running'
    if not (HILSERL/'side_preview.py').exists():return False,'side_preview.py not installed on Jetson'
    cmd='cd '+shlex.quote(str(HILSERL))+' && exec '+shlex.quote(str(INFER/'run.sh'))+' side_preview.py --headless'
    code,_,err=sh(['tmux','new-session','-d','-s','sidecam','/bin/bash','--noprofile','--norc','-c',cmd])
    return code==0,'Preview starting (camera warm-up ~3 s)' if code==0 else err

def side_crop(p):
    """Save a 640x480 crop to cameras.json: {'crop_side':[x,y,w,h]} for the side camera or {'crop':[...]} for the wrist."""
    key='crop' if 'crop' in p else 'crop_side'
    try:x,y,w,h=[int(v) for v in p.get(key)]
    except (TypeError,ValueError):return False,f'{key} must be four integers'
    if not (w>=32 and h>=32 and x>=0 and y>=0 and x+w<=SIDE_W and y+h<=SIDE_H):return False,'Crop must lie inside the 640x480 image'
    d=(load_json(CAMERAS) if CAMERAS.exists() else None) or {}
    d.update({key:[x,y,w,h],key+'_saved_at':time.strftime('%Y-%m-%dT%H:%M:%S')})
    tmp=CAMERAS.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=1));os.replace(tmp,CAMERAS)
    flag='--crop' if key=='crop' else '--crop-side'
    return True,f'Saved {"wrist" if key=="crop" else "side"} crop {x} {y} {w} {h} (used by actor.py when {flag} is not given)'


# ---------------- telemetry ----------------
def newest_session():
    best=None
    for kind,d in KINDS.items():
        if not d.exists():continue
        for s in d.iterdir():
            f=s/'flight.jsonl'
            if s.is_dir() and f.exists():
                m=f.stat().st_mtime
                if best is None or m>best[0]:best=(m,kind,s)
    return best

class FlightReader:
    """Incremental reader of one session's flight.jsonl: newest row plus a ~1 Hz history (last 15 min)."""
    def __init__(self,path):self.path=path;self.offset=0;self.buf='';self.last=None;self.history=[];self.n=0;self.lock=threading.Lock()
    def poll(self):
        with self.lock:
            with open(self.path,'rb') as f:
                f.seek(self.offset);data=f.read(8_000_000);self.offset+=len(data)
            lines=(self.buf+data.decode(errors='ignore')).split('\n');self.buf=lines.pop()
            for line in lines:
                if '"q"' not in line:continue
                try:r=json.loads(line)
                except ValueError:continue
                self.last=r;self.n+=1
                if self.n%50==1:
                    self.history.append({'t':r['t'],'g_rotor':r.get('g_temp_rotor'),'g_mos':r.get('g_temp_mos'),
                                         'j_max':max(r['temp_rotor']) if r.get('temp_rotor') else None,'mode':r.get('mode')})
                    if len(self.history)>900:self.history=self.history[-900:]
            return self.last,list(self.history)

_readers={}
def telemetry():
    if session_state().get('mode')=='teleop':
        try:data=json.loads((ROOT/'spacemouse-state.json').read_text())
        except (OSError,ValueError):data={}
        live=time.time()-data.get('updated_at',0)<2 and data.get('state') in ('teleop_active','teleop_hold')
        return {'session':None,'kind':'teleop','live':live,'spacemouse':data,
                'tip_mm':data.get('tip_mm') if live else None,'temp_rotor':data.get('temp_rotor') if live else None}
    ns=newest_session()
    if ns is None:return {'session':None}
    mtime,kind,s=ns;f=s/'flight.jsonl'
    rd=_readers.get(str(f))
    if rd is None:rd=_readers[str(f)]=FlightReader(f)
    last,hist=rd.poll()
    if last is None:return {'session':s.name,'kind':kind}
    t_end=last['t']
    out={'session':s.name,'kind':kind,'age_s':round(time.time()-mtime,1),'live':time.time()-mtime<3,
         'mode':last.get('mode'),'q_deg':[round(math.degrees(v),2) for v in last['q']],'hz':last.get('hz'),
         'grip':last.get('grip'),'temp_mos':last.get('temp_mos'),'temp_rotor':last.get('temp_rotor'),
         'g_tau':last.get('g_tau'),'g_temp_rotor':last.get('g_temp_rotor'),'g_temp_mos':last.get('g_temp_mos'),
         'i_off_deg':[round(math.degrees(v),2) for v in last.get('i_off',[0]*6)],
         'history':[{**h,'t':round(h['t']-t_end,1)} for h in hist],
         'grip_warn_c':GRIP_WARN_C,'grip_block_c':GRIP_BLOCK_C}
    try:out['tip_mm']=[round(v,1) for v in fk_mm(last['q'])]
    except Exception:pass
    return out

def usb_status():
    try:
        from usb_watch import UsbWatcher
        w=UsbWatcher();return {'connected':bool(w.connected),'target':w.describe()['targets']}
    except Exception as e:return {'connected':None,'error':repr(e)}


# ---------------- runs ----------------
_cache={}
def load_json(f):
    try:return json.loads(Path(f).read_text())
    except Exception:return None

def trial_files(s):
    return sorted(list(s.glob('trial-??.json'))+list(s.glob('run-??.json')))

def auto_excluded(t):
    return t.get('duration_s',0)<1.0 or 'too far' in str(t.get('end_reason',''))

def trial_summary(kind,s,f):
    key=(str(f),f.stat().st_mtime)
    if key in _cache:return _cache[key]
    t=load_json(f) or {}
    r={'file':f.name,'n':t.get('n'),'label':t.get('operator_label'),'usb':bool(t.get('success')),'end_reason':t.get('end_reason'),
       'duration_s':t.get('duration_s'),'excluded':bool(t.get('excluded',auto_excluded(t))),'excluded_manual':'excluded' in t,
       'video':f.with_suffix('.mp4').name if f.with_suffix('.mp4').exists() else None,'kind':t.get('kind'),
       'interventions':t.get('interventions'),'human_s':t.get('human_s'),'hil_keep':t.get('hil_keep')}
    if kind=='control':
        r['err']=t.get('end_dxyz_mm');r['hover']=t.get('hover_dxyz_mm')
    elif t.get('hand_ref_q') and t.get('log'):
        L=[e for e in t['log'] if e.get('stage','policy')=='policy'] or t['log']
        try:
            end=fk_mm(L[-1]['q']);ref=fk_mm(t['hand_ref_q']);r['err']=[round(a-b,2) for a,b in zip(end,ref)]
        except Exception:pass
    if r.get('err'):r['err_xy']=round(math.hypot(r['err'][0],r['err'][1]),2)
    _cache[key]=r;return r

def session_summary(kind,s):
    fs=trial_files(s);first=load_json(fs[0]) if fs else {}
    first=first or {}
    trials=[trial_summary(kind,s,f) for f in fs]
    ck=first.get('checkpoint') or ''
    ki=first.get('ki',0) or 0
    # integral v1 (plain, wound up under the policy) ran until ~11:58 on 2026-10-05; v2 integrates only while stuck
    kiv=first.get('ki_version') or (None if not ki else ('v1' if s.name<'20261005-115800' else 'v2'))
    return {'kind':kind,'session':s.name,'ki_version':kiv,'smooth':bool(first.get('smooth')),'model':Path(ck).parent.name if ck else ('control test' if kind=='control' else None),
            'stiffness':first.get('stiffness') if first.get('stiffness') is not None else ('kp '+'/'.join(f'{v:g}' for v in first['kp']) if first.get('kp') else None),'ki':first.get('ki',0) or 0,'epoch':first.get('epoch'),
            'trials':len(trials),'valid':sum(not x['excluded'] for x in trials),
            'labels':{k:sum(1 for x in trials if not x['excluded'] and x['label']==k) for k in ('missed','found_port','in_mouth','inserted','charging')},
            'usb':sum(1 for x in trials if not x['excluded'] and x['usb'])}

def list_runs():
    out=[]
    for kind,d in KINDS.items():
        if not d.exists():continue
        for s in sorted(d.iterdir(),reverse=True):
            if s.is_dir() and re.fullmatch(r'\d{8}-\d{6}',s.name) and trial_files(s):out.append(session_summary(kind,s))
    out.sort(key=lambda x:x['session'],reverse=True);return out

def run_detail(kind,name):
    s=KINDS[kind]/name
    if not s.is_dir():return None
    return {**session_summary(kind,s),'trials':[trial_summary(kind,s,f) for f in trial_files(s)]}

def edit_trial(p):
    kind=p.get('kind');name=p.get('session','');fname=p.get('file','')
    if kind not in KINDS or not re.fullmatch(r'\d{8}-\d{6}',name) or not re.fullmatch(r'(trial|run)-\d\d\.json',fname):return False,'Invalid parameters'
    f=KINDS[kind]/name/fname
    if not f.exists():return False,'Trial not found'
    t=json.loads(f.read_text());edits=t.setdefault('console_edits',[])
    if 'label' in p:
        lab={'1':'missed','2':'found_port' if kind!='control' else 'in_mouth','3':'inserted' if kind!='control' else 'charging'}.get(str(p['label']))
        if not lab:return False,'Label must be 1/2/3'
        edits.append({'t':time.time(),'label_from':t.get('operator_label'),'label_to':lab});t['operator_label']=lab;t['operator_label_auto']=False
    if 'excluded' in p:
        edits.append({'t':time.time(),'excluded':bool(p['excluded'])});t['excluded']=bool(p['excluded'])
    t['edited_via']='console';f.write_text(json.dumps(t));return True,'Saved'


# ---------------- report ----------------
def wilson(k,n,z=1.96):
    if n==0:return None
    p=k/n;d=1+z*z/n;c=p+z*z/(2*n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [round((c-h)/d,2),round((c+h)/d,2)]

def report():
    groups={}
    for r in list_runs():
        if r['kind']!='eval':continue
        det=run_detail('eval',r['session'])
        key=(r['model'],r['stiffness'],r['ki'],r['ki_version'],r['smooth'])
        g=groups.setdefault(key,{'model':r['model'],'stiffness':r['stiffness'],'ki':r['ki'],'ki_version':r['ki_version'],'smooth':r['smooth'],'sessions':[],'trials':[]})
        g['sessions'].append(r['session']);g['trials']+= [x for x in det['trials'] if not x['excluded']]
    out=[]
    for g in groups.values():
        T=g.pop('trials');n=len(T)
        labelled=[x for x in T if x['label']]
        found=sum(1 for x in labelled if x['label'] in ('found_port','inserted'))
        # operator label wins: USB can also fire when the operator inserts by hand before stopping the trial
        power=sum(1 for x in T if ((x['label'] in ('inserted','charging')) if x['label'] else x['usb']))
        errs=sorted(x['err_xy'] for x in T if x.get('err_xy') is not None)
        med=errs[len(errs)//2] if errs else None
        out.append({**g,'n':n,'labelled':len(labelled),'found':found,'found_ci':wilson(found,len(labelled)),
                    'power':power,'power_ci':wilson(power,n),'err_n':len(errs),'err_median':med,
                    'err_mean':round(sum(errs)/len(errs),2) if errs else None})
    out.sort(key=lambda x:(x['ki'] or 0,str(x['stiffness'] or ''),x['model'] or ''),reverse=True)  # stiffness may be a gains label
    return out


# ---------------- models / data ----------------
def models():
    out=[]
    order=['hil-e2-ft','hil-e1-ft','060000','hil-e1-scratch','020000']
    for d in sorted(CKPTS.iterdir(),key=lambda d:(order.index(d.name) if d.name in order else -1,d.name)) if CKPTS.exists() else []:
        pm=d/'pretrained_model'
        if not (pm/'model.safetensors').exists():continue
        tc=load_json(pm/'train_config.json') or {}
        ds=(tc.get('dataset') or {}).get('root') or ''
        out.append({'name':d.name,'path':str(pm),'saved':time.strftime('%Y-%m-%d %H:%M',time.localtime((pm/'model.safetensors').stat().st_mtime)),
                    'steps':int(d.name) if d.name.isdigit() else tc.get('steps'),'batch':tc.get('batch_size'),'dataset':Path(ds).name if ds else None,
                    'init':Path((tc.get('policy') or {}).get('pretrained_path') or '').parent.name or None,
                    'size_mb':round((pm/'model.safetensors').stat().st_size/1e6,1)})
    return out

DESCR={'060000':'Base ACT: 45 demos, 60k steps','020000':'Base ACT, 20k-step checkpoint','hil-e1-ft':'+1 round of HIL corrections (fine-tuned from 060000)',
       'hil-e1-scratch':'+1 round of HIL corrections (trained from scratch)','hil-e2-ft':'+2 rounds of HIL corrections (fine-tuned from hil-e1-ft)'}

def data_summary():
    col=load_json(ROOT/'collection.json') or {}
    eps=col.get('episodes',[])
    hil=[]
    d=KINDS['hil']
    for s in sorted(d.iterdir(),reverse=True) if d.exists() else []:
        fs=list(s.glob('trial-??.json'))
        if not fs:continue
        ts=[load_json(f) or {} for f in fs]
        hil.append({'session':s.name,'epoch':ts[0].get('epoch'),'model':Path(ts[0].get('checkpoint') or '').parent.name,'trials':len(ts),
                    'interventions':sum(t.get('interventions') or 0 for t in ts),'human_s':round(sum(t.get('human_s') or 0 for t in ts),1),
                    'kept':sum(1 for t in ts if t.get('hil_keep',True) and t.get('success')),'discarded':sum(1 for t in ts if t.get('hil_keep') is False)})
    return {'demos':{'total':len(eps),'usable':sum(1 for e in eps if not e.get('excluded') and e.get('usb_inserted')),
                     'excluded':sum(1 for e in eps if e.get('excluded'))},'hil':hil,
            'note':'Training sets (act-v2, agg-e1, agg-e2) are on the Mac in ~/a1z-data/usb-insert-20261003/datasets/'}


# ---------------- HTTP ----------------
class H(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def _send(self,code,body,ctype='application/json; charset=utf-8',extra=None):
        if not isinstance(body,(bytes,bytearray)):body=json.dumps(body,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store')
        for k,v in (extra or {}).items():self.send_header(k,v)
        self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
    def _host_ok(self):
        host=(self.headers.get('Host') or '').split(':')[0]
        return host in ('127.0.0.1','localhost')
    def do_GET(self):
        if not self._host_ok():return self._send(403,{'error':'host'})
        u=urlparse(self.path);p=u.path
        try:
            if p in ('/','/index.html'):
                html=(STATIC/'index.html').read_text().replace('__TOKEN__',TOKEN)
                return self._send(200,html.encode(),'text/html; charset=utf-8')
            if p.startswith('/static/'):
                f=(STATIC/p[len('/static/'):]).resolve()
                if STATIC.resolve() not in f.parents or not f.exists():return self._send(404,{'error':'not found'})
                ctype={'.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml'}.get(f.suffix,'application/octet-stream')
                return self._send(200,f.read_bytes(),ctype+'; charset=utf-8')
            if p=='/api/session':return self._send(200,session_state())
            if p=='/api/programs':return self._send(200,programs_public())
            if p=='/api/telemetry':return self._send(200,telemetry())
            if p=='/api/usb':return self._send(200,usb_status())
            if p=='/api/runs':return self._send(200,list_runs())
            if p.startswith('/api/runs/'):
                _,_,_,kind,name=p.split('/')[:5]
                if kind not in KINDS or not re.fullmatch(r'\d{8}-\d{6}',name):return self._send(400,{'error':'bad'})
                d=run_detail(kind,name);return self._send(200 if d else 404,d or {'error':'not found'})
            if p=='/api/report':return self._send(200,report())
            if p=='/api/models':return self._send(200,[{**m,'description':DESCR.get(m['name'],'')} for m in models()])
            if p=='/api/data':return self._send(200,data_summary())
            if p=='/api/live.jpg':
                if not os.path.exists(LIVE):return self._send(404,{'error':'no frame'})
                age=time.time()-os.path.getmtime(LIVE)
                return self._send(200,open(LIVE,'rb').read(),'image/jpeg',{'X-Frame-Age':f'{age:.2f}'})
            if p=='/api/side':return self._send(200,side_status())
            if p=='/api/side.jpg':
                if not os.path.exists(SIDE_LIVE):return self._send(404,{'error':'no frame'})
                return self._send(200,open(SIDE_LIVE,'rb').read(),'image/jpeg')
            if p.startswith('/api/video/'):return self._video(p)
            return self._send(404,{'error':'not found'})
        except Exception as e:
            return self._send(500,{'error':repr(e)})
    def _video(self,p):
        parts=[unquote(x) for x in p.split('/')]
        if len(parts)!=6:return self._send(400,{'error':'bad'})
        _,_,_,kind,name,fname=parts
        if kind not in KINDS or not re.fullmatch(r'\d{8}-\d{6}',name) or not re.fullmatch(r'(trial|run)-\d\d\.mp4',fname):return self._send(400,{'error':'bad'})
        f=KINDS[kind]/name/fname
        if not f.exists():return self._send(404,{'error':'not found'})
        size=f.stat().st_size;rng=self.headers.get('Range');start,end=0,size-1;code=200
        if rng:
            m=re.match(r'bytes=(\d*)-(\d*)',rng)
            if m:
                start=int(m.group(1)) if m.group(1) else 0;end=int(m.group(2)) if m.group(2) else size-1;code=206
        end=min(end,size-1);length=end-start+1
        self.send_response(code);self.send_header('Content-Type','video/mp4');self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(length))
        if code==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with open(f,'rb') as fh:
            fh.seek(start);left=length
            while left>0:
                chunk=fh.read(min(65536,left))
                if not chunk:break
                self.wfile.write(chunk);left-=len(chunk)
    def do_POST(self):
        if not self._host_ok() or self.headers.get('X-A1Z-Token')!=TOKEN:return self._send(403,{'error':'token'})
        n=int(self.headers.get('Content-Length') or 0)
        try:p=json.loads(self.rfile.read(n) or b'{}')
        except ValueError:return self._send(400,{'error':'json'})
        path=urlparse(self.path).path
        try:
            if path=='/api/session/start':ok,msg=start_session(p)
            elif path=='/api/session/key':ok,msg=send_key(str(p.get('key','')))
            elif path=='/api/session/close':ok,msg=close_session()
            elif path=='/api/trial/edit':ok,msg=edit_trial(p)
            elif path=='/api/side/preview':ok,msg=side_preview(bool(p.get('on')))
            elif path=='/api/side/crop':ok,msg=side_crop(p)
            else:return self._send(404,{'error':'not found'})
        except Exception as e:ok,msg=False,repr(e)
        return self._send(200 if ok else 409,{'ok':ok,'message':msg})


def main():
    srv=ThreadingHTTPServer(('127.0.0.1',PORT),H)
    print(f'A1Z_CONSOLE http://127.0.0.1:{PORT}',flush=True)
    srv.serve_forever()

if __name__=='__main__':main()
