// A1Z Console front end. Plain JS, no build step. Every robot action goes through /api/session/key,
// which types the same key into the tmux client window the terminal flow uses.
const TOKEN=document.querySelector('meta[name=a1z-token]').content;
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const S={session:{state:'offline'},tele:null,usb:null,models:[],page:'run',timers:[],lastPane:''};

async function get(p){const r=await fetch(p,{cache:'no-store'});if(!r.ok)throw new Error((await r.json().catch(()=>({}))).error||r.status);return r.json()}
async function post(p,body){
  if(S.session.state==='offline'){toast('Controls are waiting for Jetson data',true);return {ok:false}}
  let r;try{r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json','X-A1Z-Token':TOKEN},body:JSON.stringify(body||{})})}catch(e){toast('Request interrupted; check session state before trying again',true);return {ok:false}}
  if(r.status===403){toast('Page expired, please reload',true);return {ok:false}}
  const j=await r.json().catch(()=>({ok:false,message:'No response'}));if(!j.ok)toast(j.message||'Failed',true);return j}
function toast(msg,bad){const t=$('#statusline');t.textContent=bad?`[ERROR: ${msg}]`:`[${msg}]`;t.className='show'+(bad?' err':'');clearTimeout(t._h);t._h=setTimeout(()=>t.className='',3200)}
function openModal(html){$('#modal-body').innerHTML=html;$('#modal').classList.remove('hidden')}
function closeModal(){$('#modal').classList.add('hidden');$('#modal-body').innerHTML=''}
$('#modal').addEventListener('click',e=>{if(e.target.id==='modal')closeModal()});

const fmtSess=s=>s?`${s.slice(4,6)}-${s.slice(6,8)} ${s.slice(9,11)}:${s.slice(11,13)}`:'—';
const pct=(k,n)=>n?`${Math.round(100*k/n)}%`:'—';
const ci=c=>c?`${Math.round(c[0]*100)}–${Math.round(c[1]*100)}%`:'';
const KIND={teleop:'SpaceMouse',eval:'Eval',hil:'HIL collection',control:'Control test'};
const STATE={teleop_hold:['SpaceMouse · holding','ok','HOLD'],teleop_active:['SpaceMouse · moving','run','MOVE'],offline:['Waiting for data','','—'],none:['No session','','OFF'],starting:['Starting','run','BOOT'],await_ready:['Waiting for on-site confirmation','warn','READY?'],idle:['Ready · gravity assist','ok','IDLE'],
  running:['Running','run','RUN'],human:['Human takeover','warn','HAND'],recording:['Recording demo','run','REC'],ended:['Session ended','','END']};
const LABEL={missed:['Missed','bad'],found_port:['Found port','warn'],in_mouth:['In mouth, no charge','warn'],inserted:['Charging','ok'],charging:['Charging','ok']};

// ---------------- polling ----------------
let sessionPolling=false;
async function pollSession(){
  if(sessionPolling)return;sessionPolling=true;
  const wasOffline=S.session.state==='offline',previousMode=S.session.mode;
  try{
    S.session=await get('/api/session');
    if(wasOffline){S.progs=null;S.models=await get('/api/models').catch(()=>[]);route()}else if(previousMode!==S.session.mode){route()}
  }catch(e){
    S.session={state:'offline',pane:''};S.tele=null;S.usb=null;
    if(!wasOffline)route();
  }finally{sessionPolling=false}
  if(!S.progs&&S.session.state!=='offline')loadPrograms();
  updateChrome();if(S.page==='run'){updateRun();updateRunTele()}
}
async function pollTele(){try{const t=await get('/api/telemetry');S.tele=S.session.state==='offline'?null:t}catch(e){S.tele=null};updateChrome();if(S.page==='run')updateRunTele();if(S.page==='health')renderHealth(true)}
async function pollUsb(){try{const u=await get('/api/usb');S.usb=S.session.state==='offline'?null:u}catch(e){S.usb=null}}
setInterval(pollSession,800);setInterval(pollTele,1000);setInterval(pollUsb,1500);
pollSession();pollTele();pollUsb();get('/api/models').then(m=>S.models=m).catch(()=>{});

// ---------------- programs (what the New session form can launch) ----------------
// The Jetson serves GET /api/programs: {programs:[{id,label,description,inputs:[name],advanced:[name],start_label?,note?}], inputs:{name:{label,type,...}}}.
// A server.py from before 2026-10-10 answers 404; then this built-in copy of the four original programs is used.
const FALLBACK_PROGRAMS={programs:[
  {id:'teleop',label:'SpaceMouse',description:'Move the arm with the SpaceMouse on the Jetson; translation only at startup.',inputs:[],advanced:[],start_label:'Prepare SpaceMouse',note:'Next, confirm on-site readiness before enabling the arm.'},
  {id:'eval',label:'Eval',description:'Run a trained policy on the arm; label each trial after it ends.',inputs:['model'],advanced:['stiffness','ki','ki_max','grip_hold']},
  {id:'hil',label:'HIL collection',description:'HG-DAgger: the policy drives, you take over to correct it.',inputs:['model','epoch'],advanced:['stiffness','ki','ki_max','grip_hold']},
  {id:'control',label:'Control test',description:'Teach-and-return and record-and-replay, no model.',inputs:[],advanced:['stiffness','ki','ki_max','grip_hold']}],
  inputs:{model:{label:'Model',type:'model'},epoch:{label:'HG-DAgger round (epoch)',type:'int',min:1,default:3},
    stiffness:{label:'Stiffness × vendor kp (legacy gains)',type:'number',step:0.1,placeholder:'default dimos gains'},ki:{label:'Integral ki, 1/s',type:'number',step:0.5,placeholder:'2'},
    ki_max:{label:'Integral cap, deg',type:'number',step:0.5,placeholder:'1'},grip_hold:{label:'Grip hold between trials, N·m',type:'number',step:0.5,placeholder:'1.5'}}};
S.progs=null;S.draft={};S.advOpen=false;
const progData=()=>S.progs||FALLBACK_PROGRAMS;
const progList=()=>progData().programs;
const progById=id=>progList().find(p=>p.id===id);
function validPrograms(j){return !!j&&Array.isArray(j.programs)&&j.programs.length>0&&j.programs.every(p=>p&&typeof p.id==='string'&&Array.isArray(p.inputs))}
let programsBusy=false;
async function loadPrograms(){
  if(programsBusy)return;programsBusy=true;let got=null;
  try{
    const r=await fetch('/api/programs',{cache:'no-store'});
    if(r.status===404)got=FALLBACK_PROGRAMS;                       // older server.py: no registry yet
    else if(r.ok){const j=await r.json();got=validPrograms(j)?{programs:j.programs.map(p=>({...p,advanced:p.advanced||[]})),inputs:j.inputs||{}}:FALLBACK_PROGRAMS}
    // anything else (offline 503, 5xx): stay null and let pollSession ask again
  }catch(e){}
  programsBusy=false;
  if(got){S.progs=got;progList().forEach(p=>{if(p.label)KIND[p.id]=p.label});S.ctrlKey='';if(S.page==='run')updateRun()}
}
loadPrograms();

function gripTemp(){const t=S.tele;return t&&t.g_temp_rotor!=null?t.g_temp_rotor:null}
function gripClass(v){if(v==null)return '';const t=S.tele;return v>=t.grip_block_c?'bad':v>=t.grip_warn_c?'warn':'ok'}
function updateChrome(){
  const st=S.session.state||'none',[txt,cls]=STATE[st]||[st,''];
  $('#side-state').textContent=txt;$('#side-dot').className='dot '+(cls==='run'?'run':cls==='ok'?'on':cls==='warn'?'warn':'');
  const g=gripTemp();$('#side-grip').textContent=g==null?'Gripper temp —':`Gripper coil ${g.toFixed(0)}°C${S.tele.live?'':' (last)'}`;
}

// ---------------- router ----------------
const PAGES={/* overview:renderOverview, hidden 2026-10-05 (duplicated by Run / Runs / Report) */run:renderRun,runs:renderRuns,report:renderReport,models:renderModels,data:renderData,health:renderHealth,cameras:renderCameras};
function route(){
  const h=location.hash.replace(/^#\//,'')||'run';const [page,...rest]=h.split('/');
  S.page=PAGES[page]?page:'run';
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.dataset.page===S.page));
  S.timers.forEach(clearInterval);S.timers=[];
  const activeHash=location.hash;
  Promise.resolve(PAGES[S.page](rest)).catch(()=>{
    if(location.hash!==activeHash)return;
    $('#main').innerHTML=`<h1>${({runs:'Runs',report:'Report',models:'Models',data:'Data',health:'Hardware',cameras:'Cameras'})[S.page]||'Console'}</h1><div class="card"><p class="muted">Session data will appear here when available.</p></div>`;
  });
}
window.addEventListener('hashchange',route);

// ---------------- shared widgets (kpi / stat / segs) ----------------
const kpi=(l,v,h,c,unit)=>`<div class="card kpi"><div class="label">${l}</div><div class="value ${c?'c-'+c:''}">${esc(v)}${unit?`<small>${unit}</small>`:''}</div><div class="hint">${esc(h)}</div></div>`;
const stat=(k,v,c,unit)=>`<div class="stat"><span class="k">${k}</span><span class="v ${c?'c-'+c:''}">${v}${unit?`<small>${unit}</small>`:''}</span></div>`;
const segs=(frac,cls,n=20,hero)=>{const f=Math.round(Math.max(0,Math.min(1,frac))*n);const col=cls==='bad'?'var(--accent)':cls==='warn'?'var(--warning)':cls==='ok'?'var(--success)':'var(--text-display)';
  return `<span class="segs${hero?' hero':''}" style="grid-template-columns:repeat(${n},1fr)">${Array.from({length:n},(_,i)=>`<i style="${i<f?'background:'+col:''}"></i>`).join('')}</span>`};
/* Overview hidden 2026-10-05: everything on it is duplicated by Run / Runs / Report. Kept for reference.
async function renderOverview(){
  $('#main').innerHTML=`<div class="eyebrow">01 / Overview</div><h1>Overview</h1><p class="sub">A1Z single arm · USB insertion · on-device inference on the Jetson</p>
  <div class="hero-split" id="ov-hero"></div>
  <div class="grid g2"><div class="card"><div class="section-title"><h2>Recent runs</h2><a href="#/runs">All →</a></div><div id="ov-runs"></div></div>
  <div class="card"><div class="section-title"><h2>Models under current control settings</h2><a href="#/report">Report →</a></div><div id="ov-report"></div></div></div>`;
  const kp=()=>{const st=S.session.state||'none',[txt,,code]=STATE[st]||[st,'',st];const g=gripTemp();const t=S.tele||{};
    $('#ov-hero').innerHTML=`<div><div class="label">Session</div><div class="hero ${st==='running'?'accent':''}">${code}</div>
      <div class="hero-desc">${txt}${S.session.model?' · '+esc(S.session.model):''}</div><div class="hero-meta">${S.session.mode?KIND[S.session.mode]:'Start a session on the Run page'}</div></div>
      <div>${stat('Gripper coil',g==null?'—':g.toFixed(0),gripClass(g),g==null?'':'°C'+(t.live?'':' · last'))}
      ${stat('Hottest joint',t.temp_rotor?Math.max(...t.temp_rotor).toFixed(0):'—','',t.temp_rotor?'°C':'')}
      ${stat('USB',S.usb?.connected==null?'—':S.usb.connected?'Detected':'Not plugged in',S.usb?.connected?'ok':'')}
      ${stat('Control loop',t.hz?Math.round(t.hz):'—','',t.hz?'Hz':'')}</div>`};
  kp();S.timers.push(setInterval(kp,1000));
  try{const runs=await get('/api/runs');$('#ov-runs').innerHTML=runsTable(runs.slice(0,6),true)}catch(e){$('#ov-runs').textContent=e}
  try{const rep=(await get('/api/report')).filter(r=>r.ki>0&&r.ki_version!=='v1');$('#ov-report').innerHTML=rep.length?reportTable(rep,true):'<div class="empty-state">No evals with integral compensation yet</div>'}catch(e){}
}
*/
// ---------------- run console ----------------
let pref={};try{pref=JSON.parse(localStorage.getItem('a1z-run')||'{}')}catch(e){}
if(!pref||typeof pref!=='object'||Array.isArray(pref))pref={};
// 2026-10-10: stiffness / ki / cap / grip hold are no longer remembered form fields (server defaults apply; overrides are one-off).
// Drop the stale values and move the remembered model / epoch under pref.inputs. Earlier 2026-10-09 cleanup is subsumed.
if(pref.v!==3){
  const kept={};for(const k of ['model','epoch'])if(pref[k]!=null&&pref[k]!=='')kept[k]=pref[k];
  for(const k of ['stiffness','ki','ki_max','grip_hold','model','epoch'])delete pref[k];
  pref.inputs=kept;pref.v=3;try{localStorage.setItem('a1z-run',JSON.stringify(pref))}catch(e){}
}
if(!pref.inputs||typeof pref.inputs!=='object')pref.inputs={};
function savePref(){try{localStorage.setItem('a1z-run',JSON.stringify(pref))}catch(e){}}
function renderRun(){
  $('#main').innerHTML=`<div class="eyebrow">01 / Run</div><h1>Run</h1><div class="bar" id="run-chips" style="margin-top:16px"></div>
  <div class="runlayout">
    <div class="stack">
      <div class="grid g2 cams">
        <div class="live"><img id="live-img" alt=""><div class="ov" id="live-ov"></div><div class="corner">Wrist · D405</div><div class="empty" id="live-empty"><span>No signal</span><span style="color:#444">While a session runs</span></div></div>
        <a class="live" href="#/cameras" title="Side camera setup"><img id="run-side" alt=""><div class="corner">Side · D435i</div><div class="empty" id="run-side-empty"><span>No signal</span><span style="color:#444">Start it on the Cameras page</span></div></a>
      </div>
      <div class="card tight"><div class="grid g2"><div><h2>Gripper tip · base frame</h2><div id="tip"></div>
        <div class="faint" style="font-size:12px;margin-top:12px">x = hub long side, y = short side (measured 10-05)</div></div>
        <div><h2>Temperature · coil</h2><div id="temps"></div></div></div></div>
      <div class="card tight"><div class="section-title"><h2>This session</h2><span class="eyebrow" id="cur-sess"></span></div><div id="cur-trials" class="muted">—</div></div>
    </div>
    <div class="stack"><div id="state-hero"></div><div id="ctrl"></div><div class="card tight"><h2>Client output</h2><div class="term" id="term"></div>
      <div class="faint" style="font-size:12px;margin-top:12px">Buttons send the same keys as the terminal · fallback: <span class="mono">ssh -t jetson-direct tmux attach -t eval</span></div></div></div>
  </div>`;
  const img=$('#live-img');
  // Shown whenever a1z_robot.py is writing frames (console sessions and HIL-SERL's actor alike), judged by the frame age
  // header rather than by the console session state; frames older than 2 s count as no signal.
  let liveBusy=false;
  S.timers.push(setInterval(async()=>{if(S.session.mode==='teleop'||S.session.state==='offline'||liveBusy)return;liveBusy=true;
    try{const r=await fetch('/api/live.jpg?t='+Date.now(),{cache:'no-store'});const age=parseFloat(r.headers.get('X-Frame-Age'));
      if(!r.ok||!(age<2)){$('#live-empty').style.display='flex';return}
      const u=URL.createObjectURL(await r.blob()),old=img.src;img.src=u;if(old.startsWith('blob:'))URL.revokeObjectURL(old);$('#live-empty').style.display='none'}
    catch(e){$('#live-empty').style.display='flex'}finally{liveBusy=false}},250));
  const side=$('#run-side');let sideLive=false;
  const sideAge=()=>get('/api/side').then(st=>{sideLive=st.age!=null&&st.age<3;if(!sideLive)$('#run-side-empty').style.display='flex'}).catch(()=>{sideLive=false});
  sideAge();S.timers.push(setInterval(sideAge,2000));
  S.timers.push(setInterval(()=>{if(!sideLive)return;const n=new Image();n.onload=()=>{side.src=n.src;$('#run-side-empty').style.display='none'};n.src='/api/side.jpg?t='+Date.now()},300));
  S.ctrlKey='';updateRun();updateRunTele();refreshTrials();S.timers.push(setInterval(refreshTrials,3000));
}
function chipsHtml(){const st=S.session.state||'none',[txt,cls]=STATE[st]||[st,''];const g=gripTemp();
  return (S.session.mode?`<span class="chip">${KIND[S.session.mode]}</span>`:'')+
  (S.session.model?`<span class="chip">Model ${esc(S.session.model)}</span>`:'')+
  `<span class="chip ${gripClass(g)}">Gripper ${g==null?'—':g.toFixed(0)+'°C'}</span>`+
  `<span class="chip ${S.usb?.connected?'ok':''}">USB ${S.usb?.connected==null?'—':S.usb.connected?'Detected':'Not plugged in'}</span>`}
function updateRun(){
  if(!$('#run-chips'))return;$('#run-chips').innerHTML=chipsHtml();
  const st=S.session.state||'none',[txt,,code]=STATE[st]||[st,'',st];
  $('#state-hero').innerHTML=`<div style="padding:8px 4px 8px"><div class="label">Status</div><div class="hero ${st==='running'||st==='recording'?'accent':''}" style="font-size:72px">${code}</div><div class="hero-meta">${txt}</div></div>`;
  const pane=S.session.pane||'';if(pane!==S.lastPane){S.lastPane=pane;const t=$('#term');t.textContent=pane||'(no session)';t.scrollTop=t.scrollHeight}
  const key=`${S.session.state}|${S.session.mode}|${gripClass(gripTemp())}`;if(key===S.ctrlKey)return;S.ctrlKey=key;renderCtrl();
}
function renderCtrl(){
  const st=S.session.state||'none',mode=S.session.mode,c=$('#ctrl');const hot=gripClass(gripTemp())==='bad';
  if(st==='offline'){c.innerHTML='<div class="card"><h2>Console</h2><p class="muted">Controls will appear when Jetson data is available. This page updates automatically.</p></div>';$('#live-img').removeAttribute('src');$('#live-empty').style.display='flex';return}
  if(st==='none'){c.innerHTML=newSessionForm();bindForm();return}
  if(st==='starting'){c.innerHTML=`<div class="card"><h2>Starting…</h2><p class="muted">Loading the model and the arm process, about 10 s.</p></div>`;return}
  if(st==='await_ready'&&mode==='teleop'){c.innerHTML=`<div class="card ready-card"><h2>SpaceMouse teleop</h2><p>Support the arm, clear its movement area and stay beside it. Start holds the current pose with vendor settings. The gripper stays unchanged.</p><p>First test: translation only. Hold the left mouse button to move; release to hold. Right button pauses.</p><button class="btn primary big wide" onclick="ready()">I'm on site — start position hold</button><button class="btn wide" style="margin-top:8px" onclick="cancelReady()">Cancel</button></div>`;return}
  if(st==='await_ready'){c.innerHTML=`<div class="card ready-card"><h2>On-site confirmation</h2><p>Confirm the arm is on its support, the area is clear and you are on site. After you click, the motors enter the vendor gravity assist (same as during recording).</p>
    <button class="btn primary big wide" onclick="ready()">Arm supported, send READY</button>
    <button class="btn wide" style="margin-top:8px" onclick="cancelReady()">Cancel</button></div>`;return}
  if(st==='ended'){c.innerHTML=`<div class="card"><h2>Session ended</h2><p class="muted">Data saved. Close it to start a new session.</p><button class="btn primary wide" onclick="closeSess()">Close session</button></div>`;return}
  if(mode==='teleop'){
    c.innerHTML=`<div class="card"><h2>SpaceMouse</h2><p>Hold the left button and push the cap to move. Release to hold. The right button pauses control.</p>
      <button class="btn danger big wide" onclick="key('s')">Pause · hold position <kbd>Esc</kbd></button>
      <div class="btns" style="margin-top:12px"><button class="btn" onclick="key('g')">Resume mouse</button><button class="btn" onclick="key('r')">Toggle rotation</button></div>
      <p class="muted" id="mouse-mode">Translation only at startup</p><p class="faint">25 mm/s translation and 0.25 rad/s rotation at full deflection. Directions are relative to the arm base; verify each axis in free space first.</p>
      <button class="btn wide" onclick="quitClient()">Arm supported — disable and quit</button></div>`;return;
  }
  const running=st==='running'||st==='recording',human=st==='human',idle=st==='idle';
  let main='';
  if(mode==='control'){
    main=`<div class="btns">${b('t','Teach (when inserted)','t',idle)}${b('g','Go to taught pose A1','g',idle&&!hot,'primary')}${b('r',st==='recording'?'Stop recording':'Record demo','r',idle||st==='recording')}${b('p','Replay A2','p',idle&&!hot,'primary')}${b('w','Show position','w',idle)}</div>`;
  }else if(mode==='hil'){
    main=`<div class="btns">${b('g','Start trial','G',idle&&!hot,'primary big')}${b(' ',human?'Hand back to model':'Take over','Space',running||human,'big')}${b('x','End trial','X',human)}${b('d','Discard last trial','D',idle)}</div>`;
  }else{
    main=`<div class="btns">${b('g','Start trial','G',idle&&!hot,'primary big')}</div>`;
  }
  c.innerHTML=`<div class="card"><div class="section-title"><h2>${KIND[mode]||'Session'} · Controls</h2><span class="eyebrow">${esc(S.session.model||'')}</span></div>
    ${hot?`<div class="alert bad">Gripper coil above ${S.tele.grip_block_c}°C: let it cool down, start buttons are locked.</div>`:gripClass(gripTemp())==='warn'?`<div class="alert warn">Gripper is warm (≥${S.tele.grip_warn_c}°C): rest after this trial.</div>`:''}
    ${main}
    <button class="btn danger big wide" style="margin-top:16px" onclick="key('s')" ${running?'':'disabled'}>Stop trial <kbd>Esc</kbd></button>
    <div class="label" style="margin:28px 0 12px">After the trial</div>
    <div class="btns">${mode!=='hil'?b('i','Record reference pose (after hand insert)','I',idle):''}
      ${b('1','1 Missed','1',idle)}${b('2',mode==='control'?'2 In mouth, no charge':'2 Found port','2',idle)}${b('3','3 Charging','3',idle)}</div>
    <div class="btns" style="margin-top:12px">${b(' ','Gripper open/close','',idle&&mode!=='hil'||idle)}<button class="btn" onclick="quitClient()" ${idle?'':'disabled'}>Quit session</button></div>
    ${mode==='eval'?`<div class="faint" style="font-size:12px;margin-top:20px">Each trial: start → stop → insert by hand (until it charges) → record reference pose → label 1/2/3 → unplug and return to start.</div>`:''}
  </div>`;
}
const b=(k,label,kb,en,cls='')=>`<button class="btn ${cls}" onclick="key('${k===' '?'space':k}')" ${en?'':'disabled'}>${label}${kb?` <kbd>${kb}</kbd>`:''}</button>`;
async function key(k){const r=await post('/api/session/key',{key:k==='space'?' ':k});if(r.ok)setTimeout(pollSession,150)}
async function ready(){if(!confirm('Arm on its support and someone on site?'))return;const r=await post('/api/session/key',{key:'READY'});if(r.ok)toast('READY sent')}
async function quitClient(){if(!confirm('Quit this session? (make sure the arm is on its support)'))return;await post('/api/session/key',{key:'q'});setTimeout(()=>post('/api/session/key',{key:'q'}),300)}
async function cancelReady(){const r=await post('/api/session/key',{key:'CANCEL'});if(r.ok)toast('Cancelled. Click Close session to clean up')}
async function closeSess(){const r=await post('/api/session/close');if(r.ok){toast('Session closed');S.ctrlKey=''}}
function curProgram(){return progById(pref.mode)||progById('eval')||progList()[0]}
function pickProgram(id){pref.mode=id;savePref();S.ctrlKey='';renderCtrl()}
function fieldHtml(n,advanced){
  const d=(progData().inputs||{})[n]||{label:n,type:'text'},id='f-'+n;
  const v=S.draft[n]!==undefined?S.draft[n]:advanced?'':(pref.inputs[n]??d.default??'');
  if(d.type==='model'){const models=S.models.length?S.models:[{name:'060000'}];
    return `<div class="field"><label for="${id}">${esc(d.label)}</label><select id="${id}" data-in="${esc(n)}">${models.map(x=>`<option value="${esc(x.name)}" ${v===x.name?'selected':''}>${esc(x.name)}${x.description?' — '+esc(x.description):''}</option>`).join('')}</select></div>`}
  if(d.type==='select'&&Array.isArray(d.options))
    return `<div class="field"><label for="${id}">${esc(d.label)}</label><select id="${id}" data-in="${esc(n)}">${d.options.map(o=>`<option value="${esc(o)}" ${String(v)===String(o)?'selected':''}>${esc(o)}</option>`).join('')}</select></div>`;
  const num=d.type==='int'||d.type==='number';
  return `<div class="field"><label for="${id}">${esc(d.label)}</label><input id="${id}" data-in="${esc(n)}" type="${num?'number':'text'}" ${d.min!=null?`min="${+d.min}"`:''} ${d.max!=null?`max="${+d.max}"`:''} ${num?`step="${d.step||(d.type==='int'?1:'any')}"`:''} placeholder="${esc(d.placeholder||(advanced?'default':''))}" value="${esc(v)}"></div>`}
function newSessionForm(){
  if(!S.progs)return `<div class="card"><h2>New session</h2><p class="muted">Loading programs…</p></div>`;
  const prog=curProgram(),adv=prog.advanced||[];
  return `<div class="card"><h2>New session</h2>
  <div class="programs" role="radiogroup" aria-label="Program">${progList().map(p=>`<button class="prog ${p.id===prog.id?'on':''}" role="radio" aria-checked="${p.id===prog.id}" data-prog="${esc(p.id)}" onclick="pickProgram(this.dataset.prog)"><span class="nm">${esc(p.label||p.id)}</span>${p.description?`<span class="ds">${esc(p.description)}</span>`:''}</button>`).join('')}</div>
  ${prog.inputs.map(n=>fieldHtml(n,false)).join('')}
  ${adv.length?`<details class="adv" ${S.advOpen?'open':''} ontoggle="S.advOpen=this.open"><summary>Advanced</summary><p class="faint" style="font-size:12px;margin:12px 0 0">Control settings use the defaults unless you fill one in here. Overrides apply to this start only and are not remembered.</p><div class="row-adv">${adv.map(n=>fieldHtml(n,true)).join('')}</div></details>`:''}
  <button class="btn primary big wide" id="f-start">${esc(prog.start_label||'Start session')}</button>
  <p class="faint" style="font-size:12px;margin:10px 0 0">${esc(prog.note||'Starting only loads the program; the motors do not move. Next you confirm READY on site.')}</p></div>`}
document.addEventListener('input',e=>{const n=e.target&&e.target.dataset&&e.target.dataset.in;if(n&&e.target.closest('#ctrl'))S.draft[n]=e.target.value});
function bindForm(){const btn=$('#f-start');if(!btn)return;btn.onclick=async()=>{
  const prog=curProgram(),p={mode:prog.id};
  for(const n of prog.inputs){const el=$('#f-'+n);if(el)p[n]=el.value}
  for(const n of prog.advanced||[]){const el=$('#f-'+n);if(el&&el.value.trim()!=='')p[n]=el.value.trim()}  // only filled-in overrides are sent; the server applies its defaults otherwise
  pref.mode=prog.id;for(const n of prog.inputs)if(p[n]!==undefined&&p[n]!=='')pref.inputs[n]=p[n];savePref();
  btn.disabled=true;
  const r=await post('/api/session/start',p);        // never retried automatically
  if(r.ok){toast(r.message);for(const n of prog.advanced||[])delete S.draft[n];S.session={...S.session,state:'starting',mode:prog.id};S.ctrlKey='';renderCtrl()}else btn.disabled=false}}
function updateRunTele(){
  const t=S.tele;if(!$('#tip'))return;if(!t){$('#tip').textContent='—';$('#temps').textContent='—';return}
  $('#tip').innerHTML=t.tip_mm&&t.live?['x','y','z'].map((a,i)=>stat(a,t.tip_mm[i].toFixed(1),'','mm')).join(''):'<div class="empty-state" style="padding:24px 0">—</div>';
  const rows=[];if(t.g_temp_rotor!=null)rows.push(tbar('Grip',t.g_temp_rotor,t.grip_warn_c,t.grip_block_c));
  (t.temp_rotor||[]).forEach((v,i)=>rows.push(tbar('J'+(i+1),v,75,90)));
  $('#temps').innerHTML=rows.join('')+(t.live?'':`<div class="faint" style="font-size:12px;margin-top:8px">Last session's readings · ${fmtSess(t.session)}</div>`);
  if(S.session.mode==='teleop'){
    const m=t.spacemouse||{};const box=$('#live-empty');box.style.display='flex';
    box.innerHTML=`<span>SpaceMouse · ${m.rotation?'6 axes':'Translation only'}</span><div style="width:75%;margin-top:20px">${['X','Y','Z','RX','RY','RZ'].map((name,i)=>stat(name,Number(m.axes?.[i]||0).toFixed(2),'')).join('')}</div><span style="margin-top:20px">Left ${m.buttons&1?'pressed':'released'} · Right ${m.buttons&2?'pressed':'released'}</span>`;
    $('.live .corner').textContent='SpaceMouse Compact';$('#live-ov').innerHTML='';
    const label=$('#mouse-mode');if(label)label.textContent=(m.paused?'Paused · click Resume mouse. ':'')+(m.rotation?'Translation + rotation':'Translation only')+(m.error?' · '+m.error:'');
    return;
  }
  const ov=$('#live-ov');if(ov)ov.innerHTML=t.live&&t.mode===1?'<span class="chip run">Engaged</span>':t.live?'<span class="chip">Float</span>':'';
}
const tbar=(n,v,w,blk)=>{const cls=v>=blk?'bad':v>=w?'warn':'';
  return `<div class="tbar"><span class="nm">${n}</span>${segs((v-25)/(blk-25),cls)}<span class="v ${cls?'c-'+cls:''}">${v.toFixed(0)}°</span></div>`};
async function refreshTrials(){
  const t=S.tele;if(!t||!t.session||!$('#cur-trials'))return;
  try{const d=await get(`/api/runs/${t.kind}/${t.session}`);$('#cur-sess').textContent=`${KIND[t.kind]} · ${fmtSess(t.session)}`;
    $('#cur-trials').innerHTML=d.trials.length?trialsTable(d,true):'<span class="muted">No completed trials yet</span>'}catch(e){$('#cur-trials').textContent='No completed trials yet'}
}
document.addEventListener('keydown',e=>{
  if(S.page!=='run'||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.metaKey||e.ctrlKey)return;
  const st=S.session.state,k=e.key.toLowerCase();
  if(S.session.mode==='teleop'&&['teleop_hold','teleop_active'].includes(st)&&['escape','s','g','r'].includes(k)){key(k==='escape'?'s':k);e.preventDefault();return}
  if(k==='escape'&&(st==='running'||st==='recording')){key('s');e.preventDefault();return}
  if(k===' '&&S.session.mode==='hil'&&(st==='running'||st==='human')){key(' ');e.preventDefault();return}
  if(st!=='idle')return;
  if(['g','i','1','2','3','x','d','t','r','p','w'].includes(k)){key(k);e.preventDefault()}
});

// ---------------- runs ----------------
function labelCounts(r){const l=r.labels||{};const ok=(l.inserted||0)+(l.charging||0);const f=(l.found_port||0)+(l.in_mouth||0);
  return `<span class="mono"><span class="cnt c-ok">${ok}</span><span class="faint"> · </span><span class="cnt c-warn">${f}</span><span class="faint"> · </span><span class="cnt ${l.missed?'':'faint'}">${l.missed||0}</span></span>`}
function runsTable(runs,compact){
  if(!runs.length)return '<div class="empty-state">No runs yet</div>';
  return `<div class="tbl"><table><tr><th>Time</th><th>Type</th><th>Model</th>${compact?'':'<th class="num">Stiffness</th><th class="num">ki</th>'}<th class="num">Valid/total</th><th>Charging · Found · Missed</th></tr>
  ${runs.map(r=>`<tr class="click" onclick="location.hash='#/runs/${r.kind}/${r.session}'"><td class="mono">${fmtSess(r.session)}</td><td class="nw">${KIND[r.kind]}</td><td class="nw">${esc(r.model||'—')}${r.epoch?` <span class="faint">e${r.epoch}</span>`:''}</td>
  ${compact?'':`<td class="num">${r.stiffness??'—'}</td><td class="num">${r.ki||'off'}</td>`}<td class="num">${r.valid}/${r.trials}</td><td>${labelCounts(r)}</td></tr>`).join('')}</table></div>`}
async function renderRuns(rest){
  if(rest.length===2)return renderRunDetail(rest[0],rest[1]);
  $('#main').innerHTML=`<div class="eyebrow">02 / Runs</div><h1>Runs</h1><p class="sub">All eval, HIL collection and control test sessions. Open one to see each trial and its video, fix labels or exclude mistakes.</p>
  <div class="seg" style="max-width:420px" id="rk">${['all','eval','hil','control'].map(k=>`<button data-k="${k}">${k==='all'?'All':KIND[k]}</button>`).join('')}</div><div class="card" id="runs">Loading…</div>`;
  const runs=await get('/api/runs');const draw=k=>{if(!$('#runs'))return;document.querySelectorAll('#rk button').forEach(x=>x.classList.toggle('on',x.dataset.k===k));$('#runs').innerHTML=runsTable(k==='all'?runs:runs.filter(r=>r.kind===k))};
  document.querySelectorAll('#rk button').forEach(x=>x.onclick=()=>draw(x.dataset.k));draw('all');
}
function errCell(t){if(!t.err)return '<span class="faint">—</span>';const e=t.err;return `<b>${(t.err_xy??Math.hypot(e[0],e[1])).toFixed(1)}</b> <span class="faint">x${e[0]>=0?'+':''}${e[0].toFixed(1)} y${e[1]>=0?'+':''}${e[1].toFixed(1)} z${e[2]>=0?'+':''}${e[2].toFixed(1)}</span>`}
function trialsTable(d,compact){
  const kind=d.kind;
  return `<div class="tbl"><table><tr><th>#</th><th>Result</th><th>USB</th><th>Off-port mm · horizontal (x y z)</th>${compact?'':'<th class="num">Duration</th><th>End reason</th><th>Video</th><th>Exclude</th>'}</tr>
  ${d.trials.map(t=>{const [ln,lc]=LABEL[t.label]||['Unlabelled',''];
    return `<tr class="${t.excluded?'dim':''}"><td class="mono">${t.n??''}${t.kind?` <span class="faint">${esc(t.kind)}</span>`:''}</td>
    <td>${compact?`<span class="${lc?'c-'+lc:''}">${ln}</span>`:`<select class="lbl" onchange="editTrial('${kind}','${d.session}','${t.file}',{label:this.value})">
      <option value="" ${!t.label?'selected':''} disabled>Unlabelled</option><option value="1" ${t.label==='missed'?'selected':''}>Missed</option>
      <option value="2" ${['found_port','in_mouth'].includes(t.label)?'selected':''}>${kind==='control'?'In mouth, no charge':'Found port'}</option><option value="3" ${['inserted','charging'].includes(t.label)?'selected':''}>Charging</option></select>`}</td>
    <td>${t.usb?'<span class="chip ok">USB</span>':''}</td><td class="mono">${errCell(t)}</td>
    ${compact?'':`<td class="num">${t.duration_s??''}s</td><td class="muted">${esc(t.end_reason||'')}${t.interventions?` · ${t.interventions} takeovers`:''}</td>
    <td>${t.video?`<button class="btn sm" onclick="playVideo('${kind}','${d.session}','${t.video}')">Play</button>`:''}</td>
    <td><input type="checkbox" ${t.excluded?'checked':''} onchange="editTrial('${kind}','${d.session}','${t.file}',{excluded:this.checked})"></td>`}</tr>`}).join('')}</table></div>`}
async function renderRunDetail(kind,name){
  $('#main').innerHTML='Loading…';const d=await get(`/api/runs/${kind}/${name}`);
  const errs=d.trials.filter(t=>!t.excluded&&t.err_xy!=null).map(t=>t.err_xy).sort((a,b)=>a-b);
  $('#main').innerHTML=`<a href="#/runs" class="eyebrow" style="text-decoration:none">← 02 / Runs</a><h1>${KIND[kind]} · <span class="mono">${fmtSess(name)}</span></h1>
  <p class="sub">${esc(d.model||'')} · stiffness ${d.stiffness??'—'} · integral ${d.ki||'off'} · valid ${d.valid}/${d.trials.length} trials</p>
  <div class="grid g4" style="margin-bottom:24px">${kpi('Charging',`${(d.labels.inserted||0)+(d.labels.charging||0)}`,`USB detected ${d.usb}×`,'ok')}${kpi('Found port (incl. charging)',`${(d.labels.found_port||0)+(d.labels.in_mouth||0)+(d.labels.inserted||0)+(d.labels.charging||0)}`,'')}
  ${kpi('Missed',`${d.labels.missed||0}`,'')}${kpi('Off-port error · median',errs.length?errs[Math.floor(errs.length/2)].toFixed(1):'—',`${errs.length} trials with a reference pose`,'',errs.length?'mm':'')}</div>
  <div class="card">${trialsTable(d,false)}</div>
  <p class="faint" style="font-size:12px">Off-port = gripper tip position when the model stops − the position recorded with i after you inserted by hand. Exclusions and label edits are written back to the trial JSON (with edit history); raw logs and videos are untouched.</p>`;
}
async function editTrial(kind,session,file,patch){const r=await post('/api/trial/edit',{kind,session,file,...patch});if(r.ok){toast('Saved');if(S.page==='runs')renderRunDetail(kind,session)}}
function playVideo(kind,s,f){openModal(`<h2>${KIND[kind]} ${fmtSess(s)} · ${esc(f)}</h2><video src="/api/video/${kind}/${s}/${f}" controls autoplay playsinline></video>`)}

// ---------------- report ----------------
function reportTable(rows,compact){
  return `<div class="tbl"><table><tr><th>Model</th>${compact?'':'<th class="num">Stiffness</th><th class="num">ki</th>'}<th class="num">Trials</th><th>Found port · 95% CI</th><th>Charging · 95% CI</th><th>Off-port mm · median</th></tr>
  ${rows.map(r=>`<tr><td>${esc(r.model)}${r.smooth?' <span class="chip" title="Smooth tracking: acceleration limit, 1 s soft start, velocity feed-forward">Smooth</span>':''}${compact?` <span class="faint">stiffness ${r.stiffness}</span>`:''}</td>${compact?'':`<td class="num">${r.stiffness}</td><td class="num">${r.ki?r.ki+(r.ki_version==='v1'?' <span class="chip warn" title="Early version: winds up during motion and wobbles">v1</span>':''):'off'}</td>`}<td class="num">${r.n}</td>
  <td><span class="rate">${segs(r.labelled?r.found/r.labelled:0,'',10)}<span class="mono">${pct(r.found,r.labelled)}</span></span><span class="faint mono" style="font-size:11px">${r.found}/${r.labelled} · ${ci(r.found_ci)}</span></td>
  <td><span class="rate">${segs(r.n?r.power/r.n:0,'ok',10)}<span class="mono">${pct(r.power,r.n)}</span></span><span class="faint mono" style="font-size:11px">${r.power}/${r.n} · ${ci(r.power_ci)}</span></td>
  <td class="mono">${r.err_median!=null?`${r.err_median.toFixed(1)} <span class="faint">n=${r.err_n}</span>`:'<span class="faint">—</span>'}</td></tr>`).join('')}</table></div>`}
async function renderReport(){
  $('#main').innerHTML=`<div class="eyebrow">03 / Report</div><h1>Report</h1><p class="sub">All eval sessions grouped by model × stiffness × integral (excluded trials not counted). The ranges are 95% confidence intervals: when they overlap a lot, the difference may just be luck.</p><div class="card" id="rep">Loading…</div>
  <div style="margin-top:48px;max-width:720px"><h2>How to read</h2><div class="muted">
  <p>Found port = labelled 2 or 3; charging = labelled 3 or USB detected. Off-port error only counts trials with a reference pose (i pressed).</p>
  <p>Sessions before 2026-10-05 had no integral compensation (ki off) and 2–5 mm control error, so they are not directly comparable with later results. Rows tagged v1 used the first integral version from the morning of 10-05 (it also winds up during motion and wobbles). Compare models only within the same settings.</p></div></div>`;
  const rows=await get('/api/report');$('#rep').innerHTML=rows.length?reportTable(rows,false):'<div class="empty-state">No evals yet</div>';
}

// ---------------- models ----------------
async function renderModels(){
  const ms=await get('/api/models');S.models=ms;
  $('#main').innerHTML=`<div class="eyebrow">04 / Models</div><h1>Models</h1><p class="sub">Checkpoints deployable on the Jetson (~/a1z-infer/checkpoints).</p><div class="grid g3">${ms.map(m=>`<div class="card model-card">
    <h3>${esc(m.name)}</h3><div class="desc">${esc(m.description||'')}</div>
    <div class="kv"><div>Steps</div><div>${m.steps??'—'}</div><div>Dataset</div><div>${esc(m.dataset||'—')}</div><div>Init from</div><div>${esc(m.init||'ImageNet backbone')}</div><div>Saved</div><div>${m.saved}</div><div>Size</div><div>${m.size_mb} MB</div></div>
    <div class="btns" style="margin-top:24px">${progList().filter(p=>p.inputs.includes('model')).map((p,i)=>`<button class="btn ${i?'':'primary'}" onclick="useModel('${esc(m.name)}','${esc(p.id)}')">${esc(p.label||p.id)}</button>`).join('')}</div></div>`).join('')}</div>`;
}
function useModel(name,mode){pref.inputs.model=name;delete S.draft.model;pref.mode=mode;savePref();location.hash='#/run';if(S.session.state!=='none')toast('End the current session first to use this model')}

// ---------------- data ----------------
async function renderData(){
  const d=await get('/api/data');
  $('#main').innerHTML=`<div class="eyebrow">05 / Data</div><h1>Data</h1><p class="sub">${esc(d.note)}</p>
  <div class="grid g3" style="margin-bottom:24px">${kpi('Total demos',d.demos.total,'')}${kpi('Usable demos (USB confirmed)',d.demos.usable,'')}${kpi('HIL corrections (kept)',d.hil.reduce((a,h)=>a+h.kept,0),`${d.hil.length} collection sessions`)}</div>
  <div class="card"><h2>HIL collection sessions</h2>${d.hil.length?`<div class="tbl"><table><tr><th>Time</th><th>Epoch</th><th>Model</th><th class="num">Trials</th><th class="num">Takeovers</th><th class="num">Human time</th><th class="num">Exportable</th><th class="num">Discarded</th></tr>
  ${d.hil.map(h=>`<tr class="click" onclick="location.hash='#/runs/hil/${h.session}'"><td>${fmtSess(h.session)}</td><td>e${h.epoch??'?'}</td><td>${esc(h.model)}</td><td class="num">${h.trials}</td><td class="num">${h.interventions}</td><td class="num">${h.human_s}s</td><td class="num">${h.kept}</td><td class="num">${h.discarded}</td></tr>`).join('')}</table></div>`:'<div class="empty-state">No HIL sessions</div>'}</div>`;
}

// ---------------- health ----------------
function renderHealth(update){
  const t=S.tele;
  if(!update||!$('#h-chart'))$('#main').innerHTML=`<div class="eyebrow">06 / Hardware</div><h1>Hardware</h1><p class="sub">From the latest session's flight recorder (50 Hz). The gripper keeps applying force while it holds the plug, which is the main source of heat.</p>
    <div class="grid g4" id="h-kpi"></div><div class="grid g2" style="margin-top:16px"><div class="card"><h2>Gripper temperature (last 15 min)</h2><svg id="h-chart" class="spark" viewBox="0 0 600 120" preserveAspectRatio="none"></svg>
    <div class="legend"><span><i style="border-color:var(--text-display)"></i>Gripper coil</span><span><i style="border-color:var(--text-display);opacity:.55"></i>Gripper driver</span><span><i style="border-color:var(--text-display);opacity:.25;border-top-style:dotted"></i>Hottest joint</span><span class="faint">${t?.grip_warn_c??80}° warn · ${t?.grip_block_c??88}° lock</span></div></div>
    <div class="card"><h2>Motors</h2><div id="h-temps"></div></div></div>`;
  if(!t||!t.session){$('#h-kpi').innerHTML='<div class="empty-state">No session data yet</div>';return}
  const g=t.g_temp_rotor;
  $('#h-kpi').innerHTML=kpi('Gripper coil',g==null?'—':g.toFixed(0),t.live?'Live':'Last session reading',gripClass(g),'°C')+kpi('Gripper torque',t.g_tau==null?'—':t.g_tau.toFixed(2),'Sustained while holding the plug','','N·m')+
    kpi('Vendor control loop',t.hz?Math.round(t.hz):'—','Should be ~250 Hz','','Hz')+kpi('Source',fmtSess(t.session),KIND[t.kind]+(t.live?' · running':''));
  $('#h-temps').innerHTML=(g!=null?tbar('Grip',g,t.grip_warn_c,t.grip_block_c):'')+(t.temp_rotor||[]).map((v,i)=>tbar('J'+(i+1),v,75,90)).join('')+
    `<div class="faint" style="font-size:12px;margin-top:8px">Joint limits (vendor): coil 75°C warning / 90°C e-stop; driver 70 / 85°C. Gripper limits are this console's conservative settings.</div>`;
  const H=(t.history||[]).filter(h=>h.g_rotor!=null);if(H.length<2)return;
  const t0=H[0].t,t1=H[H.length-1].t||1,ys=[30,100];const X=v=>((v-t0)/(t1-t0||1))*600,Y=v=>120-(v-ys[0])/(ys[1]-ys[0])*120;
  const line=(k,c)=>`<polyline fill="none" stroke="${c}" stroke-width="1.5" vector-effect="non-scaling-stroke" points="${H.filter(h=>h[k]!=null).map(h=>`${X(h.t).toFixed(1)},${Y(h[k]).toFixed(1)}`).join(' ')}"/>`;
  $('#h-chart').innerHTML=[40,60].map(v=>`<line x1="0" x2="600" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--border)" stroke-width="1"/>`).join('')+
    `<line x1="0" x2="600" y1="${Y(t.grip_warn_c)}" y2="${Y(t.grip_warn_c)}" stroke="var(--warning)" stroke-dasharray="3 4"/><line x1="0" x2="600" y1="${Y(t.grip_block_c)}" y2="${Y(t.grip_block_c)}" stroke="var(--accent)" stroke-dasharray="3 4"/>`+
    line('j_max','var(--text-display)" stroke-opacity=".25" stroke-dasharray="2 3')+line('g_mos','var(--text-display)" stroke-opacity=".55')+line('g_rotor','var(--text-display)');
}

// ---------------- cameras (HIL-SERL side and wrist crops) ----------------
// Side: held by actor.py while HIL-SERL runs, or by side_preview.py --headless (tmux "sidecam") started here; either one
// publishes /dev/shm/a1z_side.jpg. Wrist: a1z_robot.py publishes /dev/shm/a1z_live.jpg while any robot session runs;
// the last frame stays usable for framing because the camera rides on the arm. Each square is the crop the policy (and
// for the side, the reward classifier) sees, resized to 128x128 (paper: crop to the area of interest, then 128x128).
// Save writes cameras.json (crop_side / crop), which actor.py uses when --crop-side / --crop are not given.
const CAM={crop:null,dirty:false,st:null,view:(()=>{try{return localStorage.getItem('a1z-cam-view')||'side'}catch(e){return 'side'}})()};
const CAMV={side:{key:'crop_side',img:'/api/side.jpg',label:'Side · D435i',flag:'--crop-side'},wrist:{key:'crop',img:'/api/live.jpg',label:'Wrist · D405',flag:'--crop'}};
function renderCameras(){
  $('#main').innerHTML=`<div class="eyebrow">07 / Cameras</div><h1>Cameras</h1>
  <div class="bar" style="margin-top:16px;gap:8px">${Object.entries(CAMV).map(([k,v])=>`<button class="btn ${k===CAM.view?'primary':''}" onclick="camView('${k}')">${v.label}</button>`).join('')}</div>
  <div class="bar" id="cam-chips" style="margin-top:12px"></div>
  <div class="runlayout">
    <div class="stack">
      <div class="live cam" id="cam-box"><img id="cam-img" alt=""><div class="thirds"></div><div class="crop" id="cam-crop"><span>Crop · 128×128</span></div>
        <div class="corner" id="cam-name">${CAMV[CAM.view].label}</div><div class="empty" id="cam-empty"><span>No signal</span><span style="color:#444" id="cam-empty-sub"></span></div></div>
      <div class="faint" style="font-size:12px">Drag the square, size it with the slider, then save. Keep the plug tip, the target port and some hub around it inside the square at every start position and insertion depth.</div>
    </div>
    <div class="stack">
      <div class="card tight"><h2 id="cam-title"></h2><div id="cam-ctrl"></div></div>
      <div class="card tight"><h2>What the network sees (128×128)</h2><canvas id="cam-prev" width="128" height="128" style="width:256px;height:256px;max-width:100%;image-rendering:pixelated;background:#000;border-radius:8px"></canvas></div>
      <div class="card tight"><h2>Crop</h2><input type="range" id="cam-size" min="64" max="480" step="8" style="width:100%">
        <div class="mono" id="cam-val" style="margin:12px 0"></div><button class="btn primary wide" onclick="saveCrop()">Save crop</button>
        <div class="faint" id="cam-saved" style="font-size:12px;margin-top:12px"></div></div>
    </div>
  </div>`;
  $('#cam-title').textContent=CAM.view==='side'?'Side camera':'Wrist camera';
  $('#cam-empty-sub').textContent=CAM.view==='side'?'Start the preview, or run the HIL-SERL actor':'Wrist frames appear once any robot session has run';
  CAM.dirty=false;CAM.crop=null;bindCrop();camStatus();S.timers.push(setInterval(camStatus,2000));
  const img=$('#cam-img');
  S.timers.push(setInterval(()=>{const st=CAM.st;if(!st)return;
    if(CAM.view==='side'&&(st.age==null||st.age>3))return;
    if(CAM.view==='wrist'&&st.wrist_age==null)return;
    const n=new Image();n.onload=()=>{img.src=n.src;$('#cam-empty').style.display='none';drawCropPrev(n)};n.src=CAMV[CAM.view].img+'?t='+Date.now()},CAM.view==='side'?250:1000));
}
function camView(v){CAM.view=v;try{localStorage.setItem('a1z-cam-view',v)}catch(e){};renderCameras()}
async function camStatus(){
  let st;try{st=await get('/api/side')}catch(e){st=null}
  CAM.st=st;if(!$('#cam-ctrl'))return;const V=CAMV[CAM.view];
  let ctrl,chips='';
  if(CAM.view==='side'){
    const live=st&&st.age!=null&&st.age<3,owner=st?.actor?'HIL-SERL actor':st?.preview?'Preview':'';
    chips=`<span class="chip ${live?'ok':''}">${live?'Live':'No frames'}</span>`+(owner?`<span class="chip">${owner}</span>`:'')+(st?.camera?`<span class="chip">${esc(st.camera.name)} · ${esc(st.camera.serial)}</span>`:'');
    if(!live)$('#cam-empty').style.display='flex';
    ctrl=!st?'<p class="muted">Waiting for Jetson.</p>':st.actor?'<p class="muted">The HIL-SERL actor holds the camera and streams it here.</p>':
      st.preview?'<button class="btn wide" onclick="camPreview(false)">Stop preview</button><p class="faint" style="font-size:12px;margin-top:12px">actor.py stops the preview by itself when it starts.</p>':
      '<button class="btn primary wide" onclick="camPreview(true)">Start preview</button><p class="faint" style="font-size:12px;margin-top:12px">Opens only the side RealSense, never the arm.</p>';
  }else{
    const a=st?.wrist_age;
    chips=`<span class="chip ${a!=null&&a<3?'ok':''}">${a==null?'No frames':a<3?'Live':'Last frame · '+(a<120?Math.round(a)+' s':a<7200?Math.round(a/60)+' min':Math.round(a/3600)+' h')+' old'}</span>`;
    if(a==null)$('#cam-empty').style.display='flex';
    ctrl='<p class="muted">The wrist camera streams while any robot session runs (calib, demo, train). An older frame is fine for framing: the camera rides on the arm, so the plug sits at the same place in the image.</p>';
  }
  $('#cam-chips').innerHTML=chips;if($('#cam-ctrl').innerHTML!==ctrl)$('#cam-ctrl').innerHTML=ctrl;
  const saved=st?.[V.key],at=CAM.view==='side'?st?.saved_at:st?.crop_saved_at;
  if(!CAM.dirty&&st){const w=Math.min(240,st.size[1]);CAM.crop=saved||[Math.round((st.size[0]-w)/2),Math.round((st.size[1]-w)/2),w,w];placeCrop()}
  $('#cam-saved').textContent=(saved?`Saved ${saved.join(' ')}${at?' · '+at.replace('T',' '):''}`:'Nothing saved yet')+(CAM.dirty?' · unsaved changes':'');
}
async function camPreview(on){const r=await post('/api/side/preview',{on});if(r.ok)toast(r.message);camStatus()}
async function saveCrop(){if(!CAM.crop)return;const r=await post('/api/side/crop',{[CAMV[CAM.view].key]:CAM.crop});if(r.ok){toast(r.message);CAM.dirty=false;camStatus()}}
function placeCrop(){
  const el=$('#cam-crop');if(!el||!CAM.crop)return;const [x,y,w,h]=CAM.crop,[W,H]=CAM.st?.size||[640,480];
  Object.assign(el.style,{left:100*x/W+'%',top:100*y/H+'%',width:100*w/W+'%',height:100*h/H+'%',display:'block'});
  $('#cam-size').value=w;$('#cam-val').textContent=`${CAMV[CAM.view].flag} ${CAM.crop.join(' ')}`;
}
function clampCrop(c){const [W,H]=CAM.st?.size||[640,480];let [x,y,w]=c;w=Math.max(64,Math.min(w,H));
  return [Math.round(Math.max(0,Math.min(x,W-w))),Math.round(Math.max(0,Math.min(y,H-w))),w,w]}
function bindCrop(){
  const el=$('#cam-crop'),box=$('#cam-box');let drag=null;
  el.addEventListener('pointerdown',e=>{if(!CAM.crop)return;el.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,c:[...CAM.crop]};e.preventDefault()});
  el.addEventListener('pointermove',e=>{if(!drag)return;const k=(CAM.st?.size?.[0]||640)/box.clientWidth;
    CAM.crop=clampCrop([drag.c[0]+(e.clientX-drag.x)*k,drag.c[1]+(e.clientY-drag.y)*k,drag.c[2]]);CAM.dirty=true;placeCrop()});
  el.addEventListener('pointerup',()=>{drag=null});
  $('#cam-size').addEventListener('input',e=>{if(!CAM.crop)return;const [x,y,w]=CAM.crop,n=+e.target.value;
    CAM.crop=clampCrop([x+(w-n)/2,y+(w-n)/2,n]);CAM.dirty=true;placeCrop()});
}
function drawCropPrev(img){const c=$('#cam-prev');if(!c||!CAM.crop)return;const [x,y,w,h]=CAM.crop;
  const g=c.getContext('2d');g.imageSmoothingQuality='high';g.drawImage(img,x,y,w,h,0,0,128,128)}

route();
