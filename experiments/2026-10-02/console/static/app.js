// A1Z Console front end. Plain JS, no build step. Every robot action goes through /api/session/key,
// which types the same key into the tmux client window the terminal flow uses.
const TOKEN=document.querySelector('meta[name=a1z-token]').content;
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const S={session:{state:'none'},tele:null,usb:null,models:[],page:'overview',timers:[],lastPane:''};

async function get(p){const r=await fetch(p,{cache:'no-store'});if(!r.ok)throw new Error((await r.json().catch(()=>({}))).error||r.status);return r.json()}
async function post(p,body){
  const r=await fetch(p,{method:'POST',headers:{'Content-Type':'application/json','X-A1Z-Token':TOKEN},body:JSON.stringify(body||{})});
  const j=await r.json().catch(()=>({ok:false,message:'无响应'}));if(!j.ok)toast(j.message||'失败',true);return j}
function toast(msg,bad){const t=$('#statusline');t.textContent=bad?`[ERROR: ${msg}]`:`[${msg}]`;t.className='show'+(bad?' err':'');clearTimeout(t._h);t._h=setTimeout(()=>t.className='',3200)}
function openModal(html){$('#modal-body').innerHTML=html;$('#modal').classList.remove('hidden')}
function closeModal(){$('#modal').classList.add('hidden');$('#modal-body').innerHTML=''}
$('#modal').addEventListener('click',e=>{if(e.target.id==='modal')closeModal()});

const fmtSess=s=>s?`${s.slice(4,6)}-${s.slice(6,8)} ${s.slice(9,11)}:${s.slice(11,13)}`:'—';
const pct=(k,n)=>n?`${Math.round(100*k/n)}%`:'—';
const ci=c=>c?`${Math.round(c[0]*100)}–${Math.round(c[1]*100)}%`:'';
const KIND={eval:'评估',hil:'HIL 采集',control:'控制测试'};
const STATE={none:['没有会话','','OFF'],starting:['启动中','run','BOOT'],await_ready:['等待现场确认','warn','READY?'],idle:['就绪 · 重力辅助','ok','IDLE'],
  running:['运行中','run','RUN'],human:['人工接管中','warn','HAND'],recording:['录制示范中','run','REC'],ended:['会话已结束','','END']};
const LABEL={missed:['没找到口','bad'],found_port:['找到口','warn'],in_mouth:['进口没充电','warn'],inserted:['充上电','ok'],charging:['充上电','ok']};

// ---------------- polling ----------------
async function pollSession(){try{S.session=await get('/api/session')}catch(e){S.session={state:'none',pane:''}};updateChrome();if(S.page==='run')updateRun()}
async function pollTele(){try{S.tele=await get('/api/telemetry')}catch(e){};updateChrome();if(S.page==='run')updateRunTele();if(S.page==='health')renderHealth(true)}
async function pollUsb(){try{S.usb=await get('/api/usb')}catch(e){}}
setInterval(pollSession,800);setInterval(pollTele,1000);setInterval(pollUsb,1500);
pollSession();pollTele();pollUsb();get('/api/models').then(m=>S.models=m).catch(()=>{});

function gripTemp(){const t=S.tele;return t&&t.g_temp_rotor!=null?t.g_temp_rotor:null}
function gripClass(v){if(v==null)return '';const t=S.tele;return v>=t.grip_block_c?'bad':v>=t.grip_warn_c?'warn':'ok'}
function updateChrome(){
  const st=S.session.state||'none',[txt,cls]=STATE[st]||[st,''];
  $('#side-state').textContent=txt;$('#side-dot').className='dot '+(cls==='run'?'run':cls==='ok'?'on':cls==='warn'?'warn':'');
  const g=gripTemp();$('#side-grip').textContent=g==null?'夹爪温度 —':`夹爪线圈 ${g.toFixed(0)}°C${S.tele.live?'':'（上次）'}`;
}

// ---------------- router ----------------
const PAGES={overview:renderOverview,run:renderRun,runs:renderRuns,report:renderReport,models:renderModels,data:renderData,health:renderHealth};
function route(){
  const h=location.hash.replace(/^#\//,'')||'overview';const [page,...rest]=h.split('/');
  S.page=PAGES[page]?page:'overview';
  document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.dataset.page===S.page));
  S.timers.forEach(clearInterval);S.timers=[];
  PAGES[S.page](rest);
}
window.addEventListener('hashchange',route);

// ---------------- overview ----------------
const kpi=(l,v,h,c,unit)=>`<div class="card kpi"><div class="label">${l}</div><div class="value ${c?'c-'+c:''}">${esc(v)}${unit?`<small>${unit}</small>`:''}</div><div class="hint">${esc(h)}</div></div>`;
const stat=(k,v,c,unit)=>`<div class="stat"><span class="k">${k}</span><span class="v ${c?'c-'+c:''}">${v}${unit?`<small>${unit}</small>`:''}</span></div>`;
const segs=(frac,cls,n=20,hero)=>{const f=Math.round(Math.max(0,Math.min(1,frac))*n);const col=cls==='bad'?'var(--accent)':cls==='warn'?'var(--warning)':cls==='ok'?'var(--success)':'var(--text-display)';
  return `<span class="segs${hero?' hero':''}" style="grid-template-columns:repeat(${n},1fr)">${Array.from({length:n},(_,i)=>`<i style="${i<f?'background:'+col:''}"></i>`).join('')}</span>`};
async function renderOverview(){
  $('#main').innerHTML=`<div class="eyebrow">01 / Overview</div><h1>概览</h1><p class="sub">A1Z 单臂 · USB 插入任务 · Jetson 本地推理</p>
  <div class="hero-split" id="ov-hero"></div>
  <div class="grid g2"><div class="card"><div class="section-title"><h2>最近的运行</h2><a href="#/runs">全部 →</a></div><div id="ov-runs"></div></div>
  <div class="card"><div class="section-title"><h2>当前控制设置下的模型</h2><a href="#/report">报告 →</a></div><div id="ov-report"></div></div></div>`;
  const kp=()=>{const st=S.session.state||'none',[txt,,code]=STATE[st]||[st,'',st];const g=gripTemp();const t=S.tele||{};
    $('#ov-hero').innerHTML=`<div><div class="label">会话状态</div><div class="hero ${st==='running'?'accent':''}">${code}</div>
      <div class="hero-desc">${txt}${S.session.model?' · '+esc(S.session.model):''}</div><div class="hero-meta">${S.session.mode?KIND[S.session.mode]:'在运行台启动会话'}</div></div>
      <div>${stat('夹爪线圈',g==null?'—':g.toFixed(0),gripClass(g),g==null?'':'°C'+(t.live?'':' · 上次'))}
      ${stat('最热关节',t.temp_rotor?Math.max(...t.temp_rotor).toFixed(0):'—','',t.temp_rotor?'°C':'')}
      ${stat('USB 信号',S.usb?.connected==null?'—':S.usb.connected?'已识别':'未插入',S.usb?.connected?'ok':'')}
      ${stat('控制循环',t.hz?Math.round(t.hz):'—','',t.hz?'Hz':'')}</div>`};
  kp();S.timers.push(setInterval(kp,1000));
  try{const runs=await get('/api/runs');$('#ov-runs').innerHTML=runsTable(runs.slice(0,6),true)}catch(e){$('#ov-runs').textContent=e}
  try{const rep=(await get('/api/report')).filter(r=>r.ki>0&&r.ki_version!=='v1');$('#ov-report').innerHTML=rep.length?reportTable(rep,true):'<div class="empty-state">还没有开积分补偿后的评估</div>'}catch(e){}
}
// ---------------- run console ----------------
let pref=JSON.parse(localStorage.getItem('a1z-run')||'{}');
function savePref(){localStorage.setItem('a1z-run',JSON.stringify(pref))}
function renderRun(){
  $('#main').innerHTML=`<div class="eyebrow">02 / Run</div><h1>运行台</h1><div class="bar" id="run-chips" style="margin-top:16px"></div>
  <div class="runlayout">
    <div class="stack">
      <div class="live"><img id="live-img" alt=""><div class="ov" id="live-ov"></div><div class="corner">Wrist · D405 · 5 fps</div><div class="empty" id="live-empty"><span>No signal</span><span style="color:#444">会话运行时显示手腕相机画面</span></div></div>
      <div class="grid g2"><div class="card tight"><h2>夹爪尖 · 基座坐标</h2><div id="tip"></div>
        <div class="faint" style="font-size:12px;margin-top:12px">x = hub 长边，y = 短边（10-05 实测）</div></div>
        <div class="card tight"><h2>温度 · 线圈</h2><div id="temps"></div></div></div>
      <div class="card tight"><div class="section-title"><h2>本次会话</h2><span class="eyebrow" id="cur-sess"></span></div><div id="cur-trials" class="muted">—</div></div>
    </div>
    <div class="stack"><div id="state-hero"></div><div id="ctrl"></div><div class="card tight"><h2>客户端输出</h2><div class="term" id="term"></div>
      <div class="faint" style="font-size:12px;margin-top:12px">按钮等同于在终端按键 · 后备：<span class="mono">ssh -t jetson-direct tmux attach -t eval</span></div></div></div>
  </div>`;
  const img=$('#live-img');
  S.timers.push(setInterval(()=>{if(['none','ended'].includes(S.session.state))return;const n=new Image();n.onload=()=>{img.src=n.src;$('#live-empty').style.display='none'};n.onerror=()=>{$('#live-empty').style.display='flex'};n.src='/api/live.jpg?t='+Date.now()},250));
  S.ctrlKey='';updateRun();updateRunTele();refreshTrials();S.timers.push(setInterval(refreshTrials,3000));
}
function chipsHtml(){const st=S.session.state||'none',[txt,cls]=STATE[st]||[st,''];const g=gripTemp();
  return (S.session.mode?`<span class="chip">${KIND[S.session.mode]}</span>`:'')+
  (S.session.model?`<span class="chip">模型 ${esc(S.session.model)}</span>`:'')+
  `<span class="chip ${gripClass(g)}">夹爪 ${g==null?'—':g.toFixed(0)+'°C'}</span>`+
  `<span class="chip ${S.usb?.connected?'ok':''}">USB ${S.usb?.connected?'已识别':'未插入'}</span>`}
function updateRun(){
  if(!$('#run-chips'))return;$('#run-chips').innerHTML=chipsHtml();
  const st=S.session.state||'none',[txt,,code]=STATE[st]||[st,'',st];
  $('#state-hero').innerHTML=`<div style="padding:8px 4px 8px"><div class="label">状态</div><div class="hero ${st==='running'||st==='recording'?'accent':''}" style="font-size:72px">${code}</div><div class="hero-meta">${txt}</div></div>`;
  const pane=S.session.pane||'';if(pane!==S.lastPane){S.lastPane=pane;const t=$('#term');t.textContent=pane||'（没有会话）';t.scrollTop=t.scrollHeight}
  const key=`${S.session.state}|${S.session.mode}|${gripClass(gripTemp())}`;if(key===S.ctrlKey)return;S.ctrlKey=key;renderCtrl();
}
function renderCtrl(){
  const st=S.session.state||'none',mode=S.session.mode,c=$('#ctrl');const hot=gripClass(gripTemp())==='bad';
  if(st==='none'){c.innerHTML=newSessionForm();bindForm();return}
  if(st==='starting'){c.innerHTML=`<div class="card"><h2>正在启动…</h2><p class="muted">加载模型和机械臂进程，约 10 秒。</p></div>`;return}
  if(st==='await_ready'){c.innerHTML=`<div class="card ready-card"><h2>现场确认</h2><p>确认机械臂在支撑上、周围没有障碍、你在现场。点击后电机进入厂家重力辅助（和录制时一样）。</p>
    <button class="btn primary big wide" onclick="ready()">机械臂已支撑，发送 READY</button>
    <button class="btn wide" style="margin-top:8px" onclick="cancelReady()">取消</button></div>`;return}
  if(st==='ended'){c.innerHTML=`<div class="card"><h2>会话已结束</h2><p class="muted">数据已保存。关闭后可以开始新的会话。</p><button class="btn primary wide" onclick="closeSess()">关闭会话</button></div>`;return}
  const running=st==='running'||st==='recording',human=st==='human',idle=st==='idle';
  let main='';
  if(mode==='control'){
    main=`<div class="btns">${b('t','示教（插好时）','t',idle)}${b('g','走回示教点 A1','g',idle&&!hot,'primary')}${b('r',st==='recording'?'停止录制':'录示范','r',idle||st==='recording')}${b('p','回放 A2','p',idle&&!hot,'primary')}${b('w','显示位置','w',idle)}</div>`;
  }else if(mode==='hil'){
    main=`<div class="btns">${b('g','开始一轮','G',idle&&!hot,'primary big')}${b(' ',human?'交还模型':'接管','Space',running||human,'big')}${b('x','结束这一轮','X',human)}${b('d','丢弃上一轮','D',idle)}</div>`;
  }else{
    main=`<div class="btns">${b('g','开始一轮','G',idle&&!hot,'primary big')}</div>`;
  }
  c.innerHTML=`<div class="card"><div class="section-title"><h2>${KIND[mode]||'会话'} · 控制</h2><span class="eyebrow">${esc(S.session.model||'')}</span></div>
    ${hot?`<div class="alert bad">夹爪线圈超过 ${S.tele.grip_block_c}°C，先休息降温，开始按钮已锁定。</div>`:gripClass(gripTemp())==='warn'?`<div class="alert warn">夹爪偏热（≥${S.tele.grip_warn_c}°C），跑完这轮建议休息。</div>`:''}
    ${main}
    <button class="btn danger big wide" style="margin-top:16px" onclick="key('s')" ${running?'':'disabled'}>停止这一轮 <kbd>Esc</kbd></button>
    <div class="label" style="margin:28px 0 12px">这一轮结束后</div>
    <div class="btns">${mode!=='hil'?b('i','记参考位置（手动插好后）','I',idle):''}
      ${b('1','1 没找到口','1',idle)}${b('2',mode==='control'?'2 进口没充电':'2 找到口','2',idle)}${b('3','3 充上电','3',idle)}</div>
    <div class="btns" style="margin-top:12px">${b(' ','夹爪 开/合','',idle&&mode!=='hil'||idle)}<button class="btn" onclick="quitClient()" ${idle?'':'disabled'}>退出会话</button></div>
    ${mode==='eval'?`<div class="faint" style="font-size:12px;margin-top:20px">每轮：开始 → 停止 → 用手插好（能充电）→ 记参考位置 → 标 1/2/3 → 拔出、带回起点。</div>`:''}
  </div>`;
}
const b=(k,label,kb,en,cls='')=>`<button class="btn ${cls}" onclick="key('${k===' '?'space':k}')" ${en?'':'disabled'}>${label}${kb?` <kbd>${kb}</kbd>`:''}</button>`;
async function key(k){const r=await post('/api/session/key',{key:k==='space'?' ':k});if(r.ok)setTimeout(pollSession,150)}
async function ready(){if(!confirm('机械臂在支撑上、现场有人？'))return;const r=await post('/api/session/key',{key:'READY'});if(r.ok)toast('READY 已发送')}
async function quitClient(){if(!confirm('退出这个会话？（确认手臂在支撑上）'))return;await post('/api/session/key',{key:'q'});setTimeout(()=>post('/api/session/key',{key:'q'}),300)}
async function cancelReady(){const r=await post('/api/session/key',{key:'CANCEL'});if(r.ok)toast('已取消，点「关闭会话」清理')}
async function closeSess(){const r=await post('/api/session/close');if(r.ok){toast('会话已关闭');S.ctrlKey=''}}
function newSessionForm(){
  const m=pref.mode||'eval',models=S.models.length?S.models:[{name:'060000'}];
  return `<div class="card"><h2>新会话</h2>
  <div class="seg">${['eval','hil','control'].map(x=>`<button class="${m===x?'on':''}" onclick="pref.mode='${x}';savePref();S.ctrlKey='';renderCtrl()">${KIND[x]}</button>`).join('')}</div>
  ${m!=='control'?`<div class="field"><label>模型</label><select id="f-model">${models.map(x=>`<option value="${esc(x.name)}" ${pref.model===x.name?'selected':''}>${esc(x.name)}${x.description?' — '+esc(x.description):''}</option>`).join('')}</select></div>`:''}
  ${m==='hil'?`<div class="field"><label>HG-DAgger 轮次（epoch）</label><input id="f-epoch" type="number" min="1" value="${pref.epoch||3}"></div>`:''}
  <div class="row2"><div class="field"><label>刚度 × kp</label><input id="f-stiff" type="number" step="0.1" value="${pref.stiffness??1.5}"></div>
  <div class="field"><label>积分 ki</label><input id="f-ki" type="number" step="0.5" value="${pref.ki??2}"></div>
  <div class="field"><label>上限 °</label><input id="f-kimax" type="number" step="0.5" value="${pref.ki_max??3}"></div></div>
  <button class="btn primary big wide" id="f-start">启动会话</button>
  <p class="faint" style="font-size:12px;margin:10px 0 0">启动只加载程序，电机不会动。下一步需要你在现场确认 READY。</p></div>`}
function bindForm(){$('#f-start').onclick=async()=>{
  const p={mode:pref.mode||'eval',stiffness:$('#f-stiff').value,ki:$('#f-ki').value,ki_max:$('#f-kimax').value};
  if($('#f-model'))p.model=$('#f-model').value;if($('#f-epoch'))p.epoch=$('#f-epoch').value;
  Object.assign(pref,{stiffness:+p.stiffness,ki:+p.ki,ki_max:+p.ki_max,model:p.model||pref.model,epoch:p.epoch||pref.epoch});savePref();
  const r=await post('/api/session/start',p);if(r.ok){toast(r.message);S.session.state='starting';S.ctrlKey='';renderCtrl()}}}
function updateRunTele(){
  const t=S.tele;if(!$('#tip')||!t)return;
  $('#tip').innerHTML=t.tip_mm&&t.live?['x','y','z'].map((a,i)=>stat(a,t.tip_mm[i].toFixed(1),'','mm')).join(''):'<div class="empty-state" style="padding:24px 0">—</div>';
  const rows=[];if(t.g_temp_rotor!=null)rows.push(tbar('夹爪',t.g_temp_rotor,t.grip_warn_c,t.grip_block_c));
  (t.temp_rotor||[]).forEach((v,i)=>rows.push(tbar('J'+(i+1),v,75,90)));
  $('#temps').innerHTML=rows.join('')+(t.live?'':`<div class="faint" style="font-size:12px;margin-top:8px">上次会话的读数 · ${fmtSess(t.session)}</div>`);
  const ov=$('#live-ov');if(ov)ov.innerHTML=t.live&&t.mode===1?'<span class="chip run">Engaged</span>':t.live?'<span class="chip">Float</span>':'';
}
const tbar=(n,v,w,blk)=>{const cls=v>=blk?'bad':v>=w?'warn':'';
  return `<div class="tbar"><span class="nm">${n}</span>${segs((v-25)/(blk-25),cls)}<span class="v ${cls?'c-'+cls:''}">${v.toFixed(0)}°</span></div>`};
async function refreshTrials(){
  const t=S.tele;if(!t||!t.session||!$('#cur-trials'))return;
  try{const d=await get(`/api/runs/${t.kind}/${t.session}`);$('#cur-sess').textContent=`${KIND[t.kind]} · ${fmtSess(t.session)}`;
    $('#cur-trials').innerHTML=d.trials.length?trialsTable(d,true):'<span class="muted">还没有完成的轮次</span>'}catch(e){$('#cur-trials').textContent='还没有完成的轮次'}
}
document.addEventListener('keydown',e=>{
  if(S.page!=='run'||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName)||e.metaKey||e.ctrlKey)return;
  const st=S.session.state,k=e.key.toLowerCase();
  if(k==='escape'&&(st==='running'||st==='recording')){key('s');e.preventDefault();return}
  if(k===' '&&S.session.mode==='hil'&&(st==='running'||st==='human')){key(' ');e.preventDefault();return}
  if(st!=='idle')return;
  if(['g','i','1','2','3','x','d','t','r','p','w'].includes(k)){key(k);e.preventDefault()}
});

// ---------------- runs ----------------
function labelCounts(r){const l=r.labels||{};const ok=(l.inserted||0)+(l.charging||0);const f=(l.found_port||0)+(l.in_mouth||0);
  return `<span class="mono"><span class="cnt c-ok">${ok}</span><span class="faint"> · </span><span class="cnt c-warn">${f}</span><span class="faint"> · </span><span class="cnt ${l.missed?'':'faint'}">${l.missed||0}</span></span>`}
function runsTable(runs,compact){
  if(!runs.length)return '<div class="empty-state">还没有运行记录</div>';
  return `<table><tr><th>Time</th><th>类型</th><th>模型</th>${compact?'':'<th class="num">刚度</th><th class="num">ki</th>'}<th class="num">有效/总</th><th>充电 · 找到 · 未找到</th></tr>
  ${runs.map(r=>`<tr class="click" onclick="location.hash='#/runs/${r.kind}/${r.session}'"><td class="mono">${fmtSess(r.session)}</td><td class="nw">${KIND[r.kind]}</td><td class="nw">${esc(r.model||'—')}${r.epoch?` <span class="faint">e${r.epoch}</span>`:''}</td>
  ${compact?'':`<td class="num">${r.stiffness??'—'}</td><td class="num">${r.ki||'关'}</td>`}<td class="num">${r.valid}/${r.trials}</td><td>${labelCounts(r)}</td></tr>`).join('')}</table>`}
async function renderRuns(rest){
  if(rest.length===2)return renderRunDetail(rest[0],rest[1]);
  $('#main').innerHTML=`<div class="eyebrow">03 / Runs</div><h1>运行记录</h1><p class="sub">评估、HIL 采集和控制测试的所有会话。点开查看每一轮、视频，修改标注或排除误操作。</p>
  <div class="seg" style="max-width:420px" id="rk">${['all','eval','hil','control'].map(k=>`<button data-k="${k}">${k==='all'?'全部':KIND[k]}</button>`).join('')}</div><div class="card" id="runs">加载中…</div>`;
  const runs=await get('/api/runs');const draw=k=>{if(!$('#runs'))return;document.querySelectorAll('#rk button').forEach(x=>x.classList.toggle('on',x.dataset.k===k));$('#runs').innerHTML=runsTable(k==='all'?runs:runs.filter(r=>r.kind===k))};
  document.querySelectorAll('#rk button').forEach(x=>x.onclick=()=>draw(x.dataset.k));draw('all');
}
function errCell(t){if(!t.err)return '<span class="faint">—</span>';const e=t.err;return `<b>${(t.err_xy??Math.hypot(e[0],e[1])).toFixed(1)}</b> <span class="faint">x${e[0]>=0?'+':''}${e[0].toFixed(1)} y${e[1]>=0?'+':''}${e[1].toFixed(1)} z${e[2]>=0?'+':''}${e[2].toFixed(1)}</span>`}
function trialsTable(d,compact){
  const kind=d.kind;
  return `<table><tr><th>#</th><th>结果</th><th>USB</th><th>离口 mm · 水平 (x y z)</th>${compact?'':'<th class="num">时长</th><th>结束原因</th><th>视频</th><th>排除</th>'}</tr>
  ${d.trials.map(t=>{const [ln,lc]=LABEL[t.label]||['未标',''];
    return `<tr class="${t.excluded?'dim':''}"><td class="mono">${t.n??''}${t.kind?` <span class="faint">${esc(t.kind)}</span>`:''}</td>
    <td>${compact?`<span class="${lc?'c-'+lc:''}">${ln}</span>`:`<select class="lbl" onchange="editTrial('${kind}','${d.session}','${t.file}',{label:this.value})">
      <option value="" ${!t.label?'selected':''} disabled>未标</option><option value="1" ${t.label==='missed'?'selected':''}>没找到口</option>
      <option value="2" ${['found_port','in_mouth'].includes(t.label)?'selected':''}>${kind==='control'?'进口没充电':'找到口'}</option><option value="3" ${['inserted','charging'].includes(t.label)?'selected':''}>充上电</option></select>`}</td>
    <td>${t.usb?'<span class="chip ok">USB</span>':''}</td><td class="mono">${errCell(t)}</td>
    ${compact?'':`<td class="num">${t.duration_s??''}s</td><td class="muted">${esc(t.end_reason||'')}${t.interventions?` · 接管 ${t.interventions} 次`:''}</td>
    <td>${t.video?`<button class="btn sm" onclick="playVideo('${kind}','${d.session}','${t.video}')">播放</button>`:''}</td>
    <td><input type="checkbox" ${t.excluded?'checked':''} onchange="editTrial('${kind}','${d.session}','${t.file}',{excluded:this.checked})"></td>`}</tr>`}).join('')}</table>`}
async function renderRunDetail(kind,name){
  $('#main').innerHTML='加载中…';const d=await get(`/api/runs/${kind}/${name}`);
  const errs=d.trials.filter(t=>!t.excluded&&t.err_xy!=null).map(t=>t.err_xy).sort((a,b)=>a-b);
  $('#main').innerHTML=`<a href="#/runs" class="eyebrow" style="text-decoration:none">← 03 / Runs</a><h1>${KIND[kind]} · <span class="mono">${fmtSess(name)}</span></h1>
  <p class="sub">${esc(d.model||'')} · 刚度 ${d.stiffness??'—'} · 积分 ${d.ki||'关'} · 有效 ${d.valid}/${d.trials.length} 轮</p>
  <div class="grid g4" style="margin-bottom:24px">${kpi('充上电',`${(d.labels.inserted||0)+(d.labels.charging||0)}`,`USB 识别 ${d.usb} 次`,'ok')}${kpi('找到口（含充上电）',`${(d.labels.found_port||0)+(d.labels.in_mouth||0)+(d.labels.inserted||0)+(d.labels.charging||0)}`,'')}
  ${kpi('没找到口',`${d.labels.missed||0}`,'')}${kpi('离口误差 · 中位',errs.length?errs[Math.floor(errs.length/2)].toFixed(1):'—',`有参考位置的 ${errs.length} 轮`,'',errs.length?'mm':'')}</div>
  <div class="card">${trialsTable(d,false)}</div>
  <p class="faint" style="font-size:12px">「离口」= 模型停下时夹爪尖位置 − 你手动插好时按 i 记下的位置。排除和改标注会写回这一轮的 JSON（记录修改历史），原始日志和视频不动。</p>`;
}
async function editTrial(kind,session,file,patch){const r=await post('/api/trial/edit',{kind,session,file,...patch});if(r.ok){toast('已保存');if(S.page==='runs')renderRunDetail(kind,session)}}
function playVideo(kind,s,f){openModal(`<h2>${KIND[kind]} ${fmtSess(s)} · ${esc(f)}</h2><video src="/api/video/${kind}/${s}/${f}" controls autoplay playsinline></video>`)}

// ---------------- report ----------------
function reportTable(rows,compact){
  return `<table><tr><th>模型</th>${compact?'':'<th class="num">刚度</th><th class="num">ki</th>'}<th class="num">轮数</th><th>找到口 · 95% CI</th><th>充上电 · 95% CI</th><th>离口 mm · 中位</th></tr>
  ${rows.map(r=>`<tr><td>${esc(r.model)}${compact?` <span class="faint">刚度 ${r.stiffness}</span>`:''}</td>${compact?'':`<td class="num">${r.stiffness}</td><td class="num">${r.ki?r.ki+(r.ki_version==='v1'?' <span class="chip warn" title="早期版本：运动中会积累，导致晃动">v1</span>':''):'关'}</td>`}<td class="num">${r.n}</td>
  <td><span class="rate">${segs(r.labelled?r.found/r.labelled:0,'',10)}<span class="mono">${pct(r.found,r.labelled)}</span></span><span class="faint mono" style="font-size:11px">${r.found}/${r.labelled} · ${ci(r.found_ci)}</span></td>
  <td><span class="rate">${segs(r.n?r.power/r.n:0,'ok',10)}<span class="mono">${pct(r.power,r.n)}</span></span><span class="faint mono" style="font-size:11px">${r.power}/${r.n} · ${ci(r.power_ci)}</span></td>
  <td class="mono">${r.err_median!=null?`${r.err_median.toFixed(1)} <span class="faint">n=${r.err_n}</span>`:'<span class="faint">—</span>'}</td></tr>`).join('')}</table>`}
async function renderReport(){
  $('#main').innerHTML=`<div class="eyebrow">04 / Report</div><h1>评估报告</h1><p class="sub">按「模型 × 刚度 × 积分」汇总所有评估会话（已排除的轮次不计）。括号里是 95% 置信区间：区间重叠很多时，差别可能只是运气。</p><div class="card" id="rep">加载中…</div>
  <div style="margin-top:48px;max-width:720px"><h2>怎么读</h2><div class="muted">
  <p>找到口 = 标 2 或 3；充上电 = 标 3 或 USB 识别。离口误差只算有参考位置（按了 i）的轮次。</p>
  <p>2026-10-05 之前的会话没有积分补偿（ki 关），控制误差 2–5 mm，和之后的结果不能直接比较。标 v1 的是 10-05 上午第一版积分（运动中也会积累，会晃）。比较模型时只看同一行设置。</p></div></div>`;
  const rows=await get('/api/report');$('#rep').innerHTML=rows.length?reportTable(rows,false):'<div class="empty-state">还没有评估</div>';
}

// ---------------- models ----------------
async function renderModels(){
  const ms=await get('/api/models');S.models=ms;
  $('#main').innerHTML=`<div class="eyebrow">05 / Models</div><h1>模型</h1><p class="sub">Jetson 上可以部署的 checkpoint（~/a1z-infer/checkpoints）。</p><div class="grid g3">${ms.map(m=>`<div class="card model-card">
    <h3>${esc(m.name)}</h3><div class="desc">${esc(m.description||'')}</div>
    <div class="kv"><div>训练步数</div><div>${m.steps??'—'}</div><div>数据集</div><div>${esc(m.dataset||'—')}</div><div>初始化自</div><div>${esc(m.init||'ImageNet backbone')}</div><div>保存时间</div><div>${m.saved}</div><div>大小</div><div>${m.size_mb} MB</div></div>
    <div class="btns" style="margin-top:24px"><button class="btn primary" onclick="useModel('${esc(m.name)}','eval')">评估</button><button class="btn" onclick="useModel('${esc(m.name)}','hil')">HIL 采集</button></div></div>`).join('')}</div>`;
}
function useModel(name,mode){pref.model=name;pref.mode=mode;savePref();location.hash='#/run';if(S.session.state!=='none')toast('先结束当前会话，再用这个模型')}

// ---------------- data ----------------
async function renderData(){
  const d=await get('/api/data');
  $('#main').innerHTML=`<div class="eyebrow">06 / Data</div><h1>数据</h1><p class="sub">${esc(d.note)}</p>
  <div class="grid g3" style="margin-bottom:24px">${kpi('示范总数',d.demos.total,'')}${kpi('可用示范（USB 确认）',d.demos.usable,'')}${kpi('HIL 纠正段（保留）',d.hil.reduce((a,h)=>a+h.kept,0),`${d.hil.length} 个采集会话`)}</div>
  <div class="card"><h2>HIL 采集会话</h2>${d.hil.length?`<table><tr><th>时间</th><th>轮次</th><th>模型</th><th class="num">试验</th><th class="num">接管</th><th class="num">人工时长</th><th class="num">可导出</th><th class="num">丢弃</th></tr>
  ${d.hil.map(h=>`<tr class="click" onclick="location.hash='#/runs/hil/${h.session}'"><td>${fmtSess(h.session)}</td><td>e${h.epoch??'?'}</td><td>${esc(h.model)}</td><td class="num">${h.trials}</td><td class="num">${h.interventions}</td><td class="num">${h.human_s}s</td><td class="num">${h.kept}</td><td class="num">${h.discarded}</td></tr>`).join('')}</table>`:'<div class="empty-state">没有 HIL 会话</div>'}</div>`;
}

// ---------------- health ----------------
function renderHealth(update){
  const t=S.tele;
  if(!update||!$('#h-chart'))$('#main').innerHTML=`<div class="eyebrow">07 / Hardware</div><h1>硬件状态</h1><p class="sub">来自最近一次会话的飞行记录（50 Hz）。夹爪合拢夹住插头时会持续出力，是主要发热源。</p>
    <div class="grid g4" id="h-kpi"></div><div class="grid g2" style="margin-top:16px"><div class="card"><h2>夹爪温度（最近 15 分钟）</h2><svg id="h-chart" class="spark" viewBox="0 0 600 120" preserveAspectRatio="none"></svg>
    <div class="legend"><span><i style="border-color:var(--text-display)"></i>夹爪线圈</span><span><i style="border-color:var(--text-display);opacity:.55"></i>夹爪驱动</span><span><i style="border-color:var(--text-display);opacity:.25;border-top-style:dotted"></i>最热关节</span><span class="faint">${t?.grip_warn_c??80}° 提醒 · ${t?.grip_block_c??88}° 锁定</span></div></div>
    <div class="card"><h2>各电机</h2><div id="h-temps"></div></div></div>`;
  if(!t||!t.session){$('#h-kpi').innerHTML='<div class="empty-state">还没有会话数据</div>';return}
  const g=t.g_temp_rotor;
  $('#h-kpi').innerHTML=kpi('夹爪线圈',g==null?'—':g.toFixed(0),t.live?'实时':'上次会话的读数',gripClass(g),'°C')+kpi('夹爪力矩',t.g_tau==null?'—':t.g_tau.toFixed(2),'夹住插头时持续出力','','N·m')+
    kpi('厂家控制循环',t.hz?Math.round(t.hz):'—','应约 250 Hz','','Hz')+kpi('数据来源',fmtSess(t.session),KIND[t.kind]+(t.live?' · 运行中':''));
  $('#h-temps').innerHTML=(g!=null?tbar('夹爪',g,t.grip_warn_c,t.grip_block_c):'')+(t.temp_rotor||[]).map((v,i)=>tbar('J'+(i+1),v,75,90)).join('')+
    `<div class="faint" style="font-size:12px;margin-top:8px">关节阈值（厂家）：线圈 75°C 警告 / 90°C 急停；驱动 70 / 85°C。夹爪阈值为本控制台的保守设定。</div>`;
  const H=(t.history||[]).filter(h=>h.g_rotor!=null);if(H.length<2)return;
  const t0=H[0].t,t1=H[H.length-1].t||1,ys=[30,100];const X=v=>((v-t0)/(t1-t0||1))*600,Y=v=>120-(v-ys[0])/(ys[1]-ys[0])*120;
  const line=(k,c)=>`<polyline fill="none" stroke="${c}" stroke-width="1.5" vector-effect="non-scaling-stroke" points="${H.filter(h=>h[k]!=null).map(h=>`${X(h.t).toFixed(1)},${Y(h[k]).toFixed(1)}`).join(' ')}"/>`;
  $('#h-chart').innerHTML=[40,60].map(v=>`<line x1="0" x2="600" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--border)" stroke-width="1"/>`).join('')+
    `<line x1="0" x2="600" y1="${Y(t.grip_warn_c)}" y2="${Y(t.grip_warn_c)}" stroke="var(--warning)" stroke-dasharray="3 4"/><line x1="0" x2="600" y1="${Y(t.grip_block_c)}" y2="${Y(t.grip_block_c)}" stroke="var(--accent)" stroke-dasharray="3 4"/>`+
    line('j_max','var(--text-display)" stroke-opacity=".25" stroke-dasharray="2 3')+line('g_mos','var(--text-display)" stroke-opacity=".55')+line('g_rotor','var(--text-display)');
}

route();
