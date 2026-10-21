const $=id=>document.getElementById(id);
const api=(u,b)=>fetch(u,b===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(async r=>{const j=await r.json().catch(()=>({error:'Bad response ('+r.status+')'}));if(!r.ok&&!j.error)j.error='Error '+r.status;return j});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function toast(t,err){const e=$('toast');e.textContent=t;e.className='toast show'+(err?' err':'');clearTimeout(e._t);e._t=setTimeout(()=>e.className='toast',2800)}
function note(id,t,k){const e=$(id);e.textContent=t||'';e.className='note'+(k?' '+k:'')}
function busy(b,on){b.disabled=on;b.classList.toggle('busy',on)}
function opts(el,list,keep){const v=el.value;el.innerHTML=list.map(([v,l])=>`<option value="${esc(v)}">${esc(l)}</option>`).join('');if(keep&&list.some(x=>x[0]==v))el.value=v}
const base=p=>p.split('/').pop();
document.querySelectorAll('[data-t]').forEach(b=>b.onclick=()=>show(b.dataset.t));document.querySelectorAll('[data-go]').forEach(b=>b.onclick=()=>show(b.dataset.go));
function show(t){document.querySelectorAll('[data-t]').forEach(b=>b.classList.toggle('on',b.dataset.t==t));['setup','train','chat','files'].forEach(s=>$(s).hidden=s!=t);if(t=='chat'||t=='files')loadRuns();if(t=='train')last='';scrollTo(0,0)}
async function load(){const m=await api('/api/models');opts($('model'),m.map(x=>[x,base(x)]),1);$('modelEmpty').hidden=m.length>0;
 const d=await api('/api/datasets');opts($('data'),d.length?d.map(x=>[x,base(x)]):[['','No datasets yet, upload one']],1)}
$('hfBtn').onclick=async e=>{const b=e.target;if(!$('hf').value.trim())return note('hfmsg','Enter a repo id like owner/Model-GGUF.','err');busy(b,1);note('hfmsg','Downloading. Large files take a while.');
 const r=await api('/api/hf',{repo:$('hf').value});busy(b,0);if(r.error)return note('hfmsg',r.error,'err');note('hfmsg','Saved '+base(r.path),'ok');await load();$('model').value=r.path};
$('file').onchange=async()=>{const f=$('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);note('upmsg','Uploading '+f.name+'…');
 const r=await fetch('/api/upload',{method:'POST',body:fd}).then(r=>r.json()).catch(()=>({error:'Upload failed'}));$('file').value='';
 if(r.error)return note('upmsg',r.error,'err');note('upmsg',base(r.path)+' ready: '+r.info,'ok');await load();$('data').value=r.path};
$('go').onclick=async e=>{const c={};['model','data','run','rank','alpha','targets','steps','seq','batch','lr','threads'].forEach(k=>c[k]=$(k).value);c.fast=$('fast').checked;c.recompute=$('recompute').checked;
 if(!c.model||!c.data)return note('trmsg','Pick a model and a dataset first.','err');busy(e.target,1);const r=await api('/api/train',c);busy(e.target,0);
 if(r.error)return note('trmsg',r.error,'err');note('trmsg','');last='';show('train')};
$('stop').onclick=async e=>{busy(e.target,1);await api('/api/stop',{});toast('Stopping after this step. The adapter is saved.');setTimeout(()=>busy(e.target,0),1500)};
$('resume').onclick=async e=>{busy(e.target,1);const r=await api('/api/resume',{});busy(e.target,0);r.error?toast(r.error,1):toast('Resumed from the saved adapter')};
const LABEL={idle:'Idle',running:'Training',stopping:'Stopping',stopped:'Stopped',finished:'Done'};
const HEAD={running:'Training',stopping:'Saving and stopping',stopped:'Paused. Resume picks up from the saved adapter.',finished:'Finished',error:'The trainer exited with an error'};
const fmt=s=>s==null?'–':s>=3600?Math.floor(s/3600)+'h '+Math.round(s%3600/60)+'m':s>=60?Math.floor(s/60)+'m '+(s%60)+'s':s+'s';
function draw(s,h){const cv=$('chart'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=200;cv.width=W*dpr;cv.height=H*dpr;const x=cv.getContext('2d');x.scale(dpr,dpr);
 if(!s.length){x.fillStyle='#8b99b3';x.font='13px IBM Plex Sans';x.fillText('Waiting for the first step…',8,H/2);return}
 const all=s.map(p=>p.loss).concat(h.map(p=>p.loss)),mn=Math.min(...all),mx=Math.max(...all),N=s[s.length-1].of||s.length,L=34,R=8,T=10,B=22;
 const X=i=>L+(W-L-R)*i/Math.max(N,1),Y=v=>T+(H-T-B)*(1-(v-mn)/((mx-mn)||1));
 x.strokeStyle='#1f2d47';x.lineWidth=1;x.fillStyle='#8b99b3';x.font='11px IBM Plex Mono';
 for(let k=0;k<=3;k++){const v=mn+(mx-mn)*k/3,y=Y(v);x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();x.fillText(v.toFixed(2),0,y+4)}
 x.fillText('0',L,H-6);x.fillText(String(N),W-R-x.measureText(String(N)).width,H-6);
 x.strokeStyle='#3987e5';x.lineWidth=2;x.lineJoin='round';x.beginPath();s.forEach((p,i)=>i?x.lineTo(X(p.step),Y(p.loss)):x.moveTo(X(p.step),Y(p.loss)));x.stroke();
 x.fillStyle='#f2c98a';h.forEach(p=>{x.beginPath();x.arc(X(p.step),Y(p.loss),3.5,0,7);x.fill()})}
let last='';
async function poll(){try{const s=await api('/api/status');const st=s.state.startsWith('error')?'error':s.state;
 $('pill').dataset.s=st;$('pilltxt').textContent=LABEL[st]||'Error';
 const has=!!s.run;$('trainEmpty').hidden=has;$('trainBody').hidden=!has;
 if(has){$('runname').textContent=s.run;$('headline').textContent=HEAD[st]||st;const l=s.steps[s.steps.length-1];
  $('tps').textContent=l?l.tps:'–';$('ram').textContent=s.rss||(l&&l.rss)||'–';$('stp').textContent=l?`step ${l.step} of ${l.of}`:'starting';
  $('eta').textContent=st=='running'?(s.eta!=null?fmt(s.eta):'…'):st=='finished'?'done':'–';
  $('prog').style.width=l?(100*l.step/l.of)+'%':'0';$('prog').style.background=st=='error'?'var(--bad)':st=='finished'?'var(--blue)':'var(--green)';
  $('loss').textContent=l?l.loss.toFixed(3):'–';const hh=s.heldout[s.heldout.length-1];$('hl').textContent=hh?hh.loss.toFixed(3):'–';
  $('stop').disabled=st!='running';$('resume').disabled=st=='running'||st=='stopping'||st=='finished';
  const key=JSON.stringify([s.steps.length,s.heldout.length,st]);if(key!=last&&$("chart").clientWidth){draw(s.steps,s.heldout);last=key}
  $('log').textContent=s.log.join('\n');
  if(st=='finished'&&poll.prev=='running')toast('Training finished. Your adapter is in Files.');poll.prev=st}}catch(e){$('pilltxt').textContent='Offline';$('pill').dataset.s='error'}
 setTimeout(poll,1500)}
addEventListener('resize',()=>last='');
async function loadRuns(){const r=await api('/api/runs');opts($('crun'),[['','Base model, no adapter']].concat(r.map(x=>[x.run,x.run])),1);
 $('flist').innerHTML=r.length?r.map(x=>`<div class="card file"><div><h2>${esc(x.run)}</h2><p>${x.eslora_mb} MB adapter${x.gguf?' · GGUF exported':''}</p></div><div class="dl">
 <a class="btn sm" href="/dl/${encodeURIComponent(x.run)}.eslora" download>.eslora</a>${x.gguf?`<a class="btn sm" href="/dl/${encodeURIComponent(x.run)}.gguf" download>llama.cpp .gguf</a>`:`<button class="btn sm" data-exp="${esc(x.run)}">Export GGUF</button>`}</div></div>`).join('')
 :'<div class="card empty-state"><p>No adapters yet. Finished or stopped runs land here.</p><button class="btn" onclick="show(\'setup\')">Start a run</button></div>';
 document.querySelectorAll('[data-exp]').forEach(b=>b.onclick=async()=>{busy(b,1);const x=await api('/api/export/'+b.dataset.exp,{model:$('model').value});busy(b,0);x.error?toast(x.error,1):toast('Exported '+b.dataset.exp+'.gguf');loadRuns()})}
function bub(html,c){const m=$('msgs');m.querySelector('.empty')?.remove();const d=document.createElement('div');d.className='b '+c;d.innerHTML=html;m.appendChild(d);d.scrollIntoView({block:'nearest',behavior:'smooth'});return d}
let fresh=false;$('reset').onclick=()=>{$('msgs').innerHTML='<p class="empty">New chat. History cleared.</p>';fresh=true};
$('cin').addEventListener('keydown',e=>{if(e.key=='Enter'&&!e.shiftKey){e.preventDefault();$('cform').requestSubmit()}});
$('cin').addEventListener('input',e=>{e.target.style.height='auto';e.target.style.height=e.target.scrollHeight+'px'});
$('cform').onsubmit=async e=>{e.preventDefault();const m=$('cin').value.trim();if(!m)return;bub(esc(m),'me');$('cin').value='';$('cin').style.height='';const d=bub('<span class="dots">Thinking</span>','bot');busy($('send'),1);
 const r=await api('/api/chat',{model:$('model').value,run:$('crun').value,msg:m,reset:fresh}).catch(()=>({error:'Connection lost'}));fresh=false;busy($('send'),0);
 if(r.error){d.classList.add('err');d.textContent=r.error;return}d.innerHTML=esc(r.reply.trim()||'(empty reply)')+(r.tps?`<small>${r.tps.toFixed(1)} tok/s · ${r.ram_mb} MB</small>`:'')};
load();poll();
