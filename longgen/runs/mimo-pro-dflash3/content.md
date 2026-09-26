Here's a stunning, fully-featured PS4-style Tetris game in a single HTML file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>TETRIS</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--cyan:#0ff;--purple:#a0f;--bg1:#080820;--bg2:#10103a;--glass:rgba(255,255,255,.05);--gborder:rgba(255,255,255,.12);--txt:#fff;--acc:#0ff}
html,body{width:100%;height:100%;overflow:hidden;background:linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg1));font-family:'Segoe UI',Helvetica,Arial,sans-serif;color:var(--txt);touch-action:none;user-select:none;-webkit-user-select:none;-webkit-tap-highlight-color:transparent}
#app{display:flex;flex-direction:column;align-items:center;height:100%;padding:5px;position:relative;z-index:1}
.game-title{font-size:1.8em;font-weight:900;letter-spacing:12px;color:var(--acc);text-shadow:0 0 10px var(--acc),0 0 20px var(--acc),0 0 40px var(--acc);margin-bottom:6px;animation:tglow 2s ease-in-out infinite alternate}
@keyframes tglow{from{text-shadow:0 0 10px var(--acc),0 0 20px var(--acc),0 0 40px var(--acc)}to{text-shadow:0 0 15px var(--acc),0 0 30px var(--acc),0 0 60px var(--acc),0 0 80px var(--acc)}}
#game-area{display:flex;align-items:flex-start;gap:12px;flex:1;min-height:0}
.panel{background:var(--glass);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:1px solid var(--gborder);border-radius:12px;padding:12px;min-width:110px}
.panel-label{font-size:.75em;font-weight:700;letter-spacing:4px;color:var(--acc);text-shadow:0 0 8px var(--acc);margin-bottom:8px;text-align:center}
.stat{display:flex;justify-content:space-between;align-items:center;margin:6px 0;padding:3px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.stat-label{font-size:.65em;font-weight:600;letter-spacing:2px;color:rgba(255,255,255,.45)}
.stat-value{font-size:1em;font-weight:700;color:var(--acc);text-shadow:0 0 5px var(--acc);font-variant-numeric:tabular-nums}
#board-canvas{display:block;border:2px solid rgba(0,255,255,.25);border-radius:4px;box-shadow:0 0 20px rgba(0,255,255,.15),0 0 60px rgba(0,255,255,.05),inset 0 0 30px rgba(0,0,0,.5)}
#left-panel{display:flex;flex-direction:column;gap:10px}
#right-panel{display:flex;flex-direction:column;gap:10px}
.controls-info{font-size:.6em;color:rgba(255,255,255,.25);line-height:1.7;letter-spacing:1px;padding:8px 0}
#touch-controls{display:none;flex-direction:column;gap:8px;margin-top:8px;width:100%;max-width:320px;padding-bottom:5px}
.touch-row{display:flex;gap:8px;justify-content:center}
.touch-btn{flex:1;padding:14px 8px;background:var(--glass);backdrop-filter:blur(5px);border:1px solid var(--gborder);border-radius:12px;color:var(--txt);font-size:1.1em;font-weight:700;cursor:pointer;touch-action:manipulation;transition:all .1s}
.touch-btn:active,.touch-btn.active{background:rgba(0,255,255,.15);border-color:var(--acc);box-shadow:0 0 15px rgba(0,255,255,.3)}
.touch-btn.wide{flex:2}
.overlay{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.85);backdrop-filter:blur(5px);display:flex;align-items:center;justify-content:center;z-index:100}
.overlay.hidden{display:none}
.overlay-content{text-align:center;padding:35px;background:var(--glass);backdrop-filter:blur(20px);border:1px solid var(--gborder);border-radius:20px;max-width:380px;width:90%;animation:ovin .3s ease}
@keyframes ovin{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}
.overlay-content h2{font-size:2.2em;font-weight:900;letter-spacing:8px;color:var(--acc);text-shadow:0 0 20px var(--acc),0 0 40px var(--acc);margin-bottom:15px}
#overlay-message{font-size:.95em;color:rgba(255,255,255,.6);margin-bottom:15px;line-height:1.5}
#overlay-stats{text-align:left;max-width:250px;margin:0 auto 15px}
#overlay-stats .stat{margin:4px 0}
.neon-btn{padding:14px 35px;font-size:1.1em;font-weight:700;letter-spacing:4px;background:transparent;color:var(--acc);border:2px solid var(--acc);border-radius:30px;cursor:pointer;transition:all .3s;text-transform:uppercase;margin-top:10px}
.neon-btn:hover,.neon-btn:active{background:rgba(0,255,255,.1);box-shadow:0 0 20px rgba(0,255,255,.4),inset 0 0 15px rgba(0,255,255,.08)}
@media(max-width:768px){
  #touch-controls{display:flex}
  .controls-info{display:none}
  .panel{min-width:85px;padding:8px}
  .game-title{font-size:1.3em;letter-spacing:8px}
  #game-area{gap:8px}
}
@media(max-width:480px){
  .panel{min-width:70px;padding:6px}
  .stat-value{font-size:.85em}
  .stat-label{font-size:.55em}
  .game-title{font-size:1.1em;letter-spacing:6px}
  .touch-btn{padding:11px 6px;font-size:.95em}
}
</style>
</head>
<body>
<div id="app">
  <div class="game-title">TETRIS</div>
  <div id="game-area">
    <div id="left-panel">
      <div class="panel"><div class="panel-label">HOLD</div><canvas id="hold-canvas" width="80" height="80"></canvas></div>
      <div class="panel">
        <div class="stat"><span class="stat-label">SCORE</span><span class="stat-value" id="score">0</span></div>
        <div class="stat"><span class="stat-label">LEVEL</span><span class="stat-value" id="level">1</span></div>
        <div class="stat"><span class="stat-label">LINES</span><span class="stat-value" id="lines">0</span></div>
        <div class="stat"><span class="stat-label">TIME</span><span class="stat-value" id="time">0:00</span></div>
        <div class="stat"><span class="stat-label">COMBO</span><span class="stat-value" id="combo">-</span></div>
        <div class="stat"><span class="stat-label">B2B</span><span class="stat-value" id="b2b">-</span></div>
      </div>
      <div class="controls-info">
        <div>← → Move</div><div>↓ Soft Drop</div><div>↑ Rotate CW</div>
        <div>Z Rotate CCW</div><div>Space Hard Drop</div><div>C Hold</div><div>P Pause</div>
      </div>
    </div>
    <div id="board-container"><canvas id="board-canvas"></canvas></div>
    <div id="right-panel">
      <div class="panel"><div class="panel-label">NEXT</div><canvas id="next-canvas" width="80" height="400"></canvas></div>
    </div>
  </div>
  <div id="touch-controls">
    <div class="touch-row">
      <button class="touch-btn" id="btn-left">◀</button>
      <button class="touch-btn" id="btn-rcw">↻</button>
      <button class="touch-btn" id="btn-right">▶</button>
    </div>
    <div class="touch-row">
      <button class="touch-btn" id="btn-hold">HOLD</button>
      <button class="touch-btn" id="btn-soft">▼</button>
      <button class="touch-btn" id="btn-rccw">↺</button>
      <button class="touch-btn wide" id="btn-drop">⇣</button>
    </div>
  </div>
</div>
<div id="overlay" class="overlay">
  <div class="overlay-content">
    <h2 id="overlay-title">TETRIS</h2>
    <div id="overlay-message"></div>
    <div id="overlay-stats"></div>
    <button id="overlay-btn" class="neon-btn">START</button>
  </div>
</div>
<script>
const COLS=10,VISIBLE_ROWS=20,HIDDEN_ROWS=2,TOTAL_ROWS=VISIBLE_ROWS+HIDDEN_ROWS,PREVIEW_CELL=20;
let CELL_SIZE=30;
const COLORS={I:'#00ffff',O:'#ffff00',T:'#cc44ff',S:'#00ff44',Z:'#ff4444',J:'#4444ff',L:'#ff8800'};
const PIECES={
I:[[[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],[[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],[[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],[[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]],
O:[[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]],[[1,1],[1,1]]],
T:[[[0,1,0],[1,1,1],[0,0,0]],[[0,1,0],[0,1,1],[0,1,0]],[[0,0,0],[1,1,1],[0,1,0]],[[0,1,0],[1,1,0],[0,1,0]]],
S:[[[0,1,1],[1,1,0],[0,0,0]],[[0,1,0],[0,1,1],[0,0,1]],[[0,0,0],[0,1,1],[1,1,0]],[[1,0,0],[1,1,0],[0,1,0]]],
Z:[[[1,1,0],[0,1,1],[0,0,0]],[[0,0,1],[0,1,1],[0,1,0]],[[0,0,0],[1,1,0],[0,1,1]],[[0,1,0],[1,1,0],[1,0,0]]],
J:[[[1,0,0],[1,1,1],[0,0,0]],[[0,1,1],[0,1,0],[0,1,0]],[[0,0,0],[1,1,1],[0,0,1]],[[0,1,0],[0,1,0],[1,1,0]]],
L:[[[0,0,1],[1,1,1],[0,0,0]],[[0,1,0],[0,1,0],[0,1,1]],[[0,0,0],[1,1,1],[1,0,0]],[[1,1,0],[0,1,0],[0,1,0]]]
};
const SPAWN={I:{x:3,y:1},O:{x:4,y:2},T:{x:3,y:2},S:{x:3,y:2},Z:{x:3,y:2},J:{x:3,y:2},L:{x:3,y:2}};
const K_JLSTZ={'0_1':[[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],'1_2':[[0,0],[1,0],[1,1],[0,-2],[1,-2]],'2_3':[[0,0],[1,0],[1,-1],[0,2],[1,2]],'3_0':[[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],'1_0':[[0,0],[1,0],[1,1],[0,-2],[1,-2]],'2_1':[[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],'3_2':[[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],'0_3':[[0,0],[1,0],[1,-1],[0,2],[1,2]]};
const K_I={'0_1':[[0,0],[-2,0],[1,0],[-2,1],[1,-2]],'1_2':[[0,0],[-1,0],[2,0],[-1,-2],[2,1]],'2_3':[[0,0],[2,0],[-1,0],[2,-1],[-1,2]],'3_0':[[0,0],[1,0],[-2,0],[1,2],[-2,-1]],'1_0':[[0,0],[2,0],[-1,0],[2,-1],[-1,2]],'2_1':[[0,0],[1,0],[-2,0],[1,2],[-2,-1]],'3_2':[[0,0],[-2,0],[1,0],[-2,1],[1,-2]],'0_3':[[0,0],[-1,0],[2,0],[-1,-2],[2,1]]};
const SCORE_TBL={none:{0:0,1:100,2:300,3:500,4:800},mini:{0:100,1:200,2:400,3:400},regular:{0:400,1:800,2:1200,3:1600}};

function lighten(h,a){let r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return`rgb(${Math.min(255,r+a)},${Math.min(255,g+a)},${Math.min(255,b+a)})`}
function darken(h,a){let r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return`rgb(${Math.max(0,r-a)},${Math.max(0,g-a)},${Math.max(0,b-a)})`}
function roundRect(c,x,y,w,h,r){c.beginPath();c.moveTo(x+r,y);c.lineTo(x+w-r,y);c.arcTo(x+w,y,x+w,y+r,r);c.lineTo(x+w,y+h-r);c.arcTo(x+w,y+h,x+w-r,y+h,r);c.lineTo(x+r,y+h);c.arcTo(x,y+h,x,y+h-r,r);c.lineTo(x,y+r);c.arcTo(x,y,x+r,y,r);c.closePath()}

class AudioManager{
  constructor(){this.ctx=null;this.playing=false;this.mTimeout=null}
  init(){this.ctx=new(window.AudioContext||window.webkitAudioContext)();this.mg=this.ctx.createGain();this.mg.gain.value=.12;this.mg.connect(this.ctx.destination);this.sg=this.ctx.createGain();this.sg.gain.value=.3;this.sg.connect(this.ctx.destination)}
  resume(){if(this.ctx&&this.ctx.state==='suspended')this.ctx.resume()}
  tone(f,d,t='square',v=.1,dest=null,st=null){
    if(!this.ctx)return;
    const s=st||this.ctx.currentTime,o=this.ctx.createOscillator(),g=this.ctx.createGain();
    o.type=t;o.frequency.setValueAtTime(f,s);g.gain.setValueAtTime(v,s);g.gain.exponentialRampToValueAtTime(.001,s+d);
    o.connect(g);g.connect(dest||this.sg);o.start(s);o.stop(s+d)
  }
  noise(d,v=.05,st=null){
    if(!this.ctx)return;
    const s=st||this.ctx.currentTime,bs=this.ctx.sampleRate*d,ba=this.ctx.createBuffer(1,bs,this.ctx.sampleRate),da=ba.getChannelData(0);
    for(let i=0;i<bs;i++)da[i]=Math.random()*2-1;
    const src=this.ctx.createBufferSource();src.buffer=ba;
    const g=this.ctx.createGain();g.gain.setValueAtTime(v,s);g.gain.exponentialRampToValueAtTime(.001,s+d);
    src.connect(g);g.connect(this.sg);src.start(s)
  }
  sMove(){this.tone(220,.03,'square',.04)}
  sRotate(){this.tone(350,.04,'square',.04)}
  sSoft(){this.tone(150,.02,'triangle',.025)}
  sHard(){this.tone(70,.12,'triangle',.12);this.noise(.08,.04)}
  sLock(){this.tone(110,.05,'triangle',.07)}
  sHold(){this.tone(440,.05,'sine',.04);this.tone(660,.07,'sine',.04)}
  sLine(){[523,659,784].forEach((f,i)=>setTimeout(()=>this.tone(f,.1,'square',.06),i*50))}
  sTetris(){[523,659,784,1047,1319].forEach((f,i)=>setTimeout(()=>this.tone(f,.15,'square',.08),i*55))}
  sTSpin(){[880,740,622,523].forEach((f,i)=>setTimeout(()=>this.tone(f,.1,'sawtooth',.05),i*40))}
  sLevelUp(){[523,659,784,1047].forEach((f,i)=>setTimeout(()=>{this.tone(f,.15,'square',.08);this.tone(f/2,.15,'triangle',.04)},i*75))}
  sCombo(n){const b=440+n*40;[b,b*1.2,b*1.5].forEach((f,i)=>setTimeout(()=>this.tone(f,.08,'square',.05),i*35))}
  sGameover(){[440,392,349,330,294,262].forEach((f,i)=>setTimeout(()=>this.tone(f,.3,'triangle',.06),i*150))}
  startMusic(){if(!this.ctx||this.playing)return;this.playing=true;this._ml()}
  stopMusic(){this.playing=false;clearTimeout(this.mTimeout)}
  _ml(){
    if(!this.playing||!this.ctx)return;
    const bd=60/140/2,now=this.ctx.currentTime+.05;
    const mel=[659,494,523,587,523,494,440,440,523,659,587,523,494,523,587,659,523,440,440,0];
    const bas=[165,0,165,0,165,0,110,0,110,0,110,0,82,0,82,0,82,0,165,0];
    for(let i=0;i<mel.length;i++){
      if(mel[i]>0)this.tone(mel[i],bd*.8,'square',.035,this.mg,now+i*bd);
      if(bas[i]>0)this.tone(bas[i],bd*.8,'triangle',.025,this.mg,now+i*bd)
    }
    this.mTimeout=setTimeout(()=>this._ml(),mel.length*bd*1000)
  }
}

class ParticleSys{
  constructor(){this.items=[];this.bg=[]}
  initBg(w,h){this.bg=[];for(let i=0;i<40;i++)this.bg.push({x:Math.random()*w,y:Math.random()*h,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,sz:1+Math.random()*2,a:.05+Math.random()*.15,col:['#0ff','#f0f','#00f','#0f0'][Math.random()*4|0]})}
  burst(x,y,color,n=8){for(let i=0;i<n;i++)this.items.push({x,y,vx:(Math.random()-.5)*12,vy:(Math.random()-.5)*12-3,col:color,life:1,dec:.015+Math.random()*.025,sz:2+Math.random()*5})}
  update(dt){
    for(let i=this.items.length-1;i>=0;i--){const p=this.items[i];p.x+=p.vx;p.y+=p.vy;p.vy+=.15;p.life-=p.dec;if(p.life<=0)this.items.splice(i,1)}
    for(const p of this.bg){p.x+=p.vx;p.y+=p.vy;if(p.x<0)p.x+=400;if(p.x>400)p.x-=400;if(p.y<0)p.y+=600;if(p.y>600)p.y-=600}
  }
  drawBg(ctx){ctx.save();for(const p of this.bg){ctx.globalAlpha=p.a;ctx.fillStyle=p.col;ctx.beginPath();ctx.arc(p.x,p.y,p.sz,0,Math.PI*2);ctx.fill()}ctx.restore()}
  draw(ctx){ctx.save();ctx.globalCompositeOperation='lighter';for(const p of this.items){ctx.globalAlpha=Math.max(0,p.life);ctx.fillStyle=p.col;ctx.beginPath();ctx.arc(p.x,p.y,p.sz*Math.max(.3,p.life),0,Math.PI*2);ctx.fill()}ctx.restore()}
}

const popups=[];
function addPop(text,x,y,col='#fff',sz=16){popups.push({text,x,y,col,sz,life:1,vy:-1.5})}
function updatePops(){for(let i=popups.length-1;i>=0;i--){const p=popups[i];p.y+=p.vy;p.life-=.012;if(p.life<=0)popups.splice(i,1)}}
function drawPops(ctx){ctx.save();ctx.textAlign='center';for(const p of popups){ctx.globalAlpha=Math.max(0,p.life);ctx.font=`bold ${p.sz}px 'Segoe UI',sans-serif`;ctx.fillStyle=p.col;ctx.shadowColor=p.col;ctx.shadowBlur=10;ctx.fillText(p.text,p.x,p.y);ctx.shadowBlur=0}ctx.restore()}

function drawCell(c,x,y,s,col,alpha=1){
  c.save();c.globalAlpha=alpha;c.shadowColor=col;c.shadowBlur=10;c.shadowOffsetX=0;c.shadowOffsetY=0;
  const g=c.createLinearGradient(x,y,x+s,y+s);g.addColorStop(0,lighten(col,50));g.addColorStop(.5,col);g.addColorStop(1,darken(col,35));
  c.fillStyle=g;roundRect(c,x+1,y+1,s-2,s-2,3);c.fill();
  c.shadowBlur=0;
  const hg=c.createLinearGradient(x,y,x,y+s*.35);hg.addColorStop(0,'rgba(255,255,255,.35)');hg.addColorStop(1,'rgba(255,255,255,0)');
  c.fillStyle=hg;roundRect(c,x+2,y+2,s-4,s*.3,2);c.fill();
  const sg=c.createLinearGradient(x,y+s*.6,x,y+s);sg.addColorStop(0,'rgba(0,0,0,0)');sg.addColorStop(1,'rgba(0,0,0,.3)');
  c.fillStyle=sg;roundRect(c,x+2,y+s*.55,s-4,s*.38,2);c.fill();
  c.strokeStyle=darken(col,25);c.lineWidth=1;roundRect(c,x+1,y+1,s-2,s-2,3);c.stroke();
  c.restore()
}
function drawGhost(c,x,y,s,col){c.save();c.globalAlpha=.18;c.fillStyle=col;roundRect(c,x+1,y+1,s-2,s-2,3);c.fill();c.globalAlpha=.4;c.strokeStyle=col;c.lineWidth=1.5;roundRect(c,x+1,y+1,s-2,s-2,3);c.stroke();c.restore()}

const boardCanvas=document.getElementById('board-canvas'),boardCtx=boardCanvas.getContext('2d');
const holdCanvas=document.getElementById('hold-canvas'),holdCtx=holdCanvas.getContext('2d');
const nextCanvas=document.getElementById('next-canvas'),nextCtx=nextCanvas.getContext('2d');
const ps=new ParticleSys(),audio=new AudioManager();
let shakeI=0,shakeD=0;

function initCanvases(){
  const w=window.innerWidth,h=window.innerHeight;
  const maxW=Math.min(w*.38,320),maxH=h*.6;
  CELL_SIZE=Math.floor(Math.min(maxW/10,maxH/20,35));
  boardCanvas.width=10*CELL_SIZE;boardCanvas.height=20*CELL_SIZE;
  boardCanvas.style.width=10*CELL_SIZE+'px';boardCanvas.style.height=20*CELL_SIZE+'px';
  holdCanvas.width=80;holdCanvas.height=80;
  nextCanvas.width=80;nextCanvas.height=400;
  ps.initBg(boardCanvas.width,boardCanvas.height);
}

const game={
  state:'ready',board:[],current:null,holdType:null,holdUsed:false,queue:[],bag:[],
  score:0,dispScore:0,level:1,lines:0,combo:-1,b2b:0,time:0,
  dropTimer:0,lockTimer:0,lockResets:0,clearTimer:0,clearDur:500,clearRows:[],
  lastWasRot:false,dasTimer:0,dasDir:0,dasActive:false,arrTimer:0,input:{left:false,right:false,down:false},
  piecesPlaced:0,

  init(){this.showStart();window.addEventListener('resize',initCanvases);initCanvases();this.bindInput()},

  showStart(){
    document.getElementById('overlay-title').textContent='TETRIS';
    document.getElementById('overlay-message').innerHTML='Experience the classic puzzle game<br>with stunning visuals & effects';
    document.getElementById('overlay-stats').innerHTML='';
    document.getElementById('overlay-btn').textContent='START';
    document.getElementById('overlay').classList.remove('hidden');
  },

  newGame(){
    this.board=Array.from({length:TOTAL_ROWS},()=>new Array(COLS).fill(null));
    this.score=0;this.dispScore=0;this.level=1;this.lines=0;this.combo=-1;this.b2b=0;this.time=0;
    this.holdType=null;this.holdUsed=false;this.queue=[];this.bag=[];this.piecesPlaced=0;
    this.dropTimer=0;this.lockTimer=0;this.lockResets=0;this.clearTimer=0;this.clearRows=[];
    this.lastWasRot=false;this.dasDir=0;this.dasActive=false;
    for(let i=0;i<5;i++)this.queue.push(this.getFromBag());
    this.spawnPiece();this.state='playing';this.updateUI();
    if(!audio.playing){audio.startMusic()}
  },

  getFromBag(){
    if(!this.bag.length)this.bag=this.shuffle(['I','O','T','S','Z','J','L']);
    return this.bag.pop();
  },
  shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.random()*(i+1)|0;[a[i],a[j]]=[a[j],a[i]]}return a},

  spawnPiece(){
    const type=this.queue.shift();this.queue.push(this.getFromBag());
    const sp=SPAWN[type];this.current={type,rotation:0,x:sp.x,y:sp.y};
    if(!this.canPlace(this.current)){this.gameOver();return}
    this.holdUsed=false;this.dropTimer=0;this.lockTimer=0;this.lockResets=0;this.lastWasRot=false;
    this.renderHold();this.renderNext();
  },

  canPlace(p){
    const m=PIECES[p.type][p.rotation];
    for(let r=0;r<m.length;r++)for(let c=0;c<m[r].length;c++)if(m[r][c]){
      const bx=p.x+c,by=p.y+r;
      if(bx<0||bx>=COLS||by>=TOTAL_ROWS)return false;
      if(by>=0&&this.board[by][bx])return false;
    }
    return true;
  },

  movePiece(dx,dy){
    if(!this.current)return false;
    const test={...this.current,x:this.current.x+dx,y:this.current.y+dy};
    if(!this.canPlace(test))return false;
    this.current.x+=dx;this.current.y+=dy;
    this.lastWasRot=false;
    if(this.isGrounded()){if(this.lockResets<15){this.lockTimer=0;this.lockResets++}}else{this.lockTimer=0;this.lockResets=0}
    return true;
  },

  rotatePiece(dir){
    if(!this.current||this.current.type==='O')return;
    const from=this.current.rotation,to=(from+(dir===1?1:3))%4,key=from+'_'+to;
    const kicks=this.current.type==='I'?K_I[key]:K_JLSTZ[key];
    for(const[dx,dy]of kicks){
      const test={...this.current,rotation:to,x:this.current.x+dx,y:this.current.y+dy};
      if(this.canPlace(test)){
        this.current.rotation=to;this.current.x+=dx;this.current.y+=dy;
        this.lastWasRot=true;
        if(this.isGrounded()){if(this.lockResets<15){this.lockTimer=0;this.lockResets++}}else{this.lockTimer=0;this.lockResets=0}
        audio.sRotate();return;
      }
    }
  },

  hardDrop(){
    if(!this.current||this.state!=='playing')return;
    let n=0;while(this.movePiece(0,1))n++;
    this.score+=n*2;audio.sHard();shakeI=5;shakeD=100;this.lockPiece();
  },

  holdPiece(){
    if(!this.current||this.holdUsed||this.state!=='playing')return;
    const ct=this.current.type;
    if(this.holdType===null){this.holdType=ct;this.spawnPiece()}
    else{const ht=this.holdType;this.holdType=ct;const sp=SPAWN[ht];
      this.current={type:ht,rotation:0,x:sp.x,y:sp.y};
      if(!this.canPlace(this.current)){this.gameOver();return}
    }
    this.holdUsed=true;this.lastWasRot=false;this.dropTimer=0;this.lockTimer=0;this.lockResets=0;
    audio.sHold();this.renderHold();this.renderNext();
  },

  isGrounded(){return this.current&&!this.canPlace({...this.current,y:this.current.y+1})},

  getGhostY(){
    if(!this.current)return 0;
    let gy=this.current.y;while(this.canPlace({...this.current,y:gy+1}))gy++;return gy;
  },

  checkTSpin(){
    if(!this.current||this.current.type!=='T'||!this.lastWasRot)return 'none';
    const px=this.current.x,py=this.current.y;
    const corners=[[px,py],[px+2,py],[px,py+2],[px+2,py+2]];
    const filled=corners.map(([cx,cy])=>{
      if(cx<0||cx>=COLS||cy>=TOTAL_ROWS)return true;
      if(cy<0)return false;
      return !!this.board[cy][cx];
    });
    const cnt=filled.filter(Boolean).length;
    if(cnt<3)return 'none';
    const s=this.current.rotation;
    const front={0:[0,1],1:[1,3],2:[2,3],3:[0,2]}[s];
    return filled[front[0]]&&filled[front[1]]?'regular':'mini';
  },

  checkLines(){const r=[];for(let y=0;y<TOTAL_ROWS;y++)if(this.board[y].every(c=>c))r.push(y);return r},

  lockPiece(){
    if(!this.current)return;
    const m=PIECES[this.current.type][this.current.rotation];
    let allHidden=true;
    for(let r=0;r<m.length;r++)for(let c=0;c<m[r].length;c++)if(m[r][c]){
      const bx=this.current.x+c,by=this.current.y+r;
      if(by>=0&&by<TOTAL_ROWS)this.board[by][bx]=this.current.type;
      if(by>=HIDDEN_ROWS)allHidden=false;
    }
    this.current=null;this.piecesPlaced++;
    if(allHidden){this.gameOver();return}

    const tspin=this.checkTSpin();
    const clrRows=this.checkLines();
    const nl=clrRows.length;
    const isDiff=(tspin!=='none'&&nl>0)||nl===4;

    if(nl>0)this.combo++;else this.combo=-1;

    const prevB2B=this.b2b;
    if(nl>0){if(isDiff)this.b2b++;else this.b2b=0}

    const sk=tspin!=='none'?tspin:'none';
    let base=SCORE_TBL[sk][nl]||0;
    let tot=base*this.level;
    if(isDiff&&prevB2B>0)tot=Math.floor(tot*1.5);
    if(this.combo>0)tot+=50*this.combo*this.level;
    this.score+=tot;

    if(tspin!=='none')audio.sTSpin();
    if(nl===4)audio.sTetris();else if(nl>0)audio.sLine();
    if(this.combo>1)audio.sCombo(this.combo);

    const cx=boardCanvas.width/2,cy=boardCanvas.height/2;
    if(nl>0||tspin!=='none'){
      let act='';
      if(tspin==='regular')act=nl>0?`T-SPIN ${['','SINGLE','DOUBLE','TRIPLE'][nl]}`:'T-SPIN';
      else if(tspin==='mini')act=nl>0?`MINI T-SPIN ${['','SINGLE','DOUBLE'][nl]}`:'MINI T-SPIN';
      else if(nl===4)act='TETRIS!';
      else if(nl>0)act=['','SINGLE','DOUBLE','TRIPLE'][nl];
      if(act){const ac=tspin!=='none'?'#f0f':nl===4?'#ff0':'#fff';addPop(act,cx,cy-40,ac,22)}
      addPop('+'+tot,cx,cy-5,'#0ff',26);
      if(this.combo>1)addPop('COMBO ×'+this.combo,cx,cy+30,'#ff0',14);
      if(this.b2b>0&&isDiff)addPop('B2B ×'+this.b2b,cx,cy+52,'#f0f',14);
    }

    const pc=this.board.every(r=>r.every(c=>!c));
    if(pc&&nl>0){this.score+=800*this.level;addPop('PERFECT CLEAR!',cx,cy+75,'#0f0',18)}

    this.lines+=nl;
    const newLvl=Math.floor(this.lines/10)+1;
    if(newLvl>this.level){this.level=newLvl;audio.sLevelUp();addPop('LEVEL UP!',cx,cy-80,'#ff0',20)}

    if(nl>0){
      this.clearRows=clrRows;this.clearTimer=this.clearDur;this.state='clearing';
    }else{audio.sLock();this.spawnPiece()}
    this.updateUI();
  },

  finishClear(){
    for(const row of this.clearRows){
      const drawY=(row-HIDDEN_ROWS)*CELL_SIZE;
      for(let x=0;x<COLS;x++){
        const col=COLORS[this.board[row][x]]||'#fff';
        ps.burst(x*CELL_SIZE+CELL_SIZE/2,drawY+CELL_SIZE/2,col,10);
      }
    }
    this.clearRows.sort((a,b)=>b-a);
    for(const row of this.clearRows){this.board.splice(row,1);this.board.unshift(new Array(COLS).fill(null))}
    this.clearRows=[];this.spawnPiece();this.state='playing';this.updateUI();
  },

  gravityDelay(){return Math.max(30,1000*Math.pow(.8,this.level-1))},

  update(dt){
    if(this.state==='playing'){
      this.time+=dt;
      const di=this.input.down?Math.min(50,this.gravityDelay()/20):this.gravityDelay();
      this.dropTimer+=dt;
      while(this.dropTimer>=di){
        this.dropTimer-=di;
        if(!this.movePiece(0,1)){
          if(this.lockTimer===0){this.lockTimer=.001;this.lockResets=0}
          break;
        }else if(this.input.down){this.score++;audio.sSoft()}
      }
      if(this.lockTimer>0){
        if(!this.isGrounded()){this.lockTimer=0;this.lockResets=0}
        else{this.lockTimer+=dt;if(this.lockTimer>=500)this.lockPiece()}
      }
      this.updateDAS(dt);
    }else if(this.state==='clearing'){
      this.clearTimer-=dt;
      if(this.clearTimer<=0)this.finishClear();
    }
    if(this.dispScore<this.score){const d=this.score-this.dispScore;this.dispScore+=Math.max(1,Math.ceil(d*.12));if(this.dispScore>this.score)this.dispScore=this.score}
    if(shakeD>0){shakeD-=dt;if(shakeD<=0)shakeI=0}
    ps.update(dt);updatePops();
    this.updateUI();
  },

  updateDAS(dt){
    if(this.dasDir!==0&&!this.dasActive){this.dasTimer-=dt;if(this.dasTimer<=0){this.dasActive=true;this.arrTimer=0}}
    if(this.dasActive){this.arrTimer-=dt;if(this.arrTimer<=0){this.movePiece(this.dasDir,0);audio.sMove();this.arrTimer=40}}
  },

  render(){
    const c=boardCtx,w=boardCanvas.width,h=boardCanvas.height;
    c.save();
    if(shakeI>0){c.translate((Math.random()-.5)*shakeI*2,(Math.random()-.5)*shakeI*2)}
    c.fillStyle='#080818';c.fillRect(0,0,w,h);
    ps.drawBg(c);
    // Grid
    c.strokeStyle='rgba(255,255,255,.03)';c.lineWidth=1;
    for(let i=1;i<COLS;i++){c.beginPath();c.moveTo(i*CELL_SIZE,0);c.lineTo(i*CELL_SIZE,h);c.stroke()}
    for(let i=1;i<VISIBLE_ROWS;i++){c.beginPath();c.moveTo(0,i*CELL_SIZE);c.lineTo(w,i*CELL_SIZE);c.stroke()}
    // Board
    for(let y=HIDDEN_ROWS;y<TOTAL_ROWS;y++)for(let x=0;x<COLS;x++)if(this.board[y][x]){
      const drawY=(y-HIDDEN_ROWS)*CELL_SIZE;
      if(drawY>=0&&drawY<h)drawCell(c,x*CELL_SIZE,drawY,CELL_SIZE,COLORS[this.board[y][x]]);
    }
    // Clearing animation
    if(this.state==='clearing'){
      const prog=1-this.clearTimer/this.clearDur;
      const flash=prog<.5?Math.sin(prog*Math.PI*12)*.5+.5:1-(prog-.5)/.5;
      for(const row of this.clearRows){
        const drawY=(row-HIDDEN_ROWS)*CELL_SIZE;
        if(drawY<0||drawY>=h)continue;
        for(let x=0;x<COLS;x++){
          c.save();c.globalAlpha=Math.max(0,flash*.8);c.fillStyle='#fff';
          c.fillRect(x*CELL_SIZE,drawY,CELL_SIZE,CELL_SIZE);c.restore();
        }
      }
    }
    // Current piece
    if(this.current&&this.state==='playing'){
      const type=this.current.type,col=COLORS[type],m=PIECES[type][this.current.rotation];
      // Ghost
      const gy=this.getGhostY();
      for(let r=0;r<m.length;r++)for(let cl=0;cl<m[r].length;cl++)if(m[r][cl]){
        const bx=this.current.x+cl,by=gy+r,drawY=(by-HIDDEN_ROWS)*CELL_SIZE;
        if(bx>=0&&bx<COLS&&drawY>=0&&drawY<h)drawGhost(c,bx*CELL_SIZE,drawY,CELL_SIZE,col);
      }
      // Piece
      for(let r=0;r<m.length;r++)for(let cl=0;cl<m[r].length;cl++)if(m[r][cl]){
        const bx=this.current.x+cl,by=this.current.y+r,drawY=(by-HIDDEN_ROWS)*CELL_SIZE;
        if(bx>=0&&bx<COLS&&drawY>=0&&drawY<h)drawCell(c,bx*CELL_SIZE,drawY,CELL_SIZE,col);
      }
    }
    ps.draw(c);drawPops(c);
    // Border overlay
    c.strokeStyle='rgba(0,255,255,.08)';c.lineWidth=1;
    c.strokeRect(0,0,w,h);
    c.restore();
    this.renderHold();this.renderNext();
  },

  renderPreview(ctx,type,slotW,slotH,cx,cy,cellSz){
    if(!type)return;
    const m=PIECES[type][0],col=COLORS[type];
    let mnX=9,mxX=-1,mnY=9,mxY=-1;
    for(let r=0;r<m.length;r++)for(let c=0;c<m[r].length;c++)if(m[r][c]){
      mnX=Math.min(mnX,c);mxX=Math.max(mxX,c);mnY=Math.min(mnY,r);mxY=Math.max(mxY,r);
    }
    const pw=(mxX-mnX+1)*cellSz,ph=(mxY-mnY+1)*cellSz;
    const sx=cx-pw/2,sy=cy-ph/2;
    for(let r=mnY;r<=mxY;r++)for(let c=mnX;c<=mxX;c++)if(m[r][c]){
      drawCell(ctx,sx+(c-mnX)*cellSz,sy+(r-mnY)*cellSz,cellSz,col);
    }
  },

  renderHold(){
    const c=holdCtx;c.clearRect(0,0,80,80);
    c.fillStyle='rgba(255,255,255,.02)';c.fillRect(0,0,80,80);
    if(this.holdType)this.renderPreview(c,this.holdType,80,80,40,40,18);
  },

  renderNext(){
    const c=nextCtx;c.clearRect(0,0,80,400);
    c.fillStyle='rgba(255,255,255,.02)';c.fillRect(0,0,80,400);
    for(let i=0;i<Math.min(5,this.queue.length);i++){
      const sz=i===0?20:15;
      this.renderPreview(c,this.queue[i],80,80,40,i*80+40,sz);
    }
  },

  updateUI(){
    document.getElementById('score').textContent=this.dispScore;
    document.getElementById('level').textContent=this.level;
    document.getElementById('lines').textContent=this.lines;
    const m=Math.floor(this.time/60000),s=Math.floor(this.time/1000)%60;
    document.getElementById('time').textContent=m+':'+String(s).padStart(2,'0');
    document.getElementById('combo').textContent=this.combo>0?this.combo:'-';
    document.getElementById('b2b').textContent=this.b2b>0?this.b2b:'-';
  },

  gameOver(){
    this.state='gameover';audio.stopMusic();audio.sGameover();
    const m=Math.floor(this.time/60000),s=Math.floor(this.time/1000)%60;
    document.getElementById('overlay-title').textContent='GAME OVER';
    document.getElementById('overlay-message').textContent='Your run has ended';
    document.getElementById('overlay-stats').innerHTML=
      `<div class="stat"><span class="stat-label">SCORE</span><span class="stat-value">${this.score}</span></div>
       <div class="stat"><span class="stat-label">LEVEL</span><span class="stat-value">${this.level}</span></div>
       <div class="stat"><span class="stat-label">LINES</span><span class="stat-value">${this.lines}</span></div>
       <div class="stat"><span class="stat-label">TIME</span><span class="stat-value">${m}:${String(s).padStart(2,'0')}</span></div>
       <div class="stat"><span class="stat-label">PIECES</span><span class="stat-value">${this.piecesPlaced}</span></div>`;
    document.getElementById('overlay-btn').textContent='RESTART';
    document.getElementById('overlay').classList.remove('hidden');
  },

  togglePause(){
    if(this.state==='playing'){
      this.state='paused';
      document.getElementById('overlay-title').textContent='PAUSED';
      document.getElementById('overlay-message').textContent='Game is paused';
      document.getElementById('overlay-stats').innerHTML='';
      document.getElementById('overlay-btn').textContent='RESUME';
      document.getElementById('overlay').classList.remove('hidden');
    }else if(this.state==='paused'){
      this.state='playing';
      document.getElementById('overlay').classList.add('hidden');
    }
  },

  bindInput(){
    document.addEventListener('keydown',e=>{
      if(e.repeat)return;
      const k=e.key.toLowerCase();
      if(k==='p'||k==='escape'){if(this.state==='playing'||this.state==='paused'){this.togglePause();e.preventDefault()}return}
      if(this.state!=='playing')return;
      switch(k){
        case'arrowleft':case'a':this.input.left=true;this.dasDir=-1;this.dasActive=false;this.dasTimer=150;
          if(this.movePiece(-1,0))audio.sMove();e.preventDefault();break;
        case'arrowright':case'd':this.input.right=true;this.dasDir=1;this.dasActive=false;this.dasTimer=150;
          if(this.movePiece(1,0))audio.sMove();e.preventDefault();break;
        case'arrowdown':case's':this.input.down=true;e.preventDefault();break;
        case'arrowup':case'w':this.rotatePiece(1);e.preventDefault();break;
        case'z':case'q':this.rotatePiece(-1);e.preventDefault();break;
        case' ':this.hardDrop();e.preventDefault();break;
        case'c':case'shift':this.holdPiece();e.preventDefault();break;
      }
    });
    document.addEventListener('keyup',e=>{
      const k=e.key.toLowerCase();
      switch(k){
        case'arrowleft':case'a':this.input.left=false;if(this.dasDir===-1){this.dasDir=this.input.right?1:0;this.dasActive=false;this.dasTimer=150}break;
        case'arrowright':case'd':this.input.right=false;if(this.dasDir===1){this.dasDir=this.input.left?-1:0;this.dasActive=false;this.dasTimer=150}break;
        case'arrowdown':case's':this.input.down=false;break;
      }
    });

    // Touch buttons
    const bindBtn=(id,down,up)=>{
      const el=document.getElementById(id);
      el.addEventListener('touchstart',e=>{e.preventDefault();el.classList.add('active');down()},{passive:false});
      el.addEventListener('touchend',e=>{e.preventDefault();el.classList.remove('active');if(up)up()},{passive:false});
      el.addEventListener('touchcancel',e=>{el.classList.remove('active');if(up)up()});
      el.addEventListener('mousedown',e=>{e.preventDefault();el.classList.add('active');down()});
      el.addEventListener('mouseup',e=>{el.classList.remove('active');if(up)up()});
    };

    bindBtn('btn-left',()=>{
      if(this.state!=='playing')return;
      this.dasDir=-1;this.dasActive=false;this.dasTimer=150;
      if(this.movePiece(-1,0))audio.sMove();
    },()=>{if(this.dasDir===-1)this.dasDir=0});

    bindBtn('btn-right',()=>{
      if(this.state!=='playing')return;
      this.dasDir=1;this.dasActive=false;this.dasTimer=150;
      if(this.movePiece(1,0))audio.sMove();
    },()=>{if(this.dasDir===1)this.dasDir=0});

    bindBtn('btn-soft',()=>{
      if(this.state!=='playing')return;
      this.input.down=true;
    },()=>{this.input.down=false});

    bindBtn('btn-rcw',()=>{
      if(this.state==='playing')this.rotatePiece(1);
    });

    bindBtn('btn-rccw',()=>{
      if(this.state==='playing')this.rotatePiece(-1);
    });

    bindBtn('btn-drop',()=>{
      if(this.state==='playing')this.hardDrop();
    });

    bindBtn('btn-hold',()=>{
      if(this.state==='playing')this.holdPiece();
    });

    // Overlay button
    document.getElementById('overlay-btn').addEventListener('click',()=>{
      audio.resume();
      if(this.state==='ready'){
        if(!audio.ctx)audio.init();
        audio.startMusic();this.hideOverlay();this.newGame();
      }else if(this.state==='paused'){
        this.togglePause();
      }else if(this.state==='gameover'){
        this.hideOverlay();this.newGame();
      }
    });
  },

  hideOverlay(){document.getElementById('overlay').classList.add('hidden')}
};

game.init();

let lastTime=0;
function loop(ts){
  const dt=Math.min(ts-lastTime,50);lastTime=ts;
  game.update(dt);game.render();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
</script>
</body>
</html>
```



This is a fully-featured, PS4-style Tetris game with all the extras in a single HTML file. Here's what's included:

**Gameplay Features:**
- Classic 7 tetromino pieces with SRS rotation & wall kicks
- Hold piece, 5-piece next queue with 7-bag randomizer
- Ghost piece, hard/soft drop
- T-spin detection (regular & mini) with corner checking
- Full scoring: single/double/triple/tetris, T-spins, combos, B2B ×1.5, perfect clear bonus
- Levels every 10 lines with exponential speed increase
- Lock delay (500ms with 15 max resets)
- DAS/ARR for smooth horizontal movement

**Visual Effects (PS4 Tetris Effect style):**
- Neon glowing pieces with gradient fills & highlights
- Particle bursts on line clears (additive blending)
- Flashing line clear animation
- Screen shake on hard drops
- Floating score popups with glow
- Background particle atmosphere
- Glassmorphism UI panels
- Animated title glow
- Smooth score counter animation

**Audio (Web Audio API):**
- Chiptune background music (Korobeiniki melody)
- 12+ distinct sound effects (move, rotate, drop, lock, line clear, tetris, T-spin, level up, combo, hold, game over)

**Controls:**
- Keyboard: arrows, Z, space, C, P
- Mobile: touch buttons with DAS support

**UI/UX:**
- Stats tracking (score, level, lines, time, combo, B2B)
- Start, pause, and game over screens
- Responsive layout for desktop & mobile