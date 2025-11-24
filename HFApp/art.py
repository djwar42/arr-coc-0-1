"""
ARR-COC Header Art Component

Butterfly wing ASCII art with 3D WebGL spinning orb and shimmer/morph effects.
"""

import gradio as gr


def create_header():
    """
    Creates the ARR-COC header with:
    - Title and subtitle
    - Butterfly wing ASCII art (left and right)
    - 3D spinning blue orb (WebGL)
    - Character shimmer animation
    - Character morphing effects

    Returns:
        gr.HTML: Gradio HTML component with the complete header
    """
    return gr.HTML("""
        <style>
        #arr-header{text-align:center;padding:10px 0 0 0;font-family:monospace}
        #arr-title{color:#4a9eff;font-size:42px;font-weight:bold;letter-spacing:8px;margin-bottom:5px;text-shadow:0 0 20px rgba(74,158,255,.5)}
        #arr-subtitle{
          color:#888;
          font-size:16px;
          font-style:italic;
          margin-bottom:0;
          letter-spacing:1px;
          background:linear-gradient(90deg,#888 0%,#888 40%,#fff 50%,#888 60%,#888 100%);
          background-size:200% 100%;
          background-clip:text;
          -webkit-background-clip:text;
          -webkit-text-fill-color:transparent;
          animation:shimmer 8s ease-in-out infinite;
        }
        @keyframes shimmer{
          0%{background-position:200% center}
          20%{background-position:-100% center}
          100%{background-position:-100% center}
        }
        #arr-container{display:flex;justify-content:center;align-items:center;gap:20px;max-width:1400px;margin:0 auto}
        #arr-left,#arr-right{flex:1 1 300px;font-family:'Courier New',monospace;font-size:11px;color:#888;line-height:1.4}
        #arr-canvas-wrapper{flex:0 0 auto}
        .wing-text{display:inline-block}
        .wing-text span{
          display:inline-block;
          animation:charShimmer 2.5s ease-in-out infinite;
        }
        @keyframes charShimmer{
          0%,100%{color:#bbb;text-shadow:0 0 3px rgba(255,255,255,0.2)}
          50%{color:#fff;text-shadow:0 0 12px rgba(255,255,255,0.8)}
        }
        </style>
        <div id="arr-header">
            <h1 id="arr-title">ARR-COC</h1>
            <p id="arr-subtitle">'What you see changes what you see'</p>
        </div>
        <div id="arr-container">
            <div id="arr-left">
                <pre style="margin:0;background:transparent;text-align:right;padding-right:20px;font-size:10px" id="wing-left">       ⢰⣆
      ⠘ARR⣧⡀⢳⡀
     ⠀⠹Adaptive⡟⣝⢾⣷⡄
    ⢿Relevance⡶⣤⣀⡀
   ⠀⠙Realization⢯⣿⠯⣒⠦⢄⣈
                </pre>
            </div>
            <div id="arr-canvas-wrapper">
                <canvas id="arr-canvas" width="300" height="300"></canvas>
            </div>
            <div id="arr-right">
                <pre style="margin:0;background:transparent;text-align:left;padding-left:20px;font-size:10px" id="wing-right">⠰⡄
⢳⡀ARR⣧⡀
⣷⡄Adaptive⡟⣝
⣀⡀Relevance⡶⣤
⣈⢄⠦⣒⠯Realization⣿⢯
                </pre>
            </div>
        </div>
        <script>
        (()=>{
          const charMap={
            A:['A','Λ','∆','Α'],R:['R','Я','ℜ','Ʀ'],C:['C','Ͻ','Ↄ','Ϲ'],O:['O','Ο','◯','Ø'],
            M:['M','Μ','ᴍ','Ϻ'],i:['i','ı','í','ï'],c:['c','ϲ','ⲥ','ς'],r:['r','ʳ','ɾ','ř'],
            o:['o','ο','º','ó'],s:['s','ѕ','ꜱ','ş'],p:['p','ρ','þ','ᵖ'],e:['e','ε','ҽ','е'],
            l:['l','ӏ','ǀ','ℓ'],a:['a','α','ą','а'],n:['n','ո','ή','ñ'],V:['V','Ѵ','ᴠ','Ʋ'],
            v:['v','ν','ѵ','ᵛ'],t:['t','τ','ţ','ť'],d:['d','ԁ','ɖ','ď'],z:['z','ʐ','ź','ž']
          };
          const wrapWords=(el)=>{
            let html=el.innerHTML;
            ['ARR','Adaptive','Relevance','Realization'].forEach(word=>{
              const wrapped='<span class="wing-text">'+word.split('').map(ch=>
                `<span style="animation-delay:${Math.random()*2.5}s">${ch}</span>`
              ).join('')+'</span>';
              html=html.replace(new RegExp(word,'g'),wrapped);
            });
            el.innerHTML=html;
          };
          const morphChar=(span)=>{
            const orig=span.textContent;
            if(!charMap[orig])return;
            const alts=charMap[orig];
            const newChar=alts[Math.floor(Math.random()*alts.length)];
            span.textContent=newChar;
            setTimeout(()=>span.textContent=orig,Math.random()*1500+1000);
          };
          const globalMorphEvent=()=>{
            const allChars=document.querySelectorAll('.wing-text span');
            const eligible=Array.from(allChars).filter(ch=>charMap[ch.textContent]);
            const count=Math.floor(Math.random()*3)+4;
            for(let i=0;i<count&&eligible.length;i++){
              const idx=Math.floor(Math.random()*eligible.length);
              morphChar(eligible[idx]);
              eligible.splice(idx,1);
            }
            const nextDelay=Math.random()*1500+2000;
            setTimeout(globalMorphEvent,nextDelay);
          };
          const left=document.getElementById('wing-left');
          const right=document.getElementById('wing-right');
          wrapWords(left);wrapWords(right);
          setTimeout(globalMorphEvent,Math.random()*3000+2000);
        })();
        </script>
        <script>
        (()=>{
          const c=document.getElementById('arr-canvas'),gl=c.getContext('webgl',{alpha:true,antialias:true});
          if(!gl)return;
          const vs=`attribute vec3 pos;attribute vec3 norm;uniform mat4 mvp;uniform mat4 model;varying vec3 vNorm;
          void main(){vNorm=normalize((model*vec4(norm,0.0)).xyz);gl_Position=mvp*vec4(pos,1.0);}`;
          const fs=`precision mediump float;varying vec3 vNorm;
          void main(){float intensity=pow(0.6-dot(vNorm,vec3(0.0,0.0,1.0)),4.5);
          gl_FragColor=vec4(0.3*intensity,0.6*intensity,1.0*intensity,intensity);}`;
          const sh=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s};
          const prog=gl.createProgram();
          gl.attachShader(prog,sh(gl.VERTEX_SHADER,vs));
          gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,fs));
          gl.linkProgram(prog);gl.useProgram(prog);
          const verts=[],norms=[],t=(1+Math.sqrt(5))/2;
          const addTri=(a,b,c)=>{verts.push(...a,...b,...c);const n=(i,j,k)=>{const d=Math.sqrt(i*i+j*j+k*k);return[i/d,j/d,k/d]};
          const na=n(...a),nb=n(...b),nc=n(...c);norms.push(...na,...nb,...nc)};
          const v=[[0,1,t],[0,-1,t],[0,1,-t],[0,-1,-t],[1,t,0],[-1,t,0],[1,-t,0],[-1,-t,0],[t,0,1],[-t,0,1],[t,0,-1],[-t,0,-1]]
          .map(p=>{const l=Math.sqrt(p[0]**2+p[1]**2+p[2]**2);return p.map(x=>x/l*.5)});
          const f=[[0,8,4],[0,5,9],[9,5,11],[4,11,10],[2,4,10],[6,2,10],[8,6,7],[9,8,1]];
          f.forEach(([a,b,c])=>addTri(v[a],v[b],v[c]));
          const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(verts),gl.STATIC_DRAW);
          const nb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(norms),gl.STATIC_DRAW);
          const aPos=gl.getAttribLocation(prog,'pos'),aNorm=gl.getAttribLocation(prog,'norm');
          const uMVP=gl.getUniformLocation(prog,'mvp'),uModel=gl.getUniformLocation(prog,'model');
          const perspective=(fov,aspect,near,far)=>{const f=1/Math.tan(fov/2),nf=1/(near-far);
          return[f/aspect,0,0,0,0,f,0,0,0,0,(far+near)*nf,-1,0,0,2*far*near*nf,0]};
          const translate=(x,y,z)=>[1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1];
          const rotateY=(a)=>{const c=Math.cos(a),s=Math.sin(a);return[c,0,s,0,0,1,0,0,-s,0,c,0,0,0,0,1]};
          const multiply=(a,b)=>{const m=[];for(let i=0;i<4;i++)for(let j=0;j<4;j++){let sum=0;
          for(let k=0;k<4;k++)sum+=a[k*4+j]*b[i*4+k];m.push(sum)}return m};
          gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);gl.clearColor(.04,.04,.04,1);
          let angle=0;
          const render=()=>{
            angle+=.01;
            gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
            const proj=perspective(Math.PI/4,1,.1,100);
            const view=translate(0,0,-2);
            const model=rotateY(angle);
            const mvp=multiply(proj,multiply(view,model));
            gl.uniformMatrix4fv(uMVP,false,mvp);
            gl.uniformMatrix4fv(uModel,false,model);
            gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(aPos);
            gl.bindBuffer(gl.ARRAY_BUFFER,nb);gl.vertexAttribPointer(aNorm,3,gl.FLOAT,false,0,0);gl.enableVertexAttribArray(aNorm);
            gl.drawArrays(gl.TRIANGLES,0,verts.length/3);
            requestAnimationFrame(render)
          };
          render();
        })();
        </script>
        """)
