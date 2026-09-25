#!/usr/bin/env python3
"""Train the Heaven's Eye read/write codec + addressing on REAL optical-disorder images
(Zenodo optical-PUF speckle, 1000x 500x500 16-bit tif). The real speckle IS the intrinsic
imperfection field (unique, random, 'full of information'); data is written as reversible
modulation RELATIVE to it (write mode A) and read back conditioned on the same field.
Also a registration test on real texture (addressing). GPU-only. Writes results_real/.
Run: CUDA_VISIBLE_DEVICES=0 python3 real_train.py
"""
import os, io, time, json, math, zipfile, shutil, urllib.request, ssl
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from PIL import Image
assert torch.cuda.is_available(),'GPU required'
DEV=torch.device('cuda'); torch.backends.cudnn.benchmark=True
RES=Path('results_real'); RES.mkdir(exist_ok=True); IMGD=Path('realdata/imgs'); IMGD.mkdir(parents=True,exist_ok=True)
SNR_LO,SNR_HI=1200.0,14000.0
# ---------- fetch real images (subset) ----------
def fetch(n_max=300):
    have=sorted(IMGD.glob('*.npy'))
    if len(have)>=n_max: return
    url="https://zenodo.org/records/8377156/files/Responses%20for%20multi-level%20HPUF.zip?download=1"
    zp=Path('realdata/resp.zip')
    if not zp.exists():
        print('downloading real dataset...'); ctx=ssl.create_default_context()
        with urllib.request.urlopen(url,timeout=600,context=ctx) as r,open(zp,'wb') as f: shutil.copyfileobj(r,f)
    z=zipfile.ZipFile(zp); tifs=[x for x in z.namelist() if x.lower().endswith('.tif')][:n_max]
    for i,nme in enumerate(tifs):
        im=Image.open(io.BytesIO(z.read(nme))); a=np.asarray(im,dtype=np.float32)
        a=(a-a.min())/(a.max()-a.min()+1e-9)          # per-image normalize to [0,1]
        np.save(IMGD/f'img_{i:04d}.npy',a.astype(np.float32))
    z.close(); zp.unlink(); print('extracted',len(tifs),'real images')
fetch(300)
IMGS=np.stack([np.load(p) for p in sorted(IMGD.glob('*.npy'))]); print('real images tensor',IMGS.shape)
IMG_T=torch.from_numpy(IMGS).to(DEV)   # (N,500,500) in [0,1]
# save a montage of 12 real images (properly normalized) for viewing
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,4,figsize=(9,7))
for k,ax in enumerate(axs.ravel()): ax.imshow(IMGS[k],cmap='gray'); ax.axis('off'); ax.set_title(f'real disorder {k}',fontsize=7)
fig.suptitle('Real optical-disorder images (Zenodo optical-PUF speckle) used for training',fontsize=10)
fig.savefig(RES/'real_montage.png',dpi=130,bbox_inches='tight'); plt.close(fig)
# ---------- sample a real disorder field at cell / render resolution ----------
def sample_disorder(B,n,ppv):
    idx=torch.randint(0,IMG_T.shape[0],(B,),device=DEV)
    # random 300x300 crop from each 500x500, then resize
    crops=torch.empty(B,1,300,300,device=DEV)
    for b in range(B):
        y=int(torch.randint(0,200,(1,))); x=int(torch.randint(0,200,(1,)))
        crops[b,0]=IMG_T[idx[b],y:y+300,x:x+300]
    d_cell=F.interpolate(crops,size=(n,n),mode='area')              # (B,1,n,n) the addressing MAP
    d_img =F.interpolate(crops,size=(n*ppv,n*ppv),mode='area')      # (B,1,H,W) render-res disorder
    return d_cell,d_img
def snr_plane(se,shape):
    v=(math.log10(se)-math.log10(SNR_LO))/(math.log10(SNR_HI)-math.log10(SNR_LO)); return torch.full(shape,float(v),device=DEV)
def gk(sigma,r=3):
    ax=torch.arange(-r,r+1.0); xx,yy=torch.meshgrid(ax,ax,indexing='ij'); k=torch.exp(-(xx**2+yy**2)/(2*sigma*sigma)); return (k/k.sum()).view(1,1,2*r+1,2*r+1)
class Chan(nn.Module):
    def __init__(self,ppv=3,sigma=1.4,rn=2.0,fw=15000.0,adc=12,contrast=0.8,dis_gain=0.35,dis_off=0.15):
        super().__init__(); self.ppv=ppv; self.rn=rn; self.fw=fw; self.adc=adc; self.contrast=contrast
        self.dg=dis_gain; self.do=dis_off; self.register_buffer('psf',gk(sigma)); self.pad=self.psf.shape[-1]//2
    def forward(self,levels,d_img,se):
        base=se*(1-self.contrast); lvl=base+levels*(se-base)
        lvl=lvl*(1+self.dg*(d_img-0.5))+self.do*se*d_img          # REAL disorder perturbs appearance (relative modulation)
        up=F.interpolate(lvl,scale_factor=self.ppv,mode='nearest') if levels.shape[-1]!=d_img.shape[-1] else lvl
        img=F.conv2d(F.pad(up,(self.pad,)*4,mode='reflect'),self.psf)
        img=img+torch.sqrt(torch.clamp(img,min=0)+1e-6)*torch.randn_like(img)+self.rn*torch.randn_like(img)
        q=self.fw/(2**self.adc); img=img+((img/q).round()*q-img).detach()
        return (img-base)/(se-base+1e-9)
class Enc(nn.Module):
    def __init__(self,M,ch=32):
        super().__init__(); self.emb=nn.Embedding(M,ch)
        self.net=nn.Sequential(nn.Conv2d(ch+2,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                               nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),nn.Conv2d(ch,1,3,padding=1))
    def forward(self,s,d,sp): return torch.sigmoid(self.net(torch.cat([self.emb(s).permute(0,3,1,2),d,sp],1)))
class FiLM(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.f=nn.Sequential(nn.Conv2d(2,ch,1),nn.ReLU(),nn.Conv2d(ch,2*ch,1))
    def forward(self,h,c): ga,be=self.f(c).chunk(2,1); return h*(1+ga)+be
class Dec(nn.Module):
    def __init__(self,M,ppv,ch=32):
        super().__init__(); self.ppv=ppv
        self.body=nn.Sequential(nn.Conv2d(1,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU())
        self.film=FiLM(ch); self.head=nn.Sequential(nn.Conv2d(ch,ch,1),nn.ReLU(),nn.Conv2d(ch,M,1))
    def forward(self,img,d,sp): h=F.avg_pool2d(self.body(img),self.ppv); return self.head(self.film(h,torch.cat([d,sp],1)))
def berf(logits,sym,bits):
    pred=logits.argmax(1); ar=torch.arange(bits,device=DEV).view(1,-1,1,1)
    return (((pred.unsqueeze(1)>>ar)&1)!=((sym.unsqueeze(1)>>ar)&1)).float().mean().item()
def train(bits,steps=2000,n=24,B=96,ch=32,use_encoder=True):
    M=2**bits; chan=Chan().to(DEV); enc=Enc(M,ch).to(DEV); dec=Dec(M,chan.ppv,ch).to(DEV)
    params=sum(p.numel() for p in dec.parameters())+(sum(p.numel() for p in enc.parameters()) if use_encoder else 0)
    mods=list(dec.parameters())+(list(enc.parameters()) if use_encoder else [])
    opt=torch.optim.AdamW(mods,lr=2e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps); lf=nn.CrossEntropyLoss()
    for s in range(steps):
        se=float(10**np.random.uniform(math.log10(SNR_LO),math.log10(SNR_HI)))
        sym=torch.randint(0,M,(B,n,n),device=DEV); d_cell,d_img=sample_disorder(B,n,chan.ppv); sp=snr_plane(se,(B,1,n,n))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lvl=enc(sym,d_cell,sp) if use_encoder else (sym.float()/(M-1)).unsqueeze(1)
            lvl_img=F.interpolate(lvl,scale_factor=chan.ppv,mode='nearest')
            loss=lf(dec(chan(lvl_img,d_img,se),d_cell,sp),sym)
        loss.backward(); nn.utils.clip_grad_norm_(mods,1.0); opt.step(); sched.step()
    return enc.eval(),dec.eval(),chan,dict(bits=bits,M=M,use_encoder=use_encoder,params=params,model_MB=round(params*4/1e6,3))
@torch.no_grad()
def evalat(enc,dec,chan,meta,se,n=24,B=256):
    M=meta['M']; sym=torch.randint(0,M,(B,n,n),device=DEV); d_cell,d_img=sample_disorder(B,n,chan.ppv); sp=snr_plane(se,(B,1,n,n))
    lvl=enc(sym,d_cell,sp) if meta['use_encoder'] else (sym.float()/(M-1)).unsqueeze(1)
    lvl_img=F.interpolate(lvl,scale_factor=chan.ppv,mode='nearest')
    return round(berf(dec(chan(lvl_img,d_img,se),d_cell,sp),sym,meta['bits']),6)
# ---------- addressing: phase-correlation registration on REAL texture ----------
@torch.no_grad()
def registration_test(trials=200,n=64,maxshift=6):
    errs=[]
    for _ in range(trials):
        i=int(torch.randint(0,IMG_T.shape[0],(1,))); y=int(torch.randint(50,380,(1,))); x=int(torch.randint(50,380,(1,)))
        ref=IMG_T[i,y:y+n,x:x+n]; dx=int(torch.randint(-maxshift,maxshift+1,(1,))); dy=int(torch.randint(-maxshift,maxshift+1,(1,)))
        mov=IMG_T[i,y+dy:y+dy+n,x+dx:x+dx+n]+0.05*torch.randn(n,n,device=DEV)
        Fa=torch.fft.rfft2(ref); Fb=torch.fft.rfft2(mov); R=Fa*Fb.conj(); R=R/(R.abs()+1e-8)
        cc=torch.fft.irfft2(R,s=(n,n)); pk=torch.argmax(cc).item(); py,px=pk//n,pk%n
        py=py-n if py>n//2 else py; px=px-n if px>n//2 else px
        errs.append(((px-dx)**2+(py-dy)**2)**0.5)
    return float(np.mean(errs)),float(np.median(errs))
if __name__=='__main__':
    t0=time.time(); SNRs=[1200,2000,3500,6000,9000,14000]; rows=[]; MODELS={}
    for ue in (True,False):
        for b in (2,3):
            e,d,c,m=train(b,steps=2000,use_encoder=ue); MODELS[('L' if ue else 'I',b)]=(e,d,c,m)
            for se in SNRs: rows.append({**m,'signal_e':se,'ber':evalat(e,d,c,m,se)})
            print(('L' if ue else 'I'),'b=%d ber@14000=%.4f'%(b,rows[-1]['ber']))
    reg_mean,reg_med=registration_test(); print('registration on REAL texture: mean %.3f px, median %.3f px'%(reg_mean,reg_med))
    # round trip on real disorder (bits=2, learned)
    e,d,c,m=MODELS[('L',2)]
    def b2s(x,bits):
        bs=np.unpackbits(np.frombuffer(x,np.uint8)); pad=(-len(bs))%bits
        if pad: bs=np.concatenate([bs,np.zeros(pad,np.uint8)])
        w=(1<<np.arange(bits-1,-1,-1)); return (bs.reshape(-1,bits)*w).sum(1)
    def s2b(sym,bits,nb):
        mm=((sym[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8); return np.packbits(mm.reshape(-1)[:nb*8]).tobytes()
    msg=b"HEAVEN'S EYE on real optical disorder."; n=24; s=b2s(msg,2); grid=np.zeros(n*n,np.int64); grid[:min(len(s),n*n)]=s[:n*n]; grid=grid.reshape(n,n)
    with torch.no_grad():
        sym=torch.from_numpy(grid).long().unsqueeze(0).to(DEV); d_cell,d_img=sample_disorder(1,n,c.ppv); sp=snr_plane(6000,(1,1,n,n))
        lvl=e(sym,d_cell,sp); lvl_img=F.interpolate(lvl,scale_factor=c.ppv,mode='nearest'); cam=c(lvl_img,d_img,6000)
        pred=d(cam,d_cell,sp).argmax(1).cpu().numpy().ravel()
    nb=min(len(msg),(n*n*2)//8); rec=s2b(pred,2,nb); ba=float(np.mean(np.frombuffer(rec,np.uint8)==np.frombuffer(msg[:nb],np.uint8)))
    # figure: real disorder | written levels | camera image | errors
    fig,axs=plt.subplots(1,4,figsize=(12,3.3))
    axs[0].imshow(d_img.squeeze().cpu(),cmap='gray'); axs[0].set_title('real disorder field'); axs[0].axis('off')
    axs[1].imshow(lvl.squeeze().cpu(),cmap='magma'); axs[1].set_title('learned write levels'); axs[1].axis('off')
    axs[2].imshow(cam.squeeze().cpu(),cmap='gray'); axs[2].set_title('camera image'); axs[2].axis('off')
    axs[3].imshow((grid!=pred.reshape(n,n)).astype(float),cmap='Reds'); axs[3].set_title('symbol errors'); axs[3].axis('off')
    fig.suptitle("Round trip on REAL disorder: byte acc %.2f"%ba,fontsize=10); fig.savefig(RES/'real_roundtrip.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    # BER figure
    import pandas as pd; df=pd.DataFrame(rows); col={2:'#4c72b0',3:'#55a868'}
    fig,ax=plt.subplots(figsize=(6,4))
    for b in (2,3):
        dl=df[(df.bits==b)&(df.use_encoder)].sort_values('signal_e'); di=df[(df.bits==b)&(~df.use_encoder)].sort_values('signal_e')
        ax.plot(di.signal_e,di.ber.clip(1e-6),'s--',color=col[b],alpha=.5,label=f'{b}b identity')
        ax.plot(dl.signal_e,dl.ber.clip(1e-6),'o-',color=col[b],label=f'{b}b learned')
    ax.axhline(1e-2,color='k',ls=':'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('signal electrons'); ax.set_ylabel('BER')
    ax.set_title('RW codec on REAL optical-disorder images'); ax.legend(fontsize=7); fig.savefig(RES/'real_ber_snr.png',dpi=130,bbox_inches='tight'); plt.close(fig)
    out={'dataset':'Zenodo 8377156 optical-PUF speckle (real)','n_images':int(IMGS.shape[0]),
         'runs':rows,'registration_px':{'mean':round(reg_mean,3),'median':round(reg_med,3)},
         'roundtrip':{'byte_accuracy':round(ba,4),'message':msg.decode('latin-1'),'recovered':rec.decode('latin-1','replace')},
         'minutes':round((time.time()-t0)/60,1)}
    json.dump(out,open(RES/'real_codec.json','w'),indent=2)
    print('roundtrip byte acc',ba,'| registration mean px',round(reg_mean,3),'| done',out['minutes'],'min')
