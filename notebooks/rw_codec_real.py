#!/usr/bin/env python3
"""Realistic-flaw RW codec: same learned Deep-JSCC read/write codec, but the intrinsic
imperfection field is now SPARSE DISCRETE SCATTERERS (bubbles/voids/scratches that
scatter light into bright points -- the darkfield 'stars'), not a dense Gaussian.
Grounded in Schott TIE-28 (bubbles/inclusions in optical glass) and darkfield/optical-PUF
imaging. Write mode A: reversible modulation addressed RELATIVE to this sparse field.
SNR-adaptive; learned vs identity write; imperfection-conditioning ablation; round trip.
GPU-only. Writes results_rw_real/. Run: CUDA_VISIBLE_DEVICES=1 python3 rw_codec_real.py
"""
import time, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
assert torch.cuda.is_available(),'GPU required'
DEV=torch.device('cuda'); torch.backends.cudnn.benchmark=True
RES=Path('results_rw_real'); RES.mkdir(exist_ok=True)
SNR_LO,SNR_HI=1200.0,14000.0
def snr_plane(se,shape):
    v=(math.log10(se)-math.log10(SNR_LO))/(math.log10(SNR_HI)-math.log10(SNR_LO)); return torch.full(shape,float(v),device=DEV)
def gk(sigma,r=3):
    ax=torch.arange(-r,r+1.0); xx,yy=torch.meshgrid(ax,ax,indexing='ij'); k=torch.exp(-(xx**2+yy**2)/(2*sigma*sigma)); return (k/k.sum()).view(1,1,2*r+1,2*r+1)
class Chan(nn.Module):
    """Optical channel with SPARSE scatterer flaws."""
    def __init__(self,ppv=3,sigma=1.4,rn=2.0,fw=15000.0,adc=12,contrast=0.8):
        super().__init__(); self.ppv=ppv; self.rn=rn; self.fw=fw; self.adc=adc; self.contrast=contrast
        self.register_buffer('psf',gk(sigma)); self.pad=self.psf.shape[-1]//2
    def imp(self,B,n,density=0.06,gvar=0.03):
        # g: near-flat gain (small variation); o: sparse bright scatterers (darkfield stars)
        g=1.0+gvar*torch.randn(B,1,n,n,device=DEV)
        mask=(torch.rand(B,1,n,n,device=DEV)<density).float()
        o=mask*(0.3+0.7*torch.rand(B,1,n,n,device=DEV))
        return g,o
    def forward(self,levels,g,o,se):
        base=se*(1-self.contrast); lvl=base+levels*(se-base); lvl=lvl*g+o*se
        up=F.interpolate(lvl,scale_factor=self.ppv,mode='nearest')
        img=F.conv2d(F.pad(up,(self.pad,)*4,mode='reflect'),self.psf)
        img=img+torch.sqrt(torch.clamp(img,min=0)+1e-6)*torch.randn_like(img)+self.rn*torch.randn_like(img)
        q=self.fw/(2**self.adc); img=img+((img/q).round()*q-img).detach()
        return (img-base)/(se-base+1e-9)
class Enc(nn.Module):
    def __init__(self,M,ch=32):
        super().__init__(); self.emb=nn.Embedding(M,ch)
        self.net=nn.Sequential(nn.Conv2d(ch+3,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                               nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),nn.Conv2d(ch,1,3,padding=1))
    def forward(self,s,g,o,sp): return torch.sigmoid(self.net(torch.cat([self.emb(s).permute(0,3,1,2),g,o,sp],1)))
class FiLM(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.f=nn.Sequential(nn.Conv2d(3,ch,1),nn.ReLU(),nn.Conv2d(ch,2*ch,1))
    def forward(self,h,c): ga,be=self.f(c).chunk(2,1); return h*(1+ga)+be
class Dec(nn.Module):
    def __init__(self,M,ppv,ch=32):
        super().__init__(); self.ppv=ppv
        self.body=nn.Sequential(nn.Conv2d(1,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU())
        self.film=FiLM(ch); self.head=nn.Sequential(nn.Conv2d(ch,ch,1),nn.ReLU(),nn.Conv2d(ch,M,1))
    def forward(self,img,g,o,sp): h=F.avg_pool2d(self.body(img),self.ppv); return self.head(self.film(h,torch.cat([g,o,sp],1)))
def berf(logits,sym,bits):
    pred=logits.argmax(1); ar=torch.arange(bits,device=DEV).view(1,-1,1,1)
    return (((pred.unsqueeze(1)>>ar)&1)!=((sym.unsqueeze(1)>>ar)&1)).float().mean().item()
def train(bits,steps=2000,n=24,B=128,ch=32,use_encoder=True,cond=True):
    M=2**bits; chan=Chan().to(DEV); enc=Enc(M,ch).to(DEV); dec=Dec(M,chan.ppv,ch).to(DEV)
    params=sum(p.numel() for p in dec.parameters())+(sum(p.numel() for p in enc.parameters()) if use_encoder else 0)
    mods=list(dec.parameters())+(list(enc.parameters()) if use_encoder else [])
    opt=torch.optim.AdamW(mods,lr=2e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps); lf=nn.CrossEntropyLoss()
    for s in range(steps):
        se=float(10**np.random.uniform(math.log10(SNR_LO),math.log10(SNR_HI)))
        sym=torch.randint(0,M,(B,n,n),device=DEV); g,o=chan.imp(B,n); sp=snr_plane(se,(B,1,n,n))
        ge,oe=(g,o) if cond else (torch.ones_like(g),torch.zeros_like(o))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lvl=enc(sym,ge,oe,sp) if use_encoder else (sym.float()/(M-1)).unsqueeze(1)
            loss=lf(dec(chan(lvl,g,o,se),ge,oe,sp),sym)
        loss.backward(); nn.utils.clip_grad_norm_(mods,1.0); opt.step(); sched.step()
    return enc.eval(),dec.eval(),chan,dict(bits=bits,M=M,use_encoder=use_encoder,cond=cond,params=params,model_MB=round(params*4/1e6,3))
@torch.no_grad()
def evalat(enc,dec,chan,meta,se,n=24,B=256):
    M=meta['M']; sym=torch.randint(0,M,(B,n,n),device=DEV); g,o=chan.imp(B,n); sp=snr_plane(se,(B,1,n,n))
    ge,oe=(g,o) if meta['cond'] else (torch.ones_like(g),torch.zeros_like(o))
    lvl=enc(sym,ge,oe,sp) if meta['use_encoder'] else (sym.float()/(M-1)).unsqueeze(1)
    return round(berf(dec(chan(lvl,g,o,se),ge,oe,sp),sym,meta['bits']),6)
if __name__=='__main__':
    t0=time.time(); BITS=[2,3,4]; SNRs=[1200,2000,3500,6000,9000,14000]; rows=[]; MODELS={}
    for ue in (True,False):
        for b in BITS:
            e,d,c,m=train(b,steps=2000,use_encoder=ue); MODELS[('L' if ue else 'I',b)]=(e,d,c,m)
            for se in SNRs: rows.append({**m,'signal_e':se,'ber':evalat(e,d,c,m,se)})
            print(('L' if ue else 'I'),'b=%d ber@14000=%.4f'%(b,rows[-1]['ber']))
    # ablation: no-cond at bits=3
    e,d,c,m=train(3,steps=2000,use_encoder=True,cond=False); abl={'no_cond_ber_b3_2600':evalat(e,d,c,m,2600)}
    e,d,c,m=MODELS[('L',3)]; abl['cond_ber_b3_2600']=evalat(e,d,c,m,2600)
    # round trip bits=2 (reuse learned)
    e,d,c,m=MODELS[('L',2)]
    def b2s(x,bits):
        bs=np.unpackbits(np.frombuffer(x,np.uint8)); pad=(-len(bs))%bits
        if pad: bs=np.concatenate([bs,np.zeros(pad,np.uint8)])
        w=(1<<np.arange(bits-1,-1,-1)); return (bs.reshape(-1,bits)*w).sum(1)
    def s2b(sym,bits,nb):
        mm=((sym[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8); return np.packbits(mm.reshape(-1)[:nb*8]).tobytes()
    msg=b"HEAVEN'S EYE on real glass flaws."; n=24; s=b2s(msg,2); grid=np.zeros(n*n,np.int64); grid[:min(len(s),n*n)]=s[:n*n]; grid=grid.reshape(n,n)
    with torch.no_grad():
        sym=torch.from_numpy(grid).long().unsqueeze(0).to(DEV); g,o=c.imp(1,n); sp=snr_plane(6000,(1,1,n,n))
        pred=d(c(e(sym,g,o,sp),g,o,6000),g,o,sp).argmax(1).cpu().numpy().ravel()
    nb=min(len(msg),(n*n*2)//8); rec=s2b(pred,2,nb); ba=float(np.mean(np.frombuffer(rec,np.uint8)==np.frombuffer(msg[:nb],np.uint8)))
    out={'flaw_model':'sparse_scatterers','density':0.06,'runs':rows,'ablation':abl,
         'roundtrip':{'byte_accuracy':round(ba,4),'message':msg.decode('latin-1'),'recovered':rec.decode('latin-1','replace')},
         'minutes':round((time.time()-t0)/60,1)}
    json.dump(out,open(RES/'codec_real.json','w'),indent=2)
    print('ablation',abl); print('roundtrip byte acc',ba); print('done',out['minutes'],'min')
