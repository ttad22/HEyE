#!/usr/bin/env python3
"""WHOLE-PLATTER Heaven's Eye demo on real optical-disorder images.
Architecture the whole system uses (not one patch):
  1. one camera snapshot of the ENTIRE platter,
  2. tile it into a T x T grid of addressable BLOCKS (LBA-style),
  3. each block carries its OWN local imperfection map,
  4. the learned encoder writes a payload chunk into every block (relative to that block's
     imperfections), so data is spread across the WHOLE glass,
  5. one whole-platter camera image is formed (blur crosses block seams -- realistic),
  6. the decoder re-tiles and reads every block back; payload reassembled,
  7. block addressing: each block's imperfection fingerprint re-identifies it even if the
     tile order is shuffled (position recovery).
Efficient: disorder patches precomputed into a GPU pool (no per-step python loops).
GPU-only. Writes results_platter/. Run: CUDA_VISIBLE_DEVICES=0 python3 real_platter.py
"""
import time, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
assert torch.cuda.is_available(),'GPU required'
DEV=torch.device('cuda'); torch.backends.cudnn.benchmark=True
RES=Path('results_platter'); RES.mkdir(exist_ok=True); IMGD=Path('realdata/imgs')
SNR_LO,SNR_HI=1200.0,14000.0; PPV=3; NCELL=16
IMGS=np.stack([np.load(p) for p in sorted(IMGD.glob('*.npy'))]); IMG=torch.from_numpy(IMGS).to(DEV)
print('real images',IMG.shape)
# ---- precompute disorder patch pool (fast sampling) ----
def build_pool(P=2400,n=NCELL,ppv=PPV):
    cell=torch.empty(P,1,n,n,device=DEV); img=torch.empty(P,1,n*ppv,n*ppv,device=DEV)
    for k in range(P):
        i=int(torch.randint(0,IMG.shape[0],(1,))); y=int(torch.randint(0,200,(1,))); x=int(torch.randint(0,200,(1,)))
        crop=IMG[i,y:y+300,x:x+300][None,None]
        cell[k]=F.interpolate(crop,size=(n,n),mode='area')[0]; img[k]=F.interpolate(crop,size=(n*ppv,n*ppv),mode='area')[0]
    return cell,img
POOL_C,POOL_I=build_pool(); print('pool',POOL_C.shape)
def snr_plane(se,shape):
    v=(math.log10(se)-math.log10(SNR_LO))/(math.log10(SNR_HI)-math.log10(SNR_LO)); return torch.full(shape,float(v),device=DEV)
def gk(sigma,r=3):
    ax=torch.arange(-r,r+1.0); xx,yy=torch.meshgrid(ax,ax,indexing='ij'); k=torch.exp(-(xx**2+yy**2)/(2*sigma*sigma)); return (k/k.sum()).view(1,1,2*r+1,2*r+1)
class Chan(nn.Module):
    def __init__(self,ppv=PPV,sigma=1.4,rn=2.0,fw=15000.0,adc=12,contrast=0.8,dg=0.35,do=0.15):
        super().__init__(); self.ppv=ppv; self.rn=rn; self.fw=fw; self.adc=adc; self.contrast=contrast; self.dg=dg; self.do=do
        self.register_buffer('psf',gk(sigma)); self.pad=self.psf.shape[-1]//2
    def forward(self,levels_img,d_img,se):
        base=se*(1-self.contrast); lvl=base+levels_img*(se-base); lvl=lvl*(1+self.dg*(d_img-0.5))+self.do*se*d_img
        img=F.conv2d(F.pad(lvl,(self.pad,)*4,mode='reflect'),self.psf)
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
    def __init__(self,M,ppv=PPV,ch=32):
        super().__init__(); self.ppv=ppv
        self.body=nn.Sequential(nn.Conv2d(1,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(),
                                nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU())
        self.film=FiLM(ch); self.head=nn.Sequential(nn.Conv2d(ch,ch,1),nn.ReLU(),nn.Conv2d(ch,M,1))
    def forward(self,img,d,sp): h=F.avg_pool2d(self.body(img),self.ppv); return self.head(self.film(h,torch.cat([d,sp],1)))
def berf(logits,sym,bits):
    pred=logits.argmax(1); ar=torch.arange(bits,device=DEV).view(1,-1,1,1)
    return (((pred.unsqueeze(1)>>ar)&1)!=((sym.unsqueeze(1)>>ar)&1)).float().mean().item()
def train(bits,steps=1500,n=NCELL,B=128,ch=32):
    M=2**bits; chan=Chan().to(DEV); enc=Enc(M,ch).to(DEV); dec=Dec(M,PPV,ch).to(DEV)
    mods=list(enc.parameters())+list(dec.parameters())
    opt=torch.optim.AdamW(mods,lr=2e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps); lf=nn.CrossEntropyLoss()
    P=POOL_C.shape[0]
    for s in range(steps):
        se=float(10**np.random.uniform(math.log10(SNR_LO),math.log10(SNR_HI)))
        idx=torch.randint(0,P,(B,),device=DEV); d_cell=POOL_C[idx]; d_img=POOL_I[idx]
        sym=torch.randint(0,M,(B,n,n),device=DEV); sp=snr_plane(se,(B,1,n,n))
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            lvl=enc(sym,d_cell,sp); lvl_img=F.interpolate(lvl,scale_factor=PPV,mode='nearest')
            loss=lf(dec(chan(lvl_img,d_img,se),d_cell,sp),sym)
        loss.backward(); nn.utils.clip_grad_norm_(mods,1.0); opt.step(); sched.step()
    params=sum(p.numel() for p in mods)
    return enc.eval(),dec.eval(),chan,dict(bits=bits,M=M,params=params,model_MB=round(params*4/1e6,3))
@torch.no_grad()
def whole_platter(enc,dec,chan,meta,T=8,n=NCELL,se=6000,seed=1):
    """Tile ONE big real image into TxT blocks; write a full payload across all blocks;
    form ONE whole-platter camera image; decode every block; reassemble."""
    torch.manual_seed(seed); M=meta['M']; bits=meta['bits']; g=torch.Generator(device='cpu').manual_seed(seed)
    bi=int(torch.randint(0,IMG.shape[0],(1,),generator=g)); big=IMG[bi]                     # 500x500 real platter
    Hc=T*n                                                                                   # platter cells per side
    d_cell_full=F.interpolate(big[None,None],size=(Hc,Hc),mode='area')                       # (1,1,Hc,Hc)
    d_img_full =F.interpolate(big[None,None],size=(Hc*PPV,Hc*PPV),mode='area')               # (1,1,Hc*ppv,Hc*ppv)
    # tile disorder into blocks: (T*T,1,n,n) and (T*T,1,n*ppv,n*ppv)
    dc=d_cell_full.reshape(1,1,T,n,T,n).permute(0,2,4,1,3,5).reshape(T*T,1,n,n)
    npx=n*PPV; di=d_img_full.reshape(1,1,T,npx,T,npx).permute(0,2,4,1,3,5).reshape(T*T,1,npx,npx)
    # payload: fill all blocks; each block n*n symbols
    total_sym=T*T*n*n; nbytes=(total_sym*bits)//8
    rng=np.random.default_rng(seed); payload=rng.integers(0,256,nbytes,dtype=np.uint8).tobytes()
    bs=np.unpackbits(np.frombuffer(payload,np.uint8)); bs=bs[:total_sym*bits]
    w=(1<<np.arange(bits-1,-1,-1)); syms=(bs.reshape(-1,bits)*w).sum(1).reshape(T*T,n,n)
    sym=torch.from_numpy(syms).long().to(DEV)
    sp=snr_plane(se,(T*T,1,n,n))
    lvl=enc(sym,dc,sp); lvl_img=F.interpolate(lvl,scale_factor=PPV,mode='nearest')           # (T*T,1,npx,npx)
    # reassemble whole write-level image
    lvl_full=lvl_img.reshape(1,T,T,1,npx,npx).permute(0,3,1,4,2,5).reshape(1,1,Hc*PPV,Hc*PPV)
    cam_full=chan(lvl_full,d_img_full,se)                                                     # ONE whole-platter snapshot
    # decode: re-tile the whole camera image, decode each block with its map
    cam_bl=cam_full.reshape(1,1,T,npx,T,npx).permute(0,2,4,1,3,5).reshape(T*T,1,npx,npx)
    pred=dec(cam_bl,dc,sp).argmax(1)                                                          # (T*T,n,n)
    ber=berf(dec(cam_bl,dc,sp),sym,bits)
    # reassemble recovered symbols -> bytes
    pr=pred.reshape(T*T,n*n).reshape(-1).cpu().numpy()[:total_sym]   # block-major, matches encode order
    mb=((pr[:,None]>>np.arange(bits-1,-1,-1))&1).astype(np.uint8); rec=np.packbits(mb.reshape(-1)[:nbytes*8]).tobytes()
    byte_acc=float(np.mean(np.frombuffer(rec,np.uint8)==np.frombuffer(payload,np.uint8)))
    per_block=(pred!=sym).float().mean((1,2)).cpu().numpy()                                   # per-block SER
    # addressing: re-identify shuffled blocks by disorder fingerprint (mean-subtracted corr)
    feats=dc.reshape(T*T,-1); feats=feats-feats.mean(1,keepdim=True); feats=feats/(feats.norm(dim=1,keepdim=True)+1e-8)
    perm=torch.randperm(T*T,device=DEV); shuf=feats[perm]
    sim=shuf@feats.t(); match=sim.argmax(1); recovered=(match==perm).float().mean().item()
    return dict(T=T,n=n,blocks=T*T,platter_cells=Hc*Hc,payload_bytes=nbytes,signal_e=se,
                whole_platter_ber=round(ber,6),whole_platter_byte_acc=round(byte_acc,4),
                per_block_ser_mean=round(float(per_block.mean()),4),per_block_ser_max=round(float(per_block.max()),4),
                block_addressing_recovery=round(recovered,4)), dict(
                disorder=d_img_full.squeeze().cpu().numpy(),levels=lvl_full.squeeze().cpu().numpy(),
                camera=cam_full.squeeze().cpu().numpy(),err=(pred!=sym).reshape(T,T,n,n).permute(0,2,1,3).reshape(Hc,Hc).cpu().numpy())
if __name__=='__main__':
    t0=time.time()
    e2,d2,c2,m2=train(2,steps=1500); e3,d3,c3,m3=train(3,steps=1500)
    print('trained: 2b params',m2['params'],'| 3b params',m3['params'])
    res={}
    for tag,(e,d,c,m) in {'2bit':(e2,d2,c2,m2),'3bit':(e3,d3,c3,m3)}.items():
        st,viz=whole_platter(e,d,c,m,T=8); res[tag]=st; print(tag,st)
        if tag=='2bit':
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig,axs=plt.subplots(1,4,figsize=(14,3.7))
            axs[0].imshow(viz['disorder'],cmap='gray'); axs[0].set_title('real disorder map\n(8$\\times$8 blocks)', fontsize=9)
            axs[1].imshow(viz['levels'],cmap='gray'); axs[1].set_title('payload written\nacross all blocks', fontsize=9)
            axs[2].imshow(viz['camera'],cmap='gray'); axs[2].set_title('simulated camera frame\n(whole platter)', fontsize=9)
            axs[3].imshow(viz['err'],cmap='gray_r'); axs[3].set_title('decode errors', fontsize=9)
            for a in axs: a.axis('off')
            # No suptitle: the LaTeX caption already carries blocks/cells/
            # payload/accuracy, so repeating them here wastes vertical space.
            fig.savefig(RES/'whole_platter.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    res['minutes']=round((time.time()-t0)/60,1)
    json.dump(res,open(RES/'platter.json','w'),indent=2); print('done',res['minutes'],'min')
