"""Can our real FFT+CNN actually LEARN to separate two insects that share a
fundamental frequency but differ in harmonics?

The synthetic demo (synth_overlap_negative.py) used a weak nearest-centroid
classifier. Here we train the ACTUAL models we deploy and measure whether they
pick up the harmonic difference the frequency-only view can't see.

Compared, across a sweep of the overtone-difference `gap`:
  * FREQ-ONLY   : logistic on the single fundamental-frequency value (baseline)
  * SpecCNN     : 2D CNN on a log-spectrogram (harmonics visible in the image)
  * LFFT-CNN    : our LearnableFFT front-end + conv1d stack (the deployed family)

If SpecCNN / LFFT-CNN accuracy rises with gap while FREQ-ONLY stays at chance,
our approach genuinely learns harmonics — which is the whole premise for one day
telling a chironomid from a mosquito at the same ~500 Hz.

Honest caveat: this is SYNTHETIC. It proves the models CAN use a harmonic
difference IF one exists; it does not measure whether real chironomids differ.

Run:
    python scripts/harmonic_learnability.py --epochs 25
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.synth_overlap_negative import make_set, SR, N  # reuse the synth
from scripts.improve_audio_detector_1d import LFFTDetector   # our deployed family
from src.utils.device import get_device
from src.utils.seed import set_seed


def logspec2d(x, n_fft=256, hop=64):
    win = np.hanning(n_fft)
    nfr = 1 + (len(x)-n_fft)//hop
    fr = np.stack([x[i*hop:i*hop+n_fft]*win for i in range(nfr)])
    S = np.log(np.abs(np.fft.rfft(fr, axis=1))**2 + 1e-8).T   # (F, T)
    return ((S - S.mean())/(S.std()+1e-8)).astype(np.float32)


def fundamental(x):
    Y = np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    fr = np.fft.rfftfreq(len(x), 1/SR); b = (fr>=200)&(fr<=1200)
    return fr[b][np.argmax(Y[b])]


class SpecCNN(nn.Module):
    def __init__(self, ch=(16,32,64)):
        super().__init__()
        layers, p = [], 1
        for c in ch:
            layers += [nn.Conv2d(p,c,3,padding=1), nn.BatchNorm2d(c), nn.ReLU(True), nn.MaxPool2d(2)]; p=c
        self.f=nn.Sequential(*layers); self.gap=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(ch[-1],2)
    def forward(self,x):
        if x.ndim==3: x=x.unsqueeze(1)
        return self.fc(self.gap(self.f(x)).flatten(1))


def train_torch(model, Xtr, ytr, Xte, yte, device, epochs, is1d=False):
    model=model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    lossf=nn.CrossEntropyLoss()
    Xtr_t=torch.as_tensor(Xtr,device=device); ytr_t=torch.as_tensor(ytr,device=device)
    Xte_t=torch.as_tensor(Xte,device=device)
    for _ in range(epochs):
        model.train(); perm=torch.randperm(len(Xtr_t),device=device)
        for i in range(0,len(perm),64):
            idx=perm[i:i+64]; xb=Xtr_t[idx]
            if is1d: xb=xb.unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            loss=lossf(model(xb),ytr_t[idx]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pr=[]
        for i in range(0,len(Xte_t),128):
            xb=Xte_t[i:i+128]
            if is1d: xb=xb.unsqueeze(1)
            pr.append(model(xb).argmax(-1).cpu().numpy())
    return float((np.concatenate(pr)==yte).mean())


def freq_only_acc(M, C):
    fm=np.array([fundamental(x) for x in M]); fc=np.array([fundamental(x) for x in C])
    X=np.concatenate([fm,fc]).reshape(-1,1); y=np.array([0]*len(fm)+[1]*len(fc))
    rng=np.random.default_rng(0); idx=rng.permutation(len(X)); X,y=X[idx],y[idx]
    cut=int(0.7*len(X)); thr=X[:cut][y[:cut]==0].mean()*0.5+X[:cut][y[:cut]==1].mean()*0.5
    # 1-feature threshold classifier
    pred=(X[cut:,0]>thr).astype(int)
    a=(pred==y[cut:]).mean(); return max(a,1-a)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--n",type=int,default=500); args=p.parse_args(argv)
    device=get_device(); print(f"== device {device}")
    print(f"{'gap':>5}{'freq-only':>11}{'SpecCNN':>10}{'LFFT-CNN':>10}")
    rows=[]
    for gap in [0.0,0.3,0.6,0.9]:
        set_seed(42)
        M=make_set("mosquito",args.n,gap,0); C=make_set("midge",args.n,gap,100000)
        y=np.array([0]*len(M)+[1]*len(C))
        rng=np.random.default_rng(0); idx=rng.permutation(len(y))
        raw=np.concatenate([M,C])[idx]; y=y[idx]; cut=int(0.7*len(y))
        # freq-only
        aF=freq_only_acc(M,C)
        # SpecCNN on 2d logspec
        S=np.array([logspec2d(x) for x in raw])
        aS=train_torch(SpecCNN(),S[:cut],y[:cut],S[cut:],y[cut:],device,args.epochs)
        # LFFT-CNN on raw 1d waveform (needs 1024 len; pad/truncate)
        W=raw.copy()
        if W.shape[1]<1024: W=np.pad(W,((0,0),(0,1024-W.shape[1])))
        else: W=W[:,:1024]
        aL=train_torch(LFFTDetector(n_filters=48),W[:cut],y[:cut],W[cut:],y[cut:],device,args.epochs,is1d=True)
        print(f"{gap:>5.1f}{aF:>11.2f}{aS:>10.2f}{aL:>10.2f}")
        rows.append((gap,aF,aS,aL))
    print("\nreading: freq-only ~0.5 always (overlap). If SpecCNN/LFFT rise with gap,")
    print("the CNNs genuinely learn the harmonic difference.")
    return 0

if __name__=="__main__":
    sys.exit(main())
