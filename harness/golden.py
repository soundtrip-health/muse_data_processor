"""Numpy-only golden reference for museproc. Reconstructs the EEG grid and
computes the four metrics at chosen window-ends, to compare against the C++
tool run with --no-notch (so we isolate the algorithm from the causal notch)."""
import json, sys, math
import numpy as np

FS = 256
CH = 4

def load_grid(path):
    prev = [None]*CH; wraps=[0]*CH
    by_idx = {}
    with open(path) as f:
        for line in f:
            if '"type":"eeg"' not in line: continue
            r = json.loads(line)
            e = r["electrode"]; raw = r["index"]
            if prev[e] is not None and prev[e]-raw > 32768: wraps[e]+=1
            prev[e]=raw
            u = raw + wraps[e]*65536
            by_idx.setdefault(u,{})[e]=np.asarray(r["samples"],dtype=np.float64)
    idxs = sorted(by_idx)
    i0 = idxs[0]
    n_total = (idxs[-1]-i0)*12 + 12 + 12
    grid = np.full((n_total,CH), np.nan)
    for u in idxs:
        chans = by_idx[u]
        if len(chans) < 4: continue
        s = (u-i0)*12
        for e in range(CH):
            grid[s:s+12,e] = chans[e]
    return grid

def make_morlet(f, tau, fs=FS):
    sigma = tau/(2*math.pi*f)
    amp = 1.0/math.sqrt(sigma*math.sqrt(math.pi))
    half = int(math.ceil(3.0*sigma*fs))
    t = (np.arange(2*half+1)-half)/fs
    g = amp*np.exp(-0.5*(t/sigma)**2)
    return g*np.cos(2*math.pi*f*t), g*np.sin(2*math.pi*f*t), half

KT = make_morlet(6,4); KA = make_morlet(10,6)

def wavelet_power(sig, ker):
    if np.isnan(sig).any(): return np.nan
    re,im,half = ker; n=len(sig)
    start=half; end=n-1-half
    if start>end: return 0.0
    sw = np.lib.stride_tricks.sliding_window_view(sig, len(re))
    rr = sw@re; ii = sw@im
    return float(np.mean(rr*rr+ii*ii))

def rms(sig):
    v = sig[~np.isnan(sig)]
    if len(v)<10: return np.nan
    v = v-v.mean(); return float(np.sqrt(np.mean(v*v)))

def qlabel(r):
    if not np.isfinite(r): return 'poor'
    return 'good' if r<50 else 'marginal' if r<100 else 'poor'

def weights(labels):
    qw={'good':1.0,'marginal':0.5,'poor':0.0}
    poor=[c for c in range(4) if labels[c]=='poor']
    drop=set(poor[:2])
    w=np.array([0.0 if c in drop else qw[labels[c]] for c in range(4)])
    tot=w.sum()
    if tot>0: return w/tot, tot
    keep=[c for c in range(4) if c not in drop]
    fb=np.zeros(4)
    if keep: fb[keep]=1.0/len(keep)
    return fb, 0.0

def coarse(x,tau):
    if tau<=1: return x
    n=(len(x)//tau)*tau
    return x[:n].reshape(-1,tau).mean(axis=1)

def sampen(sig,m,r):
    n=len(sig)
    if n<m+2: return 0.0
    L=n-m; A=0;B=0
    sig=np.asarray(sig)
    for i in range(L):
        for j in range(i+1,L):
            if np.max(np.abs(sig[i:i+m]-sig[j:j+m]))<=r:
                B+=1
                if abs(sig[i+m]-sig[j+m])<=r: A+=1
    if A==0 or B==0: return 0.0
    return -math.log(A/B)

def interp_short(x,maxgap):
    x=x.copy(); n=len(x); i=0
    while i<n:
        if not np.isnan(x[i]): i+=1; continue
        s=i
        while i<n and np.isnan(x[i]): i+=1
        e=i; L=e-s
        if L<=maxgap and s>0 and e<n:
            x[s:e]=np.interp(np.arange(s,e),[s-1,e],[x[s-1],x[e]])
    return x

def metrics_at(grid, L):
    w256 = grid[L-256:L]
    labels=[qlabel(rms(w256[:,c])) for c in range(4)]
    rmss=[rms(w256[:,c]) for c in range(4)]
    w,tot = weights(labels)
    th=[wavelet_power(w256[:,c],KT) for c in range(4)]
    al=[wavelet_power(w256[:,c],KA) for c in range(4)]
    # theta/alpha
    num=den=0.0; ok=tot>0
    for c in range(4):
        if w[c]<=0: continue
        if np.isnan(th[c]) or np.isnan(al[c]): ok=False;break
        num+=w[c]*th[c]; den+=w[c]*al[c]
    ta = num/den if ok and den>0 else np.nan
    sym = math.log(al[2]+1e-12)-math.log(al[1]+1e-12) if not (np.isnan(al[1]) or np.isnan(al[2])) else np.nan
    # mse
    ent=np.nan
    if L>=2048 and tot>0:
        block=grid[L-2048:L]
        sig=(block*w).sum(axis=1)
        # NaN where any weighted channel is NaN
        badmask=np.isnan(block[:, w>0]).any(axis=1) if (w>0).any() else np.ones(len(block),bool)
        sig=np.where(badmask,np.nan,sig)
        sig=interp_short(sig,13)
        if not np.isnan(sig).any():
            sigma=float(np.std(sig))
            if sigma<1e-6: ent=0.0
            else:
                r=0.15*sigma
                ent=float(np.mean([sampen(coarse(sig,s),2,r) for s in (1,3,5,7,9)]))
    ng=sum(1 for c in range(4) if labels[c]=='good')
    best=np.nanmin(rmss) if any(np.isfinite(x) for x in rmss) else np.nan
    clean=0.0 if np.isnan(best) else 1-min(best,100)/100
    q=0.5*(ng/4)+0.5*clean
    return ent,ta,sym,q

if __name__=='__main__':
    path=sys.argv[1]
    grid=load_grid(path)
    print(f"# grid samples={len(grid)}  dur={len(grid)/FS:.1f}s")
    for t in [float(x) for x in sys.argv[2:]]:
        L=round(t*FS)
        ent,ta,sym,q=metrics_at(grid,L)
        print(f"t={L/FS:.5f} entropy={ent:.5f} theta_alpha={ta:.5f} alpha_symmetry={sym:.5f} quality={q:.5f}")
