"""Numpy-only golden reference for museproc. Reconstructs the EEG grid and
computes the four metrics at chosen window-ends, to compare against the C++
tool run with --no-notch (so we isolate the algorithm from the causal notch).

--ppg/--imu mirror museproc's flags: a held-latest-value readout of the raw
PPG infrared channel and/or accel/gyro, synced to whatever grid position is
queried (see load_session / hold_at)."""
import json, sys, math, bisect
import numpy as np

FS = 256
CH = 4

def load_session(path):
    """Single linear pass over the JSONL file, in line order — the same order
    museproc's main loop reads it in. Builds the eeg grid via sequential
    commit-on-packet-completion (an unwrapped index's 4 electrodes are
    buffered until all arrive, gaps NaN-filled, overlaps dropped — mirrors
    EegStream::write_packet in cpp/jsonl_reader.cpp) and records ppg/accel/
    gyro samples as (grid_len_at_arrival, value...) events: the running eeg
    grid length at the moment each line was read. That running length is
    exactly what a --ppg/--imu query later holds at any queried grid
    position, mirroring PpgTracker/ImuTracker's latest-value-wins behavior."""
    prev = [None]*CH; wraps=[0]*CH
    have_first = False; first_idx = 0; grid_len = 0
    partial = {}       # unwrapped idx -> {electrode: samples}
    chunks = []         # (start, ndarray[n,CH]) eeg blocks in grid order
    commit_points = []  # grid_len right after each eeg packet completion —
                         # exactly the points museproc's main loop checks
                         # `grid_len() >= next_emit` against (see emit_schedule)
    ppg_events = []      # (grid_len, sample)
    accel_events = []    # (grid_len, x, y, z)
    gyro_events = []     # (grid_len, x, y, z)

    with open(path) as f:
        for line in f:
            if '"type":"eeg"' in line:
                r = json.loads(line)
                e = r["electrode"]; raw = r["index"]
                if prev[e] is not None and prev[e]-raw > 32768: wraps[e]+=1
                prev[e]=raw
                u = raw + wraps[e]*65536
                part = partial.setdefault(u,{})
                part[e] = np.asarray(r["samples"],dtype=np.float64)
                if len(part) == CH:
                    if not have_first:
                        have_first = True; first_idx = u; grid_len = 0
                    s = (u-first_idx)*12
                    if s >= grid_len:          # else: overlap, an earlier packet already won
                        if s > grid_len:
                            chunks.append((grid_len, np.full((s-grid_len,CH), np.nan)))
                            grid_len = s
                        block = np.stack([part[c] for c in range(CH)], axis=1)
                        chunks.append((grid_len, block))
                        grid_len += block.shape[0]
                        commit_points.append(grid_len)
                    del partial[u]
                    for k in [k for k in partial if k < u-2]: del partial[k]
            elif '"type":"ppg"' in line:
                r = json.loads(line)
                s = r.get("samples")
                if r.get("ppgChannel") == 1 and s:
                    ppg_events.append((grid_len, float(s[-1])))
            elif '"type":"accel"' in line:
                r = json.loads(line); s = r.get("samples")
                if s:
                    last = s[-1]; accel_events.append((grid_len, last["x"], last["y"], last["z"]))
            elif '"type":"gyro"' in line:
                r = json.loads(line); s = r.get("samples")
                if s:
                    last = s[-1]; gyro_events.append((grid_len, last["x"], last["y"], last["z"]))

    grid = np.full((grid_len,CH), np.nan)
    for start, block in chunks:
        grid[start:start+block.shape[0]] = block
    return grid, ppg_events, accel_events, gyro_events, commit_points

def load_grid(path):
    return load_session(path)[0]

def emit_schedule(commit_points, hop=128, win=256):
    """Reproduces museproc's exact output-row grid positions from the true
    eeg-packet-completion sequence (see main.cpp's emit loop: `next_emit`
    starts at EEG_WIN and jumps to the next multiple of hop past whatever
    grid_len last triggered an emit). Works whether or not the session has
    gaps, because it walks the real commit sequence rather than assuming a
    fixed +12-per-packet cadence."""
    Ls = []
    next_emit = win
    for g in commit_points:
        if g >= next_emit:
            Ls.append(g)
            next_emit = ((g // hop) + 1) * hop
    return Ls

def hold_at(events, L, ncols=1):
    """Value of the last event with grid_len < L — nan (or an nan tuple)
    if none has arrived yet. Strictly less-than, not <=: grid_len only
    advances on eeg lines, so an event stamped grid_len==L was necessarily
    read on a *later* line than the eeg packet that pushed the grid to L —
    at the real emit moment (which fires immediately after that eeg line)
    it has not been fed to the tracker yet. Same latest-value-wins
    semantics as PpgTracker/ImuTracker in cpp/jsonl_reader.cpp."""
    nan = (math.nan,)*ncols if ncols > 1 else math.nan
    if not events: return nan
    idx = bisect.bisect_left([e[0] for e in events], L) - 1
    if idx < 0: return nan
    vals = events[idx][1:]
    return vals if ncols > 1 else vals[0]

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

def metrics_at(grid, L, want_entropy=True):
    """want_entropy=False skips the O(n^2) sample-entropy computation, which
    dominates runtime (~5s/call on a 2048-sample window vs. a few ms for
    everything else) — set it False for fast, exhaustive checks of the other
    three metrics, and only pay for entropy on a sampled subset of rows."""
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
    if want_entropy and L>=2048 and tot>0:
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
    args = sys.argv[1:]
    want_ppg = '--ppg' in args
    want_imu = '--imu' in args
    args = [a for a in args if a not in ('--ppg','--imu')]
    path = args[0]
    grid, ppg_events, accel_events, gyro_events, _ = load_session(path)
    print(f"# grid samples={len(grid)}  dur={len(grid)/FS:.1f}s")
    if len(args) == 1:
        print("# no timestamps given — pass one or more seconds to query, e.g.:")
        print(f"#   python3 {sys.argv[0]} {path} --ppg --imu 5 30 90")
        print("# or use harness/compare.py to check every row against museproc automatically")
    fmt1 = lambda v: 'nan' if math.isnan(v) else f"{v:.5f}"
    for t in [float(x) for x in args[1:]]:
        L=round(t*FS)
        ent,ta,sym,q=metrics_at(grid,L)
        line = f"t={L/FS:.5f} entropy={ent:.5f} theta_alpha={ta:.5f} alpha_symmetry={sym:.5f} quality={q:.5f}"
        if want_ppg:
            line += f" ppg={fmt1(hold_at(ppg_events, L))}"
        if want_imu:
            ax,ay,az = hold_at(accel_events, L, 3)
            gx,gy,gz = hold_at(gyro_events, L, 3)
            line += (f" accel_x={fmt1(ax)} accel_y={fmt1(ay)} accel_z={fmt1(az)}"
                     f" gyro_x={fmt1(gx)} gyro_y={fmt1(gy)} gyro_z={fmt1(gz)}")
        print(line)
