# -*- coding: utf-8 -*-
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from dataclasses import dataclass

# ---------------- IO / paths ----------------
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- global knobs ----------------
GRID = 600
VAL_PER_ANCHOR = 24
CAND_MULTS = (1.0, 1.5, 2.0)
MED_WIN_H2 = 13
MED_WIN_E2 = 11

# AFKNN training (轻量，seed=7)
SEED_BASE = 7
N_PER_SEED = 120
KCANDS = (15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 130, 150)
VAR_W = 6.0
ALPHA_LIN = 0.08
UPPER_FRAC = 0.35
ANCHORS_PER_REP = 50
TESTS_PER_ANCHOR = 24

# colors
COL_FIXED = "#FF7F0E"   # orange
COL_ADAPT = "#2CA02C"   # green
COL_AFKNN = "#D62728"   # red

# ---------------- CSV loader ----------------
def _norm_col(c: str) -> str:
    c = str(c).strip().lower().replace('（','(').replace('）',')')
    keep = set('abcdefghijklmnopqrstuvwxyz0123456789()')
    return ''.join(ch for ch in c if ch in keep)

def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    nm = {c: _norm_col(c) for c in df.columns}
    df = df.rename(columns={
        next(k for k,v in nm.items() if v in ('i','rep')): 'rep',
        next(k for k,v in nm.items() if v == 't'): 't',
        next(k for k,v in nm.items() if v in ('v(t)','vt','v')): 'Y',
    })
    df['rep'] = pd.to_numeric(df['rep'], errors='coerce').astype(int)
    df['t']   = pd.to_numeric(df['t'],   errors='coerce')
    df['Y']   = pd.to_numeric(df['Y'],   errors='coerce')
    df = df.dropna(subset=['rep','t','Y']).sort_values(['rep','t']).reset_index(drop=True)
    return df

# ---------------- KFE（与前一致） ----------------
def _as_fn(x):
    if callable(x): return lambda t: max(0.0, float(x(t)))
    v = float(x);   return lambda t: v
def lamE(t):  return 2.0 + np.sin(0.5*t)       # E2(t)
def lamH1(t): return 2.0 + 0.6*np.sin(0.5*t)   # H2 phase 1
def lamH2(t): return 0.8 + 0.4*np.cos(0.3*t)   # H2 phase 2
P_MIX = 0.6
MU, C = 1.0, 8

def build_Q_H2M1c(lambda1, lambda2, p_mix, mu: float, c: int):
    l1f=_as_fn(lambda1); l2f=_as_fn(lambda2); pf=_as_fn(p_mix)
    def Q_of_t(t: float) -> np.ndarray:
        l1=l1f(t); l2=l2f(t); p=max(0.0,min(1.0,pf(t)))
        nst=2*(c+1); Q=np.zeros((nst,nst), float)
        for ph in (0,1):
            lam = l1 if ph==0 else l2
            for j in range(c+1):
                idx=ph*(c+1)+j; out=0.0
                if j>0: Q[idx, ph*(c+1)+(j-1)] += MU; out += MU
                tgt = j+1 if j<c else c
                Q[idx, 0*(c+1)+tgt] += p*lam
                Q[idx, 1*(c+1)+tgt] += (1-p)*lam
                out += lam
                Q[idx, idx] -= out
        return Q
    return Q_of_t

def build_Q_E2M1c(lam, mu: float, c: int):
    lf=_as_fn(lam)
    def Q_of_t(t: float) -> np.ndarray:
        l=lf(t); nst=2*(c+1); Q=np.zeros((nst,nst), float)
        for ph in (0,1):
            for j in range(c+1):
                idx=ph*(c+1)+j; out=0.0
                if j>0: Q[idx, ph*(c+1)+(j-1)] += MU; out += MU
                if ph==0:
                    Q[idx, 1*(c+1)+j] += l; out += l
                else:
                    tgt = j+1 if j<c else c
                    Q[idx, 0*(c+1)+tgt] += l; out += l
                Q[idx, idx] -= out
        return Q
    return Q_of_t

def rk4_integrate(Q_of_t, p0: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    nT=len(t_grid); nst=p0.size
    P=np.zeros((nT,nst), float); P[0]=p0
    f=lambda t,p: Q_of_t(t).T @ p
    for k in range(nT-1):
        t=t_grid[k]; h=t_grid[k+1]-t; p=P[k]
        k1=f(t,p); k2=f(t+0.5*h,p+0.5*h*k1); k3=f(t+0.5*h,p+0.5*h*k2); k4=f(t+h,p+h*k3)
        pn=p+(h/6.0)*(k1+2*k2+2*k3+k4)
        pn=np.maximum(pn,0.0); s=pn.sum()
        if s<=0: pn=p.copy(); s=pn.sum()
        if s!=1.0: pn/=s
        P[k+1]=pn
    return P

def v_wait_arrival_H2(P: np.ndarray, c: int, mu: float, t_grid: np.ndarray, lambda1, lambda2):
    j=np.arange(0,c+1, dtype=float); w=(j/mu)[:c]
    v=np.zeros(P.shape[0], float)
    l1f=_as_fn(lambda1); l2f=_as_fn(lambda2)
    for k in range(P.shape[0]):
        p1=P[k,:c+1]; p2=P[k,c+1:2*(c+1)]
        l1=l1f(t_grid[k]); l2=l2f(t_grid[k])
        num=l1*np.dot(w,p1[:c]) + l2*np.dot(w,p2[:c])
        den=l1*np.sum(p1[:c])  + l2*np.sum(p2[:c])
        v[k]=(num/den) if den>1e-14 else np.nan
    return v

def v_wait_arrival_E2(P: np.ndarray, c: int, mu: float):
    j=np.arange(0,c+1, dtype=float); w=(j/mu)[:c]
    v=np.zeros(P.shape[0], float)
    for k in range(P.shape[0]):
        p2=P[k,c+1:2*(c+1)]
        den=np.sum(p2[:c]); num=np.dot(w,p2[:c])
        v[k]=(num/den) if den>1e-14 else np.nan
    return v

def solve_h2_kfe(t_end: float, steps=5000):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_H2M1c(lamH1, lamH2, P_MIX, MU, C)
    nst=2*(C+1); p0=np.zeros(nst); p0[0]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_H2(P, C, MU, t, lamH1, lamH2)
    return t, v

def solve_e2_kfe(t_end: float, steps=5000):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_E2M1c(lamE, MU, C)
    nst=2*(C+1); p0=np.zeros(nst); p0[0]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_E2(P, C, MU)
    return t, v

# ---------------- kNN utils ----------------
def _edge_flags(t0, t_min, t_max, frac=0.03):
    nearL = (t0 - t_min) <= frac*(t_max - t_min)
    nearR = (t_max - t0) <= frac*(t_max - t_min)
    return bool(nearR), bool(nearL)

def _gather(ts, reps, t0, k, exclude_rep=None, left_only=False, right_only=False):
    mask=np.ones_like(ts, dtype=bool)
    if exclude_rep is not None: mask &= (reps != exclude_rep)
    if left_only:  mask &= (ts <= t0)
    if right_only: mask &= (ts >= t0)
    idx=np.where(mask)[0]
    if idx.size==0: return []
    d=np.abs(ts[idx]-t0)
    take=min(k, idx.size)
    part=np.argpartition(d, take-1)[:take]
    near=idx[part]
    near=near[np.argsort(np.abs(ts[near]-t0))]
    return near.tolist()

def knn_equal_estimate(ts, ys, t0, k, pool_idx=None):
    ts=np.asarray(ts); ys=np.asarray(ys)
    if pool_idx is None:
        d=np.abs(ts - t0); take=min(k, len(ts))
        idx=np.argpartition(d, take-1)[:take]
    else:
        pool=np.asarray(pool_idx, dtype=int)
        d=np.abs(ts[pool] - t0); take=min(k, pool.size)
        idx=pool[np.argpartition(d, take-1)[:take]]
    y=ys[idx].astype(float)
    mu=float(np.mean(y))
    var=0.0 if y.size<=1 else float(np.var(y, ddof=1))/y.size
    mu=float(np.clip(mu, float(np.min(y)), float(np.max(y))))
    return mu, var

def lorocv_best_k_equal(df: pd.DataFrame, k_max: int = 1500):
    reps=df['rep'].unique(); N=len(df)
    Mj=df.groupby('rep').size().to_dict()
    kU=int(min(k_max, min(N - Mj[r] for r in reps)))
    if kU<3: return 3
    sse=np.zeros(kU+1, float); total=0
    for r in reps:
        test=df[df['rep']==r]; train=df[df['rep']!=r]
        t_te=test['t'].to_numpy(); y_te=test['Y'].to_numpy()
        t_tr=train['t'].to_numpy(); y_tr=train['t'].index.values # corrected later
        t_tr=train['t'].to_numpy(); y_tr=train['Y'].to_numpy()
        D=np.abs(t_te[:,None] - t_tr[None,:])
        idx_part=np.argpartition(D, kth=kU-1, axis=1)[:, :kU]
        D_part=np.take_along_axis(D, idx_part, axis=1)
        order=np.argsort(D_part, axis=1)
        nn=np.take_along_axis(idx_part, order, axis=1)
        Ynn=y_tr[nn]
        cums=np.cumsum(Ynn, axis=1)
        ks=np.arange(1, kU+1)
        preds=cums/ks
        errs=(preds - y_te[:,None])**2
        sse[1:] += np.sum(errs, axis=0)
        total += len(y_te)
    emse=sse[1:] / total
    return int(np.argmin(emse) + 1)

def running_median_int(arr, win=11):
    if win<=1: return arr
    w=win if win%2==1 else win+1; r=w//2
    pad=np.r_[np.repeat(arr[0],r), arr, np.repeat(arr[-1],r)]
    out=np.empty_like(arr)
    for i in range(len(arr)):
        out[i]=int(np.median(pad[i:i+w]))
    return out

# ---------------- Fixed-k on grid ----------------
def fixed_knn_on_grid(df: pd.DataFrame, t_grid: np.ndarray):
    reps=df['rep'].to_numpy(); ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    k_star=max(3, lorocv_best_k_equal(df, k_max=2000))
    v_hat=np.zeros_like(t_grid); var_hat=np.zeros_like(t_grid)
    t_min, t_max=float(ts.min()), float(ts.max())
    for i,t0 in enumerate(t_grid):
        left_only, right_only=_edge_flags(t0, t_min, t_max)
        pool=_gather(ts, reps, t0, 3000, left_only=left_only, right_only=right_only)
        mu, var = knn_equal_estimate(ts, ys, t0, k_star, pool_idx=pool)
        v_hat[i], var_hat[i] = mu, var
    return v_hat, var_hat, k_star

# ---------------- Adaptive on grid ----------------
def adaptive_knn_on_grid(df: pd.DataFrame, t_grid: np.ndarray):
    reps=df['rep'].to_numpy(); ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    t_min, t_max=float(ts.min()), float(ts.max())
    k_base=max(3, lorocv_best_k_equal(df, k_max=2000))
    cand_base = sorted(set(max(3, int(round(k_base*m))) for m in CAND_MULTS))
    v_hat=np.zeros_like(t_grid); var_hat=np.zeros_like(t_grid)
    for i,t0 in enumerate(t_grid):
        left_only, right_only=_edge_flags(t0, t_min, t_max)
        pool=_gather(ts, reps, t0, 2000, left_only=left_only, right_only=right_only)
        if len(pool)<40: pool=_gather(ts, reps, t0, min(3000,len(ts)-1))
        # local CV choose k
        best_k, best_err = cand_base[0], np.inf
        d_pool=np.abs(ts[pool]-t0); order=np.argsort(d_pool)
        near=[pool[j] for j in order[:min(500,len(pool))]]
        chosen, seen=[], set()
        for idx in near:
            rr=int(reps[idx])
            if rr in seen: continue
            seen.add(rr); chosen.append(idx)
            if len(chosen)>=VAL_PER_ANCHOR: break
        if len(chosen)<max(8, VAL_PER_ANCHOR//2): chosen=near[:VAL_PER_ANCHOR]
        for k in cand_base:
            errs=[]
            for idx in chosen:
                rr=reps[idx]
                pool_tr=[p for p in pool if reps[p]!=rr]
                if len(pool_tr)<3: continue
                mu,_ = knn_equal_estimate(ts, ys, ts[idx], k, pool_idx=pool_tr)
                errs.append((mu - ys[idx])**2)
            if errs:
                loc=float(np.mean(errs))
                if loc<best_err: best_err, best_k = loc, k
        mu, var = knn_equal_estimate(ts, ys, t0, best_k, pool_idx=pool)
        v_hat[i], var_hat[i] = mu, var
    return v_hat, var_hat

# ---------------- AFKNN (forest) ----------------
import numpy.random as npr
def simulate_union(model="H2", seed_base=SEED_BASE, n_per_seed=N_PER_SEED):
    import arrival_wait_pairs as awp
    rows=[]
    for r in range(1, n_per_seed+1):
        seed = seed_base + r
        if model.upper()=="E2":
            pairs = awp.generate_pairs_E2(16.0, MU, C, lamE, 3.0, seed=seed)
        else:
            pairs = awp.generate_pairs_H2(16.0, MU, C, P_MIX, lamH1, 2.6, lamH2, 1.2, seed=seed)
        rep_id = seed_base*10_000 + r
        for (t,v) in pairs:
            rows.append((rep_id, t, v))
    return pd.DataFrame(rows, columns=["rep","t","Y"]).sort_values(["rep","t"]).reset_index(drop=True)

def _local_feats_for_t(ts, ys, t0, K0=25):
    d=np.abs(ts - t0); K0=min(int(K0), len(ts))
    nn=np.argpartition(d, K0-1)[:K0]
    dK=np.partition(d, K0-1)[K0-1] if K0>0 else 0.0
    var_local=np.var(ys[nn], ddof=1) if K0>1 else 0.0
    left=np.sum(ts[nn] < t0); right=K0-left
    asym=(right-left)/max(1,K0)
    return dK, var_local, asym

def make_features(df, t_points):
    ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    t_min, t_max=float(ts.min()), float(ts.max())
    feats=[]
    for t0 in np.asarray(t_points, float):
        t_norm=(t0 - t_min)/max(1e-9, (t_max - t_min))
        dK, vloc, asym=_local_feats_for_t(ts, ys, t0, K0=25)
        edgeL = 1.0 if (t0 - t_min) <= 0.03*(t_max - t_min) else 0.0
        edgeR = 1.0 if (t_max - t0) <= 0.03*(t_max - t_min) else 0.0
        feats.append([t_norm, dK, vloc, asym, edgeL, edgeR])
    return np.asarray(feats, float)

@dataclass
class ForestCfg:
    n_estimators: int = 300
    max_depth: int = 9
    min_samples_leaf: int = 40
    kmin_global: int = 15

class KForest:
    def __init__(self, cfg: ForestCfg):
        self.cfg=cfg; self.rf=None
    def fit(self, df: pd.DataFrame, tag: str, kfe_t, kfe_v):
        from sklearn.ensemble import RandomForestRegressor
        lab_t, lab_k = label_k_star_with_kfe(
            df, KCANDS, TESTS_PER_ANCHOR, VAR_W, ALPHA_LIN, kfe_t, kfe_v,
            anchors_per_rep=ANCHORS_PER_REP, rng=npr.default_rng(0)
        )
        if lab_t.size==0: raise RuntimeError(f"[{tag}] no labels.")
        X = make_features(df, lab_t); y = np.log(np.maximum(lab_k,3))
        self.rf = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=0, n_jobs=-1
        ).fit(X,y)
    def predict_k_on_points(self, df_csv: pd.DataFrame, t_points, is_h2: bool):
        ts=df_csv['t'].to_numpy()
        Xg=make_features(df_csv, t_points)
        k_rf=np.clip(np.round(np.exp(self.rf.predict(Xg))).astype(int), 3, None)
        neff=min(250, len(ts))
        LOWER_FRAC = 0.12 if is_h2 else 0.08
        for i in range(len(k_rf)):
            k_rf[i]=max(int(k_rf[i]), self.cfg.kmin_global, int(LOWER_FRAC*neff))
            k_rf[i]=min(int(k_rf[i]), int(0.5*neff))
        win = MED_WIN_H2 if is_h2 else MED_WIN_E2
        k_rf = running_median_int(k_rf, win=win)
        return k_rf

def label_k_star_with_kfe(df: pd.DataFrame, k_candidates, tests_per_anchor,
                          var_w, alpha_lin, kfe_t, kfe_v,
                          anchors_per_rep=ANCHORS_PER_REP, rng=npr.default_rng(0)):
    reps=df['rep'].to_numpy(); ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    uniq=np.unique(reps); t_min, t_max=float(ts.min()), float(ts.max())
    labels_t=[]; labels_k=[]
    idx_by_rep={r: np.where(reps==r)[0] for r in uniq}
    for r in uniq:
        idx_r=idx_by_rep[r]
        anchors = idx_r if len(idx_r)<=anchors_per_rep else rng.choice(idx_r, size=anchors_per_rep, replace=False)
        for j in anchors:
            t0=ts[j]
            left_only, right_only=_edge_flags(t0, t_min, t_max)
            pool=_gather(ts, reps, t0, 1600, exclude_rep=r, left_only=left_only, right_only=right_only)
            if len(pool)<60: pool=_gather(ts, reps, t0, min(2600,len(ts)-1), exclude_rep=r)
            if len(pool)<30: continue
            d_pool=np.abs(ts[pool]-t0); order=np.argsort(d_pool)
            near=[pool[i] for i in order[:min(400,len(pool))]]
            chosen=[]; seen=set()
            for idx in near:
                rr=int(reps[idx])
                if rr in seen: continue
                seen.add(rr); chosen.append(idx)
                if len(chosen)>=tests_per_anchor: break
            if len(chosen)<max(8, tests_per_anchor//2): chosen=near[:tests_per_anchor]
            k_anchor_max=max(5, min(max(KCANDS), int(np.floor(UPPER_FRAC*len(pool)))))
            bias2_bank={}; var_bank={}
            for k in k_candidates:
                k_eff=min(k, k_anchor_max)
                if k_eff<3: continue
                b2=[]; vv=[]
                for idx in chosen:
                    rr=reps[idx]
                    pool_tr=[p for p in pool if reps[p]!=rr]
                    if len(pool_tr)<3: continue
                    mu, var = knn_equal_estimate(ts, ys, ts[idx], k_eff, pool_idx=pool_tr)
                    b2.append((mu - np.interp(ts[idx], kfe_t, kfe_v))**2)
                    vv.append(var)
                if b2:
                    bias2_bank[k_eff]=np.asarray(b2,float)
                    var_bank[k_eff]=np.asarray(vv,float)
            if not bias2_bank: continue
            med_b=np.median(np.concatenate(list(bias2_bank.values())))+1e-12
            med_v=np.median(np.concatenate(list(var_bank.values())))+1e-12
            best_k, best_obj=None, np.inf
            for k_eff,b2 in bias2_bank.items():
                v_arr=var_bank[k_eff]
                obj=float(np.mean(b2/med_b + var_w*(v_arr/med_v))) + alpha_lin*(k_eff/k_anchor_max)
                if obj<best_obj: best_obj, best_k=obj, k_eff
            if best_k is not None:
                labels_t.append(t0); labels_k.append(best_k)
    return np.asarray(labels_t,float), np.asarray(labels_k,int)

# ---------------- metrics ----------------
def compute_metrics_curves(v_hat, var_hat, t_grid, t_kfe, v_kfe):
    v_true = np.interp(t_grid, t_kfe, v_kfe)
    bias = v_hat - v_true
    abs_bias = np.abs(bias)
    emse = bias**2 + var_hat
    return abs_bias, var_hat, emse

def summarize_and_print(tag, abs_bias, var, emse):
    mab  = float(np.nanmean(abs_bias))
    mvar = float(np.nanmean(var))
    memse= float(np.nanmean(emse))
    print(f"{tag:>10s}  mean|bias|={mab:.4f}   mean Var={mvar:.4f}   EMSE={memse:.4f}")
    return mab, mvar, memse

# ---------------- drawing 2×3 ----------------
def draw_2x3(tH, curves_H2, tE, curves_E2, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex='col')
    # column titles
    col_titles = [r"$|bias|(t)$", r"$Var(t)$", r"$EMSE(t)=bias^2+Var$"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(f"H2 — {title}")
        axes[1, j].set_title(f"E2 — {title}")

    # helper to plot a row
    def plot_row(ax_row, t_grid, curves):
        # |bias|
        for name,(ab,var,emse,color) in curves.items():
            ax_row[0].plot(t_grid, ab, lw=2.0, label=name, color=color)
        ax_row[0].grid(True, ls='--', alpha=0.35)
        ax_row[0].set_ylabel("|bias|")

        # Var
        for name,(ab,var,emse,color) in curves.items():
            ax_row[1].plot(t_grid, var, lw=2.0, label=name, color=color)
        ax_row[1].grid(True, ls='--', alpha=0.35)
        ax_row[1].set_ylabel("Var")

        # EMSE
        for name,(ab,var,emse,color) in curves.items():
            ax_row[2].plot(t_grid, emse, lw=2.2, label=name, color=color)
        ax_row[2].grid(True, ls='--', alpha=0.35)
        ax_row[2].set_ylabel("EMSE")

        for ax in ax_row:
            ax.set_xlabel("t")

    plot_row(axes[0, :], tH, curves_H2)
    plot_row(axes[1, :], tE, curves_E2)

    # single global legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"[Saved] {out_png}")

# ---------------- main ----------------
def main():
    # load CSVs
    dfH = load_pairs("H2_pairs.csv")
    dfE = load_pairs("E2_pairs.csv")

    # grids & KFE
    tH = np.linspace(float(dfH['t'].min()), float(dfH['t'].max()), GRID, endpoint=False)
    tE = np.linspace(float(dfE['t'].min()), float(dfE['t'].max()), GRID, endpoint=False)
    tH_kfe, vH_kfe = solve_h2_kfe(t_end=float(dfH['t'].max()), steps=5000)
    tE_kfe, vE_kfe = solve_e2_kfe(t_end=float(dfE['t'].max()), steps=5000)

    # AFKNN 训练（seed=7）
    dfH_sim = simulate_union("H2", SEED_BASE, N_PER_SEED)
    dfE_sim = simulate_union("E2", SEED_BASE, N_PER_SEED)
    kfH = KForest(ForestCfg()); kfH.fit(dfH_sim, "H2", tH_kfe, vH_kfe)
    kfE = KForest(ForestCfg()); kfE.fit(dfE_sim, "E2", tE_kfe, vE_kfe)

    # H2：三方法
    vH_fix, sH_fix, kH_star = fixed_knn_on_grid(dfH, tH)
    vH_adp, sH_adp = adaptive_knn_on_grid(dfH, tH)
    tsH, ysH = dfH['t'].to_numpy(), dfH['Y'].to_numpy()
    kH_seq = kfH.predict_k_on_points(dfH, tH, is_h2=True)
    vH_af = np.zeros_like(tH); sH_af = np.zeros_like(tH)
    for i,t0 in enumerate(tH):
        mu, var = knn_equal_estimate(tsH, ysH, t0, int(kH_seq[i]))
        vH_af[i], sH_af[i] = mu, var

    # E2：三方法
    vE_fix, sE_fix, kE_star = fixed_knn_on_grid(dfE, tE)
    vE_adp, sE_adp = adaptive_knn_on_grid(dfE, tE)
    tsE, ysE = dfE['t'].to_numpy(), dfE['Y'].to_numpy()
    kE_seq = kfE.predict_k_on_points(dfE, tE, is_h2=False)
    vE_af = np.zeros_like(tE); sE_af = np.zeros_like(tE)
    for i,t0 in enumerate(tE):
        mu, var = knn_equal_estimate(tsE, ysE, t0, int(kE_seq[i]))
        vE_af[i], sE_af[i] = mu, var

    # 误差分解曲线
    H_fix = compute_metrics_curves(vH_fix, sH_fix, tH, tH_kfe, vH_kfe)
    H_adp = compute_metrics_curves(vH_adp, sH_adp, tH, tH_kfe, vH_kfe)
    H_af  = compute_metrics_curves(vH_af , sH_af , tH, tH_kfe, vH_kfe)
    curves_H2 = {
        f"kNN (fixed k={kH_star})" : (*H_fix, COL_FIXED),
        "Adaptive kNN"             : (*H_adp, COL_ADAPT),
        "AFKNN (forest k(t))"      : (*H_af , COL_AFKNN),
    }

    E_fix = compute_metrics_curves(vE_fix, sE_fix, tE, tE_kfe, vE_kfe)
    E_adp = compute_metrics_curves(vE_adp, sE_adp, tE, tE_kfe, vE_kfe)
    E_af  = compute_metrics_curves(vE_af , sE_af , tE, tE_kfe, vE_kfe)
    curves_E2 = {
        f"kNN (fixed k={kE_star})" : (*E_fix, COL_FIXED),
        "Adaptive kNN"             : (*E_adp, COL_ADAPT),
        "AFKNN (forest k(t))"      : (*E_af , COL_AFKNN),
    }

    # 绘 2×3
    draw_2x3(tH, curves_H2, tE, curves_E2, os.path.join(OUT_DIR, "metrics_2x3.png"))

    # 终端汇总
    print("\n[H2] Summary metrics (time-average)")
    summarize_and_print("fixed-k", *H_fix)
    summarize_and_print("adaptive", *H_adp)
    summarize_and_print("AFKNN", *H_af)

    print("\n[E2] Summary metrics (time-average)")
    summarize_and_print("fixed-k", *E_fix)
    summarize_and_print("adaptive", *E_adp)
    summarize_and_print("AFKNN", *E_af)

if __name__ == "__main__":
    main()
