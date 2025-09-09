# -*- coding: utf-8 -*-
"""
2x2 comparison figure:
左上:  H2 — KFE + kNN + Adaptive
右上:  H2 — KFE + kNN + AFKNN
左下:  E2 — KFE + kNN + Adaptive
右下:  E2 — KFE + kNN + AFKNN
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ consistent, high-contrast colors ------------------
COL = {
    "kfe":    "#1f77b4",  # deep blue
    "knn":    "#ff7f0e",  # orange
    "adapt":  "#2ca02c",  # green
    "afknn":  "#d62728",  # crimson
}

# ============================ I/O helpers =============================
def _norm_col(c: str) -> str:
    c = str(c).strip().lower().replace('（', '(').replace('）', ')')
    keep = set('abcdefghijklmnopqrstuvwxyz0123456789()')
    return ''.join(ch for ch in c if ch in keep)

def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError("CSV must have >=3 columns: i/rep, t, V(t)")
    norm_map = {c: _norm_col(c) for c in df.columns}
    df = df.rename(columns={
        next(k for k,v in norm_map.items() if v in ('i','rep')): 'rep',
        next(k for k,v in norm_map.items() if v=='t'): 't',
        next(k for k,v in norm_map.items() if v in ('v(t)','vt','v')): 'Y',
    })
    df['rep'] = pd.to_numeric(df['rep'], errors='coerce')
    df['t']   = pd.to_numeric(df['t'],   errors='coerce')
    df['Y']   = pd.to_numeric(df['Y'],   errors='coerce')
    df = df.dropna(subset=['rep','t','Y']).copy()
    df['rep'] = df['rep'].astype(int)
    return df.sort_values(['rep','t']).reset_index(drop=True)

# ============================== KFE ===================================
def _as_fn(x):
    if callable(x): return lambda t: max(0.0, float(x(t)))
    v = float(x);   return lambda t: v

def lamE(t):   return 2.0 + np.sin(0.5*t)               # E2(t)
def lamH1(t):  return 2.0 + 0.6*np.sin(0.5*t)           # H2(t) phase-1
def lamH2(t):  return 0.8 + 0.4*np.cos(0.3*t)           # H2(t) phase-2
P_MIX = 0.6
MU, C = 1.0, 8
T_END = 16.0

def build_Q_H2M1c(lambda1, lambda2, p_mix, mu: float, c: int):
    l1f=_as_fn(lambda1); l2f=_as_fn(lambda2); pf=_as_fn(p_mix)
    def Q_of_t(t: float) -> np.ndarray:
        l1=l1f(t); l2=l2f(t); p=max(0.0, min(1.0, pf(t)))
        nst=2*(c+1); Q=np.zeros((nst,nst), float)
        for ph in (0,1):
            lam = l1 if ph==0 else l2
            for j in range(c+1):
                idx=ph*(c+1)+j; out=0.0
                if j>0: Q[idx, ph*(c+1)+(j-1)] += mu; out += mu
                tgt=j+1 if j<c else c
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
                if j>0: Q[idx, ph*(c+1)+(j-1)] += mu; out += mu
                if ph==0:
                    Q[idx, 1*(c+1)+j] += l; out += l
                else:
                    tgt=j+1 if j<c else c
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
        if s<=0.0: pn=p.copy(); s=pn.sum()
        if s!=1.0: pn/=s
        P[k+1]=pn
    return P

def v_wait_arrival_H2(P: np.ndarray, c: int, mu: float, t_grid: np.ndarray, lambda1, lambda2):
    nT=P.shape[0]; v=np.zeros(nT, float)
    j=np.arange(0,c+1, dtype=float); w=(j/mu)[:c]
    l1f=_as_fn(lambda1); l2f=_as_fn(lambda2)
    for k in range(nT):
        p1=P[k,:c+1]; p2=P[k,c+1:2*(c+1)]
        l1=l1f(t_grid[k]); l2=l2f(t_grid[k])
        num=l1*np.dot(w,p1[:c]) + l2*np.dot(w,p2[:c])
        den=l1*np.sum(p1[:c])  + l2*np.sum(p2[:c])
        v[k]=(num/den) if den>1e-14 else np.nan
    return v

def v_wait_arrival_E2(P: np.ndarray, c: int, mu: float):
    nT=P.shape[0]; v=np.zeros(nT,float)
    j=np.arange(0,c+1, dtype=float); w=(j/mu)[:c]
    for k in range(nT):
        p2=P[k,c+1:2*(c+1)]
        den=np.sum(p2[:c]); num=np.dot(w,p2[:c])
        v[k]=(num/den) if den>1e-14 else np.nan
    return v

def solve_h2_kfe(t_end=T_END, mu=MU, c=C, steps=4000, p0_phase=1):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_H2M1c(lamH1, lamH2, P_MIX, mu, c)
    nst=2*(c+1); p0=np.zeros(nst); p0[0 if p0_phase==1 else c+1]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_H2(P, c, mu, t, lamH1, lamH2)
    return t, v

def solve_e2_kfe(t_end=T_END, mu=MU, c=C, steps=4000, p0_phase=1):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_E2M1c(lamE, mu, c)
    nst=2*(c+1); p0=np.zeros(nst); p0[0 if p0_phase==1 else c+1]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_E2(P, c, mu)
    return t, v

# ============================ fixed-k kNN ==============================
def lorocv_best_k_equal(df: pd.DataFrame, k_max: int = 2000):
    reps = df['rep'].unique()
    N    = len(df)
    Mj   = df.groupby('rep').size().to_dict()
    min_train = min(N - Mj[r] for r in reps)
    kU = int(min(k_max, min_train))
    if kU < 1: raise ValueError("LORO needs >=2 replications.")

    sse = np.zeros(kU + 1, float); total=0
    for r in reps:
        test  = df[df['rep']==r]
        train = df[df['rep']!=r]
        t_te, y_te = test['t'].to_numpy(),  test['Y'].to_numpy()
        t_tr, y_tr = train['t'].to_numpy(), train['Y'].to_numpy()
        D = np.abs(t_te[:,None] - t_tr[None,:])               # (m_test, m_train)
        idx_part = np.argpartition(D, kU-1, axis=1)[:,:kU]    # top-kU
        D_part   = np.take_along_axis(D, idx_part, axis=1)
        order    = np.argsort(D_part, axis=1)
        nn_idx   = np.take_along_axis(idx_part, order, axis=1)
        Y_nn     = y_tr[nn_idx]                               # (m_test, kU)
        cums     = np.cumsum(Y_nn, axis=1)
        ks       = np.arange(1, kU+1)
        preds    = cums / ks
        errs     = (preds - y_te[:,None])**2
        sse[1:] += errs.sum(axis=0)
        total   += len(y_te)
    emse = sse[1:] / total
    return int(np.argmin(emse) + 1)

def knn_curve_equal(df: pd.DataFrame, k: int, grid_points: int = 600):
    t_tr = df['t'].to_numpy()
    y_tr = df['Y'].to_numpy()
    t_grid = np.linspace(float(t_tr.min()), float(t_tr.max()), grid_points)
    D = np.abs(t_grid[:,None] - t_tr[None,:])
    idx_k = np.argpartition(D, k-1, axis=1)[:,:k]
    vbar  = y_tr[idx_k].mean(axis=1)
    return t_grid, vbar

# ======================== Adaptive (local CV) =========================
def _side_flags(t0, t_min, t_max, frac=0.02):
    near_left  = (t0 - t_min) <= frac*(t_max - t_min)
    near_right = (t_max - t0) <= frac*(t_max - t_min)
    return bool(near_right), bool(near_left)

def _gather(ts, reps, t0, k, exclude_rep=None, left_only=False, right_only=False):
    mask = np.ones_like(ts, dtype=bool)
    if exclude_rep is not None: mask &= (reps != exclude_rep)
    if left_only:  mask &= (ts <= t0)
    if right_only: mask &= (ts >= t0)
    idxs = np.where(mask)[0]
    if idxs.size == 0: return []
    d = np.abs(ts[idxs] - t0)
    take = min(k, idxs.size)
    part = np.argpartition(d, take-1)[:take]
    near = idxs[part]
    near = near[np.argsort(np.abs(ts[near] - t0))]
    return near.tolist()

def _rep_pick_balanced(idxs, reps, k, max_per_rep):
    taken, cnt = [], {}
    for j in idxs:
        r = int(reps[j])
        if cnt.get(r,0) < max_per_rep:
            taken.append(j); cnt[r]=cnt.get(r,0)+1
            if len(taken)==k: return taken
    for j in idxs:
        if j not in taken:
            taken.append(j)
            if len(taken)==k: break
    return taken[:k]

def global_lorocv_k_equal_light(reps, ts, vs, sample_per_rep=25, seed=42):
    rng = np.random.default_rng(seed)
    rep_ids = np.unique(reps); N = ts.size
    Mj = {r: np.sum(reps==r) for r in rep_ids}
    k_upper = int(min(250, max(1, min(N - Mj[r] for r in rep_ids) - 1)))
    grid = np.linspace(3, max(10, k_upper), 22)
    k_cands = sorted(set(int(round(x)) for x in grid if x >= 3))
    idx_by_rep = {r: np.where(reps==r)[0] for r in rep_ids}
    sampled=[]
    for r in rep_ids:
        idxs = idx_by_rep[r]
        chosen = idxs if idxs.size<=sample_per_rep else rng.choice(idxs, size=25, replace=False)
        sampled.extend(chosen.tolist())
    sampled = np.array(sorted(sampled), dtype=int)
    best_k, best_mse = k_cands[0], float("inf")
    for k in k_cands:
        sqerrs=[]
        for idx in sampled:
            r=reps[idx]; t0=ts[idx]; v_true=vs[idx]
            left_only,right_only=_side_flags(t0, ts[0], ts[-1])
            cand=_gather(ts,reps,t0,k*4,exclude_rep=r,left_only=left_only,right_only=right_only)
            if len(cand)<max(20,k):
                cand=_gather(ts,reps,t0,max(6*k,80),exclude_rep=r)
            if not cand: continue
            max_per_rep=max(2, min(6, int(np.ceil(k/max(1,len(rep_ids))))+2))
            pick=_rep_pick_balanced(cand,reps,k,max_per_rep)
            if len(pick)==0: continue
            v_hat=float(np.mean(vs[pick]))
            sqerrs.append((v_hat - v_true)**2)
        if sqerrs:
            mse=float(np.mean(sqerrs))
            if mse<best_mse: best_mse, best_k=mse, k
    return best_k

def running_median_int(arr, win=11):
    if win<=1: return arr
    w=win if win%2==1 else win+1; r=w//2
    pad=np.r_[np.repeat(arr[0],r), arr, np.repeat(arr[-1],r)]
    out=np.empty_like(arr)
    for i in range(len(arr)):
        out[i]=int(np.median(pad[i:i+w]))
    return out

def local_adaptive_k_equal(t0, k_star, reps, ts, vs, seed=17):
    rng=np.random.default_rng(seed)
    t_min,t_max=float(ts[0]),float(ts[-1])
    left_only,right_only=_side_flags(t0,t_min,t_max)
    boost = (t_max - t0) <= 0.05*(t_max - t_min) or (t0 - t_min) <= 0.05*(t_max - t_min)
    M=min(400, ts.size)
    pool=_gather(ts,reps,t0,M,exclude_rep=None,left_only=left_only,right_only=right_only)
    if len(pool)<60:
        pool=_gather(ts,reps,t0,min(800,ts.size),exclude_rep=None)
    if len(pool)<12:
        return max(3, min(int(k_star), len(pool)-1))
    test_idx = pool if len(pool)<=60 else rng.choice(pool, size=60, replace=False)
    base=[int(round(k_star)), int(round(1.5*k_star)), int(round(2.0*k_star))]
    if boost: base.append(int(round(3.0*k_star)))
    cand_ks=sorted(set(k for k in base if k>=max(3,int(round(k_star)))))
    best_k, best_mse = cand_ks[0], float("inf")
    for k in cand_ks:
        if k>=len(pool): continue
        sqerrs=[]
        for j in test_idx:
            rj=reps[j]; vj=vs[j]
            cand=[x for x in pool if reps[x]!=rj]
            if len(cand)<3: continue
            cand=cand[:min(len(cand), 4*k)]
            n_rep=len(np.unique(reps[cand]))
            max_per_rep=max(2, min(6, int(np.ceil(k/max(1,n_rep)))+2))
            pick=_rep_pick_balanced(cand,reps,k,max_per_rep)
            if len(pick)==0: continue
            v_hat=float(np.mean(vs[pick]))
            sqerrs.append((v_hat - vj)**2)
        if sqerrs:
            mse=float(np.mean(sqerrs))
            if mse<best_mse: best_mse,best_k=mse,k
    return best_k

def adaptive_knn_curve_equal(df: pd.DataFrame, k_star: int, grid_points: int = 600):
    reps=df['rep'].to_numpy(); ts=df['t'].to_numpy(); vs=df['Y'].to_numpy()
    t_min, t_max=float(ts.min()), float(ts.max())
    t_grid=np.linspace(t_min, t_max, int(grid_points), endpoint=False)
    k_seq=np.zeros_like(t_grid, dtype=int)
    for i,t0 in enumerate(t_grid):
        k_seq[i]=local_adaptive_k_equal(t0, k_star, reps, ts, vs)
    k_seq=running_median_int(k_seq, win=13)  # 稍强一点去抖
    v_hat=np.full_like(t_grid, np.nan, dtype=float)
    for idx,t0 in enumerate(t_grid):
        k_loc=int(k_seq[idx])
        D=np.abs(ts - t0); idx_k=np.argpartition(D, k_loc-1)[:k_loc]
        v_hat[idx]=float(np.mean(vs[idx_k]))
    return t_grid, v_hat

# ============================ AFKNN (简版) ============================
# 这里复用“用KFE标注 + 随机森林学 log k(t) ”的最小实现，seed=7
from sklearn.ensemble import RandomForestRegressor

def make_feats(ts, t_points):
    t_min, t_max = float(ts.min()), float(ts.max())
    t_norm = (t_points - t_min) / max(1e-9, (t_max - t_min))
    return np.c_[t_norm]  # 用一个简单特征即可（已足够把时间窗口映射到k）

def train_kforest_on_csv(df: pd.DataFrame, is_h2: bool, steps_kfe=5000, seed=7):
    # 用KFE做平滑“真值”，在均匀网格上用局部CV+方差正则标注k*
    if is_h2:
        tk, vk = solve_h2_kfe(steps=steps_kfe)
    else:
        tk, vk = solve_e2_kfe(steps=steps_kfe)

    ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    t_min, t_max=float(ts.min()), float(ts.max())
    rng=np.random.default_rng(seed)
    anchors = np.linspace(t_min, t_max, 300, endpoint=False)  # 轻量
    k_cands = [20,35,50,80,120,160]
    lab_t, lab_k = [], []
    for t0 in anchors:
        # 取一小片邻域做local CV（不与rep相关，简单处理）
        D=np.abs(ts - t0)
        pool = np.argsort(D)[:800]
        if pool.size < 60:
            continue
        tj = ts[pool[:40]]
        v_kfe_local = np.interp(tj, tk, vk)
        best_k, best_obj = None, 1e9
        for k in k_cands:
            idxk = pool[:min(k, pool.size)]
            vhat = np.mean(ys[idxk])
            # bias^2 + 小方差惩罚
            b2 = np.mean((vhat - v_kfe_local)**2)
            var = np.var(ys[idxk], ddof=1)/max(1,len(idxk))
            obj = b2 + 3.0*var + 0.05*(k/160.0)
            if obj < best_obj:
                best_obj, best_k = obj, k
        if best_k is not None:
            lab_t.append(t0); lab_k.append(best_k)
    lab_t = np.asarray(lab_t); lab_k = np.asarray(lab_k)
    X = make_feats(ts, lab_t)
    rf = RandomForestRegressor(n_estimators=300, max_depth=9, min_samples_leaf=30,
                               random_state=0, n_jobs=-1)
    rf.fit(X, np.log(lab_k))
    return rf

def afknn_curve_from_csv(df: pd.DataFrame, rf: RandomForestRegressor, grid_points=600):
    ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    t_min, t_max=float(ts.min()), float(ts.max())
    t_grid=np.linspace(t_min, t_max, grid_points, endpoint=False)
    Xg = make_feats(ts, t_grid)
    k_pred = np.clip(np.round(np.exp(rf.predict(Xg))).astype(int), 5, 160)
    # 中位数去抖
    k_pred = running_median_int(k_pred, win=13)
    # 估计 v(t)
    v_hat=np.empty_like(t_grid)
    for i,t0 in enumerate(t_grid):
        k = int(k_pred[i])
        D = np.abs(ts - t0); idxk = np.argpartition(D, k-1)[:k]
        v_hat[i]=float(np.mean(ys[idxk]))
    return t_grid, v_hat

# ============================== Plotting ==============================
def plot_2x2(H2_csv="H2_pairs.csv", E2_csv="E2_pairs.csv", outpng="compare_2x2.png"):
    # load data
    dfH = load_pairs(H2_csv)
    dfE = load_pairs(E2_csv)

    # KFE truth on common grids
    tH_kfe, vH_kfe = solve_h2_kfe()
    tE_kfe, vE_kfe = solve_e2_kfe()

    # fixed-k via LORO
    kH = lorocv_best_k_equal(dfH, 2000); tH_knn, vH_knn = knn_curve_equal(dfH, kH)
    kE = lorocv_best_k_equal(dfE, 2000); tE_knn, vE_knn = knn_curve_equal(dfE, kE)

    # local-CV adaptive
    repsH, tsH, vsH = dfH['rep'].to_numpy(), dfH['t'].to_numpy(), dfH['Y'].to_numpy()
    repsE, tsE, vsE = dfE['rep'].to_numpy(), dfE['t'].to_numpy(), dfE['Y'].to_numpy()
    k_star_H = global_lorocv_k_equal_light(repsH, tsH, vsH, sample_per_rep=25)
    k_star_E = global_lorocv_k_equal_light(repsE, tsE, vsE, sample_per_rep=25)
    tH_ad, vH_ad = adaptive_knn_curve_equal(dfH, k_star_H)
    tE_ad, vE_ad = adaptive_knn_curve_equal(dfE, k_star_E)

    # AFKNN (seed=7)
    rfH = train_kforest_on_csv(dfH, is_h2=True,  seed=7)
    rfE = train_kforest_on_csv(dfE, is_h2=False, seed=7)
    tH_af, vH_af = afknn_curve_from_csv(dfH, rfH)
    tE_af, vE_af = afknn_curve_from_csv(dfE, rfE)

    # ------ figure ------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    lw_kfe, lw_o = 2.6, 1.9

    # 左上: H2 — KFE + kNN + Adaptive
    ax = axes[0,0]
    ax.plot(tH_kfe, vH_kfe, color=COL["kfe"],   lw=lw_kfe, label="KFE (truth)")
    ax.plot(tH_knn, vH_knn, color=COL["knn"],   lw=lw_o,   label="kNN (fixed k)")
    ax.plot(tH_ad,  vH_ad,  color=COL["adapt"], lw=lw_o,   label="Adaptive kNN")
    ax.set_title("H2(t)/M/1/8: KFE + kNN + Adaptive")
    ax.set_xlabel("Arrival time t"); ax.set_ylabel("Virtual waiting time v(t)")
    ax.grid(True, ls='--', alpha=0.35); ax.legend(loc="upper left")

    # 右上: H2 — KFE + kNN + AFKNN
    ax = axes[0,1]
    ax.plot(tH_kfe, vH_kfe, color=COL["kfe"],   lw=lw_kfe, label="KFE (truth)")
    ax.plot(tH_knn, vH_knn, color=COL["knn"],   lw=lw_o,   label="kNN (fixed k)")
    ax.plot(tH_af,  vH_af,  color=COL["afknn"], lw=lw_o,   label="AFKNN (forest k(t))")
    ax.set_title("H2(t)/M/1/8: KFE + kNN + AFKNN")
    ax.set_xlabel("Arrival time t"); ax.set_ylabel("Virtual waiting time v(t)")
    ax.grid(True, ls='--', alpha=0.35); ax.legend(loc="upper left")

    # 左下: E2 — KFE + kNN + Adaptive
    ax = axes[1,0]
    ax.plot(tE_kfe, vE_kfe, color=COL["kfe"],   lw=lw_kfe, label="KFE (truth)")
    ax.plot(tE_knn, vE_knn, color=COL["knn"],   lw=lw_o,   label="kNN (fixed k)")
    ax.plot(tE_ad,  vE_ad,  color=COL["adapt"], lw=lw_o,   label="Adaptive kNN")
    ax.set_title("E2(t)/M/1/8: KFE + kNN + Adaptive")
    ax.set_xlabel("Arrival time t"); ax.set_ylabel("Virtual waiting time v(t)")
    ax.grid(True, ls='--', alpha=0.35); ax.legend(loc="upper left")

    # 右下: E2 — KFE + kNN + AFKNN
    ax = axes[1,1]
    ax.plot(tE_kfe, vE_kfe, color=COL["kfe"],   lw=lw_kfe, label="KFE (truth)")
    ax.plot(tE_knn, vE_knn, color=COL["knn"],   lw=lw_o,   label="kNN (fixed k)")
    ax.plot(tE_af,  vE_af,  color=COL["afknn"], lw=lw_o,   label="AFKNN (forest k(t))")
    ax.set_title("E2(t)/M/1/8: KFE + kNN + AFKNN")
    ax.set_xlabel("Arrival time t"); ax.set_ylabel("Virtual waiting time v(t)")
    ax.grid(True, ls='--', alpha=0.35); ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(outpng, dpi=170)
    plt.close()
    print(f"[Saved] {outpng}")

if __name__ == "__main__":
    plot_2x2("H2_pairs.csv", "E2_pairs.csv", outpng="compare_2x2.png")
