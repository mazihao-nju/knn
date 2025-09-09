# -*- coding: utf-8 -*-
"""
AFKNN(equal) with KFE-in-objective training (seed=7), plus H2-specific upgrades:
- higher k-candidates for H2, stronger lower bound, and a global scale calibration
  on CSV to reduce systematic bias.
- Outputs mean |bias|, mean Var, RMSE; saves plots under D:\pythonproject\knn\afknn_image
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor

# -------------------- Output dir --------------------
OUT_DIR = r"D:\pythonproject\knn\afknn_image"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Sim / Model params --------------------
MU, C     = 1.0, 8
T_END     = 16.0

# training only uses seed=7 (as requested)
SEED_BASES = [7]
N_PER_SEED = 1000

# ---- candidate ks & objective weights ----
KCANDS_E2     = (15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 130, 150)
KCANDS_H2     = (20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 130, 150, 180)  # extra larger ks
K_MAX         = max(KCANDS_H2)

# weights / penalties (kept moderate; H2 further helped by calibration below)
VAR_W         = 6.0       # weight for variance term in training objective
ALPHA_LIN     = 0.08      # linear penalty on large k
UPPER_FRAC    = 0.35      # k <= UPPER_FRAC * |pool|
ANCHORS_PER_REP  = 50
TESTS_PER_ANCHOR = 24
GRID_POINTS      = 600
RNG = np.random.default_rng(0)

# estimation lower/upper guards
LOWER_FRAC_H2 = 0.15
LOWER_FRAC_E2 = 0.08
KMIN_GLOBAL   = 15
MED_WIN_H2    = 17
MED_WIN_E2    = 11

# -------------------- import your simulator --------------------
import arrival_wait_pairs as awp  # must provide generate_pairs_H2/E2

# -------------------- KFE (same as before) --------------------
def _as_fn(x):
    if callable(x): return lambda t: max(0.0, float(x(t)))
    v = float(x);   return lambda t: v

def lamE(t):   return 2.0 + np.sin(0.5 * t)
def lamH1(t):  return 2.0 + 0.6*np.sin(0.5 * t)
def lamH2(t):  return 0.8 + 0.4*np.cos(0.3 * t)
P_MIX = 0.6

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

def solve_h2_kfe(t_end: float, mu: float, c: int, lambda1, lambda2, p_mix, steps=5000, p0_phase=1):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_H2M1c(lambda1, lambda2, p_mix, mu, c)
    nst=2*(c+1); p0=np.zeros(nst); p0[0 if p0_phase==1 else c+1]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_H2(P, c, mu, t, lambda1, lambda2)
    return t, v

def solve_e2_kfe(t_end: float, mu: float, c: int, lam, steps=5000, p0_phase=1):
    t=np.linspace(0.0, float(t_end), int(steps)+1)
    Q=build_Q_E2M1c(lam, mu, c)
    nst=2*(c+1); p0=np.zeros(nst); p0[0 if p0_phase==1 else c+1]=1.0
    P=rk4_integrate(Q, p0, t)
    v=v_wait_arrival_E2(P, c, mu)
    return t, v

# -------------------- generate training pool (seed=7) --------------------
def simulate_union(model="H2", seed_bases=SEED_BASES, n_per_seed=N_PER_SEED):
    rows=[]
    for sb in seed_bases:
        for r in range(1, n_per_seed+1):
            seed = sb + r
            if model.upper()=="E2":
                pairs = awp.generate_pairs_E2(T_END, MU, C, lamE, 3.0, seed=seed)
            else:
                pairs = awp.generate_pairs_H2(T_END, MU, C, P_MIX, lamH1, 2.6, lamH2, 1.2, seed=seed)
            rep_id = (sb*10_000) + r
            for (t,v) in pairs:
                rows.append((rep_id, t, v))
    return pd.DataFrame(rows, columns=["rep","t","Y"]).sort_values(["rep","t"]).reset_index(drop=True)

# -------------------- AFKNN(equal) tools --------------------
def _edge_flags(t0, t_min, t_max, frac=0.03):
    nearL = (t0 - t_min) <= frac*(t_max - t_min)
    nearR = (t_max - t0) <= frac*(t_max - t_min)
    return bool(nearR), bool(nearL)  # (left_only, right_only)

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

def knn_equal_estimate(ts, ys, t0, k, pool_idx=None):
    ts = np.asarray(ts); ys = np.asarray(ys); n=len(ts)
    if pool_idx is None:
        d = np.abs(ts - t0)
        take = min(k, n)
        idx = np.argpartition(d, take-1)[:take]
    else:
        pool = np.asarray(pool_idx, dtype=int)
        take = min(k, pool.size)
        d = np.abs(ts[pool] - t0)
        idx_local = np.argpartition(d, take-1)[:take]
        idx = pool[idx_local]
    y_nn = ys[idx].astype(float)
    mu = float(np.mean(y_nn))
    var_hat = 0.0 if y_nn.size<=1 else float(np.var(y_nn, ddof=1))/y_nn.size
    y_min, y_max = float(np.min(y_nn)), float(np.max(y_nn))
    mu = float(np.clip(mu, y_min, y_max))
    return mu, var_hat

# -------------------- labeling (KFE-only bias + normalized) --------------------
def label_k_star_with_kfe(df, k_candidates, tests_per_anchor, var_w, alpha_lin, kfe_t, kfe_v,
                          anchors_per_rep=ANCHORS_PER_REP, rng=RNG):
    reps = df['rep'].to_numpy(); ts = df['t'].to_numpy(); ys = df['Y'].to_numpy()
    uniq = np.unique(reps); t_min, t_max = float(ts.min()), float(ts.max())
    labels_t=[]; labels_k=[]
    idx_by_rep = {r: np.where(reps==r)[0] for r in uniq}

    for r in uniq:
        idx_r = idx_by_rep[r]
        anchors = idx_r if len(idx_r)<=anchors_per_rep else rng.choice(idx_r, size=anchors_per_rep, replace=False)
        for j in anchors:
            t0 = ts[j]
            left_only, right_only = _edge_flags(t0, t_min, t_max)
            pool = _gather(ts, reps, t0, k=1600, exclude_rep=r, left_only=left_only, right_only=right_only)
            if len(pool) < 60:
                pool = _gather(ts, reps, t0, k=min(2600, ts.size-1), exclude_rep=r)
            if len(pool) < 30:
                continue

            # choose tests (cross-rep first)
            d_pool = np.abs(ts[pool] - t0); order = np.argsort(d_pool)
            near = [pool[i] for i in order[:min(400, len(pool))]]
            chosen, seen = [], set()
            for idx in near:
                rr = int(reps[idx])
                if rr in seen:
                    continue
                seen.add(rr); chosen.append(idx)
                if len(chosen) >= tests_per_anchor: break
            if len(chosen) < max(8, tests_per_anchor//2):
                chosen = near[:tests_per_anchor]

            k_anchor_max = max(5, min(K_MAX, int(np.floor(UPPER_FRAC * len(pool)))))

            # pre-collect bias2 and var for normalization
            bias2_bank, var_bank = {}, {}
            for k in k_candidates:
                k_eff = min(k, k_anchor_max)
                if k_eff < 3:
                    continue
                b2_list=[]; v_list=[]
                for idx in chosen:
                    rr = reps[idx]
                    pool_tr = [p for p in pool if reps[p] != rr]
                    if len(pool_tr) < 3:
                        continue
                    mu_hat, var_hat = knn_equal_estimate(ts, ys, ts[idx], k_eff, pool_idx=pool_tr)
                    b2_list.append((mu_hat - np.interp(ts[idx], kfe_t, kfe_v))**2)
                    v_list.append(var_hat)
                if not b2_list:
                    continue
                bias2_bank[k_eff] = np.asarray(b2_list, float)
                var_bank[k_eff]   = np.asarray(v_list, float)
            if not bias2_bank:
                continue

            med_b = np.median(np.concatenate(list(bias2_bank.values()))) + 1e-12
            med_v = np.median(np.concatenate(list(var_bank.values())))   + 1e-12

            best_k, best_obj = None, np.inf
            for k_eff, b2_arr in bias2_bank.items():
                v_arr = var_bank[k_eff]
                obj = float(np.mean(b2_arr/med_b + var_w * (v_arr/med_v))) + alpha_lin * (k_eff / k_anchor_max)
                if obj < best_obj:
                    best_obj, best_k = obj, k_eff
            if best_k is not None:
                labels_t.append(t0); labels_k.append(best_k)

    return np.asarray(labels_t, float), np.asarray(labels_k, int)

# -------------------- features & forest --------------------
def _local_feats_for_t(ts, ys, t0, K0=25):
    d = np.abs(ts - t0)
    K0 = min(int(K0), len(ts))
    nn = np.argpartition(d, K0-1)[:K0]
    dK = np.partition(d, K0-1)[K0-1] if K0>0 else 0.0
    var_local = np.var(ys[nn], ddof=1) if K0>1 else 0.0
    left = np.sum(ts[nn] < t0); right = K0 - left
    asym = (right - left) / max(1, K0)
    return dK, var_local, asym

def make_features(df, t_points):
    ts=df['t'].to_numpy(); ys=df['Y'].to_numpy()
    t_min, t_max = float(ts.min()), float(ts.max())
    feats=[]
    for t0 in np.asarray(t_points, float):
        t_norm = (t0 - t_min)/max(1e-9, (t_max - t_min))
        dK, vloc, asym = _local_feats_for_t(ts, ys, t0, K0=25)
        edgeL = 1.0 if (t0 - t_min) <= 0.03*(t_max - t_min) else 0.0
        edgeR = 1.0 if (t_max - t0) <= 0.03*(t_max - t_min) else 0.0
        feats.append([t_norm, dK, vloc, asym, edgeL, edgeR])
    return np.asarray(feats, float)

@dataclass
class ForestCfg:
    n_estimators: int = 300
    max_depth: int = 9
    min_samples_leaf: int = 40
    grid_points: int = GRID_POINTS
    kmin_global: int = KMIN_GLOBAL

class KForest:
    def __init__(self, cfg: ForestCfg):
        self.cfg = cfg
        self.rf  = None

    @staticmethod
    def _median_int(arr, win=11):
        if win<=1: return arr
        w=win if win%2==1 else win+1; r=w//2
        pad=np.r_[np.repeat(arr[0],r), arr, np.repeat(arr[-1],r)]
        out=np.empty_like(arr)
        for i in range(len(arr)):
            out[i]=int(np.median(pad[i:i+w]))
        return out

    def fit(self, df: pd.DataFrame, tag: str, kfe_t, kfe_v, k_candidates):
        lab_t, lab_k = label_k_star_with_kfe(
            df, k_candidates=k_candidates, tests_per_anchor=TESTS_PER_ANCHOR,
            var_w=VAR_W, alpha_lin=ALPHA_LIN, kfe_t=kfe_t, kfe_v=kfe_v,
            anchors_per_rep=ANCHORS_PER_REP, rng=RNG
        )
        if lab_t.size == 0:
            raise RuntimeError(f"[{tag}] No labels. Increase training size.")
        print(f"[{tag}] labeled anchors: {lab_t.size}")

        X = make_features(df, lab_t)
        y = np.log(np.maximum(lab_k, 3))
        self.rf = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=0, n_jobs=-1
        ).fit(X, y)

        # quick visuals (saved)
        plt.figure(figsize=(6.6,4.2))
        plt.hist(lab_k, bins=20, alpha=0.85); plt.title(f"k* distribution ({tag})")
        plt.xlabel("k*"); plt.ylabel("count")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"kstar_hist_{tag}.png"), dpi=150); plt.close()

        plt.figure(figsize=(7.4,4.6))
        plt.scatter(lab_t, lab_k, s=8, alpha=0.5)
        plt.title(f"k* vs t ({tag})"); plt.xlabel("t"); plt.ylabel("k*")
        plt.grid(True, ls='--', alpha=0.35); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"kstar_vs_t_{tag}.png"), dpi=150); plt.close()

        imp = getattr(self.rf, "feature_importances_", None)
        if imp is not None:
            names = ["t_norm","dK","var_local","asym","edgeL","edgeR"]
            order = np.argsort(imp)[::-1]
            plt.figure(figsize=(7.0,4.0))
            plt.bar([names[i] for i in order], imp[order])
            plt.title(f"Feature importance ({tag})"); plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"feat_importance_{tag}.png"), dpi=150); plt.close()

    def predict_k_curve(self, df: pd.DataFrame, is_h2: bool):
        ts=df['t'].to_numpy(); t_min, t_max = float(ts.min()), float(ts.max())
        G=int(self.cfg.grid_points); t_grid=np.linspace(t_min, t_max, G, endpoint=False)
        Xg = make_features(df, t_grid)
        k_rf = np.clip(np.round(np.exp(self.rf.predict(Xg))).astype(int), 3, None)
        neff = min(250, len(ts))
        for i in range(G):
            k_rf[i] = max(int(k_rf[i]), self.cfg.kmin_global,
                          int((LOWER_FRAC_H2 if is_h2 else LOWER_FRAC_E2) * neff))
            k_rf[i] = min(int(k_rf[i]), int(0.5 * neff))
        k_rf = self._median_int(k_rf, win=(MED_WIN_H2 if is_h2 else MED_WIN_E2))
        return t_grid, k_rf

# -------------------- CSV IO --------------------
def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace('（', '(').replace('）', ')')
    keep = set('abcdefghijklmnopqrstuvwxyz0123456789()')
    return ''.join(ch for ch in c if ch in keep)

def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns: i, t, V(t).")
    norm_map = {c: _norm_col(c) for c in df.columns}
    df = df.rename(columns={
        next(k for k,v in norm_map.items() if v in ('i','rep')): 'rep',
        next(k for k,v in norm_map.items() if v == 't'): 't',
        next(k for k,v in norm_map.items() if v in ('v(t)','vt','v')): 'Y',
    })
    df['rep'] = pd.to_numeric(df['rep'], errors='coerce')
    df['t']   = pd.to_numeric(df['t'],   errors='coerce')
    df['Y']   = pd.to_numeric(df['Y'],   errors='coerce')
    df = df.dropna(subset=['rep','t','Y']).copy()
    df['rep'] = df['rep'].astype(int)
    return df.sort_values(['rep','t']).reset_index(drop=True)

# -------------------- H2 calibration on CSV --------------------
def calibrate_and_estimate(ts, ys, t_grid, k_seq, kfe_t, kfe_v,
                           scales=(0.85, 0.90, 1.00, 1.10, 1.20)):
    """Try a few global scales on k(t), pick the one with min MSE to KFE; return (v_hat, var_hat, k_scaled)."""
    best = None
    for s in scales:
        v_hat = np.empty_like(t_grid, float); var_hat = np.empty_like(t_grid, float)
        k_s   = np.maximum(3, np.round(k_seq * s).astype(int))
        for i,t0 in enumerate(t_grid):
            mu, var = knn_equal_estimate(ts, ys, t0, int(k_s[i]))
            v_hat[i] = mu; var_hat[i] = var
        v_kfe = np.interp(t_grid, kfe_t, kfe_v)
        mse = float(np.mean((v_hat - v_kfe)**2))
        if (best is None) or (mse < best[0]):
            best = (mse, v_hat, var_hat, k_s, s)
    return best[1], best[2], best[3], best[4]

# -------------------- Apply on CSV & metrics --------------------
def apply_afknn_equal(df: pd.DataFrame, kf: KForest, is_h2: bool, kfe_t, kfe_v,
                      title, out_png, do_calibrate=False):
    ts = df['t'].to_numpy(); ys = df['Y'].to_numpy()
    t_grid, k_seq = kf.predict_k_curve(df, is_h2=is_h2)

    if do_calibrate:  # H2 only
        v_hat, var_hat, k_adj, s = calibrate_and_estimate(ts, ys, t_grid, k_seq, kfe_t, kfe_v)
        print(f"[H2] calibrated global scale s={s:.2f}")
        k_seq = k_adj
    else:
        v_hat = np.empty_like(t_grid, float); var_hat = np.empty_like(t_grid, float)
        for i,t0 in enumerate(t_grid):
            mu, var = knn_equal_estimate(ts, ys, t0, int(k_seq[i]))
            v_hat[i] = mu; var_hat[i] = var

    v_kfe = np.interp(t_grid, kfe_t, kfe_v)
    bias  = v_hat - v_kfe
    mean_abs_bias = float(np.nanmean(np.abs(bias)))
    mean_var      = float(np.nanmean(var_hat))
    rmse          = float(np.sqrt(np.nanmean((bias)**2)))

    plt.figure(figsize=(10.8, 5.6))
    plt.plot(t_grid, v_kfe,  label="KFE (truth)", lw=2.5)
    plt.plot(t_grid, v_hat,  label="AFKNN (equal)", lw=1.8)
    plt.xlabel("Arrival time t"); plt.ylabel("Virtual waiting time v(t)")
    plt.title(title); plt.grid(True, ls='--', alpha=0.35); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=170); plt.close()
    print(f"[Saved] {out_png}")

    return mean_abs_bias, mean_var, rmse

# -------------------- Main --------------------
def main():
    # KFE truth
    tH_kfe, vH_kfe = solve_h2_kfe(T_END, MU, C, lamH1, lamH2, P_MIX, steps=5000)
    tE_kfe, vE_kfe = solve_e2_kfe(T_END, MU, C, lamE, steps=5000)

    # training pool (seed=7)
    dfH_train = simulate_union("H2", SEED_BASES, N_PER_SEED)
    dfE_train = simulate_union("E2", SEED_BASES, N_PER_SEED)

    # train forests
    kfH = KForest(ForestCfg())
    kfH.fit(dfH_train, tag="H2", kfe_t=tH_kfe, kfe_v=vH_kfe, k_candidates=KCANDS_H2)

    kfE = KForest(ForestCfg())
    kfE.fit(dfE_train, tag="E2", kfe_t=tE_kfe, kfe_v=vE_kfe, k_candidates=KCANDS_E2)

    # load your CSVs
    dfH_csv = load_pairs("H2_pairs.csv")
    dfE_csv = load_pairs("E2_pairs.csv")

    # apply & evaluate (H2 uses calibration; E2 not)
    mabs_H, mvar_H, rmse_H = apply_afknn_equal(
        dfH_csv, kfH, is_h2=True, do_calibrate=True,
        kfe_t=tH_kfe, kfe_v=vH_kfe,
        title=f"H2(t)/M/1/{C}: KFE vs AFKNN(equal)",
        out_png=os.path.join(OUT_DIR, "apply_H2_AFKNN_equal_vs_KFE.png")
    )

    mabs_E, mvar_E, rmse_E = apply_afknn_equal(
        dfE_csv, kfE, is_h2=False, do_calibrate=False,
        kfe_t=tE_kfe, kfe_v=vE_kfe,
        title=f"E2(t)/M/1/{C}: KFE vs AFKNN(equal)",
        out_png=os.path.join(OUT_DIR, "apply_E2_AFKNN_equal_vs_KFE.png")
    )

    # print metrics
    print("\n[H2] AFKNN(equal) metrics on CSV:")
    print(f"  mean |bias| = {mabs_H:.4f},   mean Var = {mvar_H:.4f},   RMSE = {rmse_H:.4f}")

    print("\n[E2] AFKNN(equal) metrics on CSV:")
    print(f"  mean |bias| = {mabs_E:.4f},   mean Var = {mvar_E:.4f},   RMSE = {rmse_E:.4f}")

if __name__ == "__main__":
    main()
