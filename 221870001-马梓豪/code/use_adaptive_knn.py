# -*- coding: utf-8 -*-
"""
Adaptive kNN for v(t): 等权版（只做 k(t) 自适应，向上调整 + 轻度去抖）
- 读取 E2_pairs.csv / H2_pairs.csv（列名兼容 i/t/V(t)）
- 全局 LORO-CV 先选基准 k*（评估器=等权 kNN，和最终一致）
- 局部自适应 k(t)：仅允许在 {k*, 1.25k*, 1.5k*, 2k*}（边界再加 2.5k*）中“向上”调整
- 对 k(t) 做轻度运行中位数去抖（窗口可调，默认9）
- 单侧邻居（左端→只取右；右端→只取左），候选不足自动放宽为双侧
- 复制均衡（每复制软上限 2~6），避免单一复制主导
- 图内英文，右上角标注 n=、k*=

保存为 use_adaptive_knn.py 并在含 E2_pairs.csv/H2_pairs.csv 的目录运行。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ========== 读 CSV 与列名归一化 ==========

def _norm_col(c: str) -> str:
    c = str(c).strip().lower()
    c = c.replace('（', '(').replace('）', ')')
    keep = set('abcdefghijklmnopqrstuvwxyz0123456789()')
    return ''.join(ch for ch in c if ch in keep)

def load_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] < 3:
        raise ValueError("CSV 至少需要三列：i, t, V(t)")
    norm_map = {c: _norm_col(c) for c in df.columns}
    rep_col = next(k for k, v in norm_map.items() if v in ('i', 'rep'))
    t_col   = next(k for k, v in norm_map.items() if v == 't')
    y_col   = next(k for k, v in norm_map.items() if v in ('v(t)', 'vt', 'v'))
    df = df.rename(columns={rep_col: 'rep', t_col: 't', y_col: 'Y'})
    df['rep'] = pd.to_numeric(df['rep'], errors='coerce').astype('Int64')
    df['t']   = pd.to_numeric(df['t'], errors='coerce')
    df['Y']   = pd.to_numeric(df['Y'], errors='coerce')
    df = df.dropna(subset=['rep', 't', 'Y']).copy()
    df['rep'] = df['rep'].astype(int)
    return df.sort_values(['rep', 't']).reset_index(drop=True)


# ========== 邻居检索 / 边界处理 / 复制均衡 ==========

def _local_side_flags(t0, t_min, t_max, frac=0.02):
    """
    边界单侧邻居策略：
    - 靠左端 -> 只取右侧 (right_only=True)
    - 靠右端 -> 只取左侧 (left_only=True)
    """
    near_left  = (t0 - t_min) <= frac * (t_max - t_min)
    near_right = (t_max - t0) <= frac * (t_max - t_min)
    left_only  = bool(near_right)  # 右端 → 只取左
    right_only = bool(near_left)   # 左端 → 只取右
    return left_only, right_only

def _gather_neighbors(ts, reps, t0, k, exclude_rep=None, left_only=False, right_only=False):
    """按 |t-t0| 从近到远收集候选邻居（不做复制均衡）"""
    mask = np.ones_like(ts, dtype=bool)
    if exclude_rep is not None:
        mask &= (reps != exclude_rep)
    if left_only:
        mask &= (ts <= t0)
    if right_only:
        mask &= (ts >= t0)
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return []
    d = np.abs(ts[idxs] - t0)
    take = min(k, idxs.size)
    part = np.argpartition(d, take - 1)[:take]
    near = idxs[part]
    near = near[np.argsort(np.abs(ts[near] - t0))]
    return near.tolist()

def _rep_balanced_pick(idxs, reps, k, max_per_rep):
    """
    复制均衡挑选 k 个邻居：
    - 第一轮在“每复制软上限”约束下从近到远选
    - 若不足，再无视上限补齐
    """
    taken, cnt = [], {}
    for j in idxs:
        r = int(reps[j])
        if cnt.get(r, 0) < max_per_rep:
            taken.append(j)
            cnt[r] = cnt.get(r, 0) + 1
            if len(taken) == k:
                return taken
    for j in idxs:
        if j not in taken:
            taken.append(j)
            if len(taken) == k:
                break
    return taken[:k]


# ========== 全局 LORO-CV（评估器=等权 kNN）选择基准 k* ==========

def global_loro_cv_k_equal(reps, ts, vs, k_candidates=None, sample_per_rep=25, seed=42):
    """
    全局 LORO-CV：按“等权 kNN”评估 MSE，选择 k*。
    - 为控算力，每复制抽样 sample_per_rep 个点做验证；可增大以更精确。
    - 候选自动生成：在 [3, k_upper] 等距取 ~22 个整数；软上限 250。
    """
    rng = np.random.default_rng(seed)
    rep_ids = np.unique(reps)
    n_rep = len(rep_ids)
    N = ts.size

    # 每折最小训练量 -> k 的上界
    Mj = {r: np.sum(reps == r) for r in rep_ids}
    k_upper_data = max(1, min(N - Mj[r] for r in rep_ids) - 1)
    k_upper = int(min(250, k_upper_data))

    if k_candidates is None:
        grid = np.linspace(3, max(10, k_upper), 22)
        k_candidates = sorted(set(int(round(x)) for x in grid if x >= 3))

    # 验证点抽样
    idx_by_rep = {r: np.where(reps == r)[0] for r in rep_ids}
    sampled = []
    for r in rep_ids:
        idxs = idx_by_rep[r]
        chosen = idxs if idxs.size <= sample_per_rep else rng.choice(idxs, size=sample_per_rep, replace=False)
        sampled.extend(chosen.tolist())
    sampled = np.array(sorted(sampled), dtype=int)

    best_k, best_mse = k_candidates[0], float("inf")
    for k in k_candidates:
        if k < 1 or k > k_upper:
            continue
        sqerrs = []
        for idx in sampled:
            r = reps[idx]; t0 = ts[idx]; v_true = vs[idx]
            left_only, right_only = _local_side_flags(t0, ts[0], ts[-1])

            # 大候选（先按单侧）；不足则放宽为双侧
            cand = _gather_neighbors(ts, reps, t0, k * 4, exclude_rep=r,
                                     left_only=left_only, right_only=right_only)
            if len(cand) < max(20, k):
                cand = _gather_neighbors(ts, reps, t0, k=max(6 * k, 80), exclude_rep=r,
                                         left_only=False, right_only=False)
            if not cand:
                continue

            # 复制均衡后取 k 个邻居（等权平均）
            max_per_rep = max(2, min(6, int(np.ceil(k / max(1, n_rep))) + 2))
            pick = _rep_balanced_pick(cand, reps, k=k, max_per_rep=max_per_rep)
            if len(pick) == 0:
                continue

            v_hat = float(np.mean(vs[pick]))
            sqerrs.append((v_hat - v_true) ** 2)

        if sqerrs:
            mse = float(np.mean(sqerrs))
            if mse < best_mse:
                best_mse, best_k = mse, k
    return best_k


# ========== 局部自适应：只“向上”调整 k(t) + 轻度去抖 ==========

def _local_candidates(k_star, boost=False):
    # 只向上调整：{k*, 1.25k*, 1.5k*, 2k*}，边界再加 2.5k*
    base = [int(round(k_star)),
            int(round(1.25 * k_star)),
            int(round(1.5  * k_star)),
            int(round(2.0  * k_star))]
    if boost:
        base.append(int(round(2.5 * k_star)))
    ks = sorted(set(k for k in base if k >= max(3, int(round(k_star)))))
    return ks

def local_adaptive_k_equal(t0, k_star, reps, ts, vs, seed=17):
    """
    以 k* 为中心的“向上候选”，通过局部 LOO 选择等权 kNN 的最优 k(t)。
    """
    rng = np.random.default_rng(seed)
    t_min, t_max = float(ts[0]), float(ts[-1])
    left_only, right_only = _local_side_flags(t0, t_min, t_max)
    boost = (t_max - t0) <= 0.05*(t_max - t_min) or (t0 - t_min) <= 0.05*(t_max - t_min)

    # 局部大候选池（单侧），不足则双侧扩大
    M = min(400, ts.size)
    pool = _gather_neighbors(ts, reps, t0, k=M, exclude_rep=None,
                             left_only=left_only, right_only=right_only)
    if len(pool) < 60:
        pool = _gather_neighbors(ts, reps, t0, k=min(800, ts.size), exclude_rep=None,
                                 left_only=False, right_only=False)

    if len(pool) < 12:
        return max(3, min(int(k_star), len(pool) - 1))

    # 局部 CV：从 pool 取最多 60 个点为验证集
    test_idx = pool if len(pool) <= 60 else rng.choice(pool, size=60, replace=False)
    cand_ks = _local_candidates(k_star, boost=boost)

    best_k, best_mse = cand_ks[0], float("inf")
    for k in cand_ks:
        if k >= len(pool):
            continue
        sqerrs = []
        for j in test_idx:
            rj = reps[j]; vj = vs[j]
            # 候选邻居：排除同复制，保留近的 4k，并做复制均衡
            cand = [x for x in pool if reps[x] != rj]
            if len(cand) < 3:
                continue
            cand = cand[:min(len(cand), 4*k)]
            n_rep = len(np.unique(reps[cand]))
            max_per_rep = max(2, min(6, int(np.ceil(k / max(1, n_rep))) + 2))
            pick = _rep_balanced_pick(cand, reps, k=k, max_per_rep=max_per_rep)
            if len(pick) == 0:
                continue

            v_hat = float(np.mean(vs[pick]))  # 等权平均
            sqerrs.append((v_hat - vj) ** 2)

        if sqerrs:
            mse = float(np.mean(sqerrs))
            if mse < best_mse:
                best_mse, best_k = mse, k
    return best_k

def _running_median_int(arr, win=9):
    """整数序列的一维运行中位数（奇数窗口），用于轻度去抖 k(t)"""
    if win <= 1:
        return arr
    w = win if win % 2 == 1 else win + 1
    r = w // 2
    pad = np.r_[np.repeat(arr[0], r), arr, np.repeat(arr[-1], r)]
    out = np.empty_like(arr)
    for i in range(len(arr)):
        out[i] = int(np.median(pad[i:i+w]))
    return out


# ========== 计算自适应等权 kNN 曲线 ==========

def adaptive_knn_curve_equal(reps, ts, vs, k_star, grid_points=450):
    t_min, t_max = float(ts.min()), float(ts.max())
    # 不在终点取值，避免“无未来数据”极端
    t_grid = np.linspace(t_min, t_max, int(grid_points), endpoint=False)

    # 先得到原始 k(t)
    k_seq = np.zeros_like(t_grid, dtype=int)
    for i, t0 in enumerate(t_grid):
        k_seq[i] = local_adaptive_k_equal(t0, k_star, reps, ts, vs)

    # 对 k(t) 做轻度运行中位数去抖
    k_seq = _running_median_int(k_seq, win=9)

    # 用平滑后的 k(t) 估计 v(t)
    v_hat = np.full_like(t_grid, fill_value=np.nan, dtype=float)
    for idx, t0 in enumerate(t_grid):
        k_loc = int(k_seq[idx])

        left_only, right_only = _local_side_flags(t0, t_min, t_max)
        cand = _gather_neighbors(ts, reps, t0, k=max(4 * k_loc, 40),
                                 exclude_rep=None,
                                 left_only=left_only, right_only=right_only)
        if len(cand) < max(20, k_loc):
            cand = _gather_neighbors(ts, reps, t0, k=max(6 * k_loc, 80),
                                     exclude_rep=None,
                                     left_only=False, right_only=False)
        if not cand:
            continue

        n_rep = len(np.unique(reps))
        max_per_rep_est = max(2, min(6, int(np.ceil(k_loc / max(1, n_rep))) + 2))
        pick = _rep_balanced_pick(cand, reps, k=k_loc, max_per_rep=max_per_rep_est)
        if len(pick) == 0:
            continue

        v_hat[idx] = float(np.mean(vs[pick]))  # 等权平均

    return t_grid, v_hat


# ========== 绘图 ==========

def plot_curve(t_grid, vbar, dataset_name, outfile, n_paths, k_star):
    fig, ax = plt.subplots()
    ax.plot(t_grid, vbar, linewidth=2, label='Adaptive kNN v(t) (equal-weight)')
    ax.set_xlabel('Arrival time t')
    ax.set_ylabel('Virtual waiting time v(t)')
    ax.set_title(f'{dataset_name}(t)/M/1/c by adaptive kNN (equal-weight)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best')
    ax.text(
        0.985, 0.985, f'n={n_paths}, k*={k_star}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85)
    )
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f'[Saved] {outfile}')


# ========== 主流程 ==========

def main():
    datasets = {
        'E2': 'E2_pairs.csv',
        'H2': 'H2_pairs.csv',
    }

    for name, path in datasets.items():
        print(f'\n=== Processing {name} from {path} ===')
        df = load_pairs(path)
        reps = df['rep'].to_numpy()
        ts   = df['t'].to_numpy()
        vs   = df['Y'].to_numpy()
        n_paths = int(df['rep'].nunique())

        # 全局 LORO-CV（等权 kNN）选 k*
        k_star = global_loro_cv_k_equal(reps, ts, vs, sample_per_rep=25)
        print(f'[Result] {name}: global best k* = {k_star} (n={n_paths}, N={len(df)})')

        # 计算自适应等权 kNN 曲线
        t_grid, vbar = adaptive_knn_curve_equal(reps, ts, vs, k_star=k_star, grid_points=450)

        # 绘图
        outfile = f'Adaptive_knn_{name}.png'
        ds_title = 'E2' if name.upper().startswith('E2') else 'H2'
        plot_curve(t_grid, vbar, ds_title, outfile, n_paths=n_paths, k_star=k_star)

if __name__ == '__main__':
    main()
