import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# ---- Math text 设置，避免 Unicode 数学符号的字体告警 ----
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = ['STIXGeneral', 'DejaVu Sans']

# ---------- 读入与列名规整（兼容 i, t, V（t）） ----------
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
        next(k for k,v in norm_map.items() if v == 'i'): 'rep',
        next(k for k,v in norm_map.items() if v == 't'): 't',
        next(k for k,v in norm_map.items() if v in ('v(t)','vt')): 'Y',
    })
    df['rep'] = pd.to_numeric(df['rep'], errors='coerce')
    df['t']   = pd.to_numeric(df['t'],   errors='coerce')
    df['Y']   = pd.to_numeric(df['Y'],   errors='coerce')
    df = df.dropna(subset=['rep','t','Y']).copy()
    df['rep'] = df['rep'].astype(int)
    return df.sort_values(['rep','t']).reset_index(drop=True)

# ---------- LORO-CV 选 k ----------
def lorocv_best_k(df: pd.DataFrame, k_max: int = 1000):
    reps = df['rep'].unique()
    if len(reps) < 2:
        raise ValueError("LORO requires at least 2 replications.")
    N  = len(df)
    Mj = df.groupby('rep').size().to_dict()
    min_train = min(N - Mj[r] for r in reps)
    print(min_train)
    kU = int(min(k_max, min_train))
    print(kU)
    if kU < 1:
        raise ValueError("Insufficient training points in LORO folds (kU < 1).")
    sse = np.zeros(kU + 1, dtype=float)
    total_tests = 0

    for r in reps:
        test  = df[df['rep'] == r]
        train = df[df['rep'] != r]
        t_te, y_te = test['t'].to_numpy(), test['Y'].to_numpy()
        t_tr, y_tr = train['t'].to_numpy(), train['Y'].to_numpy()

        D = np.abs(t_te[:, None] - t_tr[None, :])                # (m_test, m_train)
        idx_part = np.argpartition(D, kth=kU-1, axis=1)[:, :kU]  # 先取前kU个
        D_part   = np.take_along_axis(D, idx_part, axis=1)
        order    = np.argsort(D_part, axis=1)                    # 再排序
        nn_idx   = np.take_along_axis(idx_part, order, axis=1)

        Y_nn = y_tr[nn_idx]                 # (m_test, kU)，从近到远
        cums = np.cumsum(Y_nn, axis=1)
        ks   = np.arange(1, kU+1)
        preds = cums / ks                    # \bar V = (前k项)/k

        errs = (preds - y_te[:, None])**2
        sse[1:] += np.sum(errs, axis=0)
        total_tests += len(y_te)

    emse = sse[1:] / total_tests
    k_star = int(np.argmin(emse) + 1)
    return k_star, emse

# ---------- 画图：EMSE ----------
def plot_emse(emse, dataset_name, outfile, k_star, n_paths):
    """
    修改点：
    1) 取消箭头标注；
    2) 图例“EMSE(k)”放在与右上角 n=、k*= 同一行（顶部一行的中间）；
    3) 右上角统一标注 n= 与 k*= 。
    """
    ks = np.arange(1, len(emse) + 1)
    fig, ax = plt.subplots()
    ax.plot(ks, emse, marker='o', linewidth=1.5, label='EMSE(k)')
    ax.set_xlabel('k (number of neighbors)')
    ax.set_ylabel('EMSE(k)')
    ax.set_title(f'EMSE vs k ({dataset_name})')
    ax.grid(True, linestyle='--', alpha=0.4)

    # —— 图例放“同一行的中间”：使用 upper center + bbox_to_anchor 设到顶行 ——
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 0.985),     # x=0.5 居中，y≈1 的顶行
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        handlelength=1.6
    )

    # —— 右上角标注 n 与 k*（与上面同一行的 y 坐标）——
    ax.text(
        0.985, 0.985, f'n={n_paths}, k*={k_star}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85)
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f'[Saved] {outfile}')


# ---------- KNN 估计 v(t) ----------
def knn_vbar_curve(df: pd.DataFrame, k: int, grid_points: int = 300):
    t_tr = df['t'].to_numpy()
    y_tr = df['Y'].to_numpy()
    t_grid = np.linspace(float(t_tr.min()), float(t_tr.max()), grid_points)
    D = np.abs(t_grid[:, None] - t_tr[None, :])           # (G, N)
    idx_k = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    vbar  = y_tr[idx_k].mean(axis=1)
    return t_grid, vbar

def plot_vbar_curve(t_grid, vbar, k_star, dataset_name, outfile, n_paths):
    """
    修改点：
    1) 取消任何箭头；
    2) 右下角原来的标注改为右上角，与 EMSE 图保持一致。
    """
    cap = 8  # 若你的容量不是 8，可根据实际传参或改成变量
    name_upper = str(dataset_name).upper()
    if name_upper.startswith('E2'):
        title_prefix = 'E2(t)/M/1'
    elif name_upper.startswith('H2'):
        title_prefix = 'H2(t)/M/1'
    else:
        title_prefix = f'{dataset_name}(t)/M/1'

    fig, ax = plt.subplots()
    ax.plot(t_grid, vbar, linewidth=2, label='v(t)')
    ax.set_xlabel('Arrival time t')
    ax.set_ylabel('Virtual waiting time v(t)')
    ax.set_title(f'{title_prefix}/{cap} by knn')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best')

    # 右上角标注 n 与 k*
    ax.text(
        0.98, 0.98, f'n={n_paths}, k*={k_star}',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.85)
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f'[Saved] {outfile}')
# ---------- 主流程：同时处理 E2 与 H2 ----------
if __name__ == '__main__':
    datasets = {
        'E2': 'E2_pairs.csv',
        'H2': 'H2_pairs.csv',
    }

    for name, path in datasets.items():
        print(f'\n=== Processing {name} from {path} ===')
        df = load_pairs(path)
        n_paths = df['rep'].nunique()  # 样本路径数 n
        k_star, emse = lorocv_best_k(df, k_max=300)
        print(f'[Result] {name}: optimal k* = {k_star} (n={n_paths})')

        # EMSE 图（带箭头与文本框）
        plot_emse(emse, name, outfile=f'LOROCV_for_select_k_{name}.png',
                  k_star=k_star, n_paths=n_paths)

        # v(t) 曲线图（右下角文本框）
        t_grid, vbar = knn_vbar_curve(df, k_star, grid_points=400)
        plot_vbar_curve(t_grid, vbar, k_star, name,
                        outfile=f'knn_{name}.png', n_paths=n_paths)
