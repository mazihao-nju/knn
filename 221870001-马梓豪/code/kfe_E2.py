# e2m1c_virtual_wait_cn.py
# -*- coding: utf-8 -*-
"""
用数值方法（RK4 积分 Kolmogorov 前向方程）计算 E2(t)/M/1/c 的“虚拟等待时间” v(t)，
并绘制到达时间 t 对 v(t) 的曲线。到达过程采用两相串联的 Erlang-2（E2(t)），
服务为指数（单服务台），容量 c（含在服者），满员丢弃（loss）。
相位规则（E2 串联系统）：
  - 处于相位1时，以速率 lambda(t) 进入相位2（不产生到达、人数不变）；
  - 处于相位2时，以速率 lambda(t) 发生“到达事件”：若 j<c 则 j->j+1 并回到相位1；若 j=c 则丢弃并回到相位1。
指数服务率为 mu，j>0 以速率 mu 使 j->j-1（相位不变）。
虚拟等待时间：当系统内人数为 j(<c) 时，新到达顾客期望等待 j/mu；对“未满”条件化后取期望。
"""

from __future__ import annotations
from typing import Callable, Union, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 保险：重置 matplotlib 全局设置，避免外部环境遗留字体配置影响
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# =====================================================
# 【在此改参数】—— 与之前 H2(t) 脚本保持一致的默认值
# =====================================================
T_END: float = 16.0   # 仿真时间终点
MU: float = 1.0       # 服务率 mu
C: int = 8            # 系统容量 c（含在服者；满员丢弃）

def LAMBDA(t: float) -> float:
    """E2(t) 两相的时间变率：lambda(t) = 2.0 + sin(0.5*t)（两相同速率）。"""
    return 2.0 + np.sin(0.5 * t)

STEPS: int = 2000              # RK4 时间步数（越大越平滑，计算更慢）
SAVE_PATH: str = "kfe_E2.png"  # 输出图片文件名

# ===========================
#        核心实现函数
# ===========================
RateLike = Union[float, Callable[[float], float]]

def _as_fn(x: RateLike) -> Callable[[float], float]:
    """把常数或可调用对象统一成 f(t)->非负数 的函数。"""
    if callable(x):
        return lambda t: max(0.0, float(x(t)))
    val = float(x)
    return lambda t: val

def build_Q_E2M1c(lam: RateLike, mu: float, c: int) -> Callable[[float], np.ndarray]:
    """
    构造时间相关的生成矩阵 Q(t)。状态为 (phase, j)：
    - phase∈{1,2}（实现中用 0/1 表示），j∈{0,...,c}。
    规则（E2 串联）：
      * 服务完成（j>0）：(phase, j) -> (phase, j-1)，速率 mu；
      * 相位推进/到达：
          - 若 phase=0（相位1）：以速率 lambda(t) 转到相位2，人数 j 保持不变；
          - 若 phase=1（相位2）：以速率 lambda(t) 发生到达并回到相位1：
                · 若 j<c：j->j+1；
                · 若 j=c：丢弃（j 保持 c）。
    """
    lamf = _as_fn(lam)

    def Q_of_t(t: float) -> np.ndarray:
        l = lamf(t)
        nstate = 2 * (c + 1)
        Q = np.zeros((nstate, nstate), dtype=float)

        for phase in (0, 1):  # 0: 相位1, 1: 相位2
            for j in range(c + 1):
                idx = phase * (c + 1) + j
                out = 0.0

                # 服务完成（相位不变）
                if j > 0:
                    Q[idx, phase * (c + 1) + (j - 1)] += mu
                    out += mu

                if phase == 0:
                    # 相位1：以速率 l 进入相位2，人数不变
                    Q[idx, 1 * (c + 1) + j] += l
                    out += l
                else:
                    # 相位2：以速率 l 发生到达并回到相位1
                    target_j = j + 1 if j < c else c
                    Q[idx, 0 * (c + 1) + target_j] += l
                    out += l

                # 对角元：负的总流出率
                Q[idx, idx] -= out

        return Q

    return Q_of_t

def rk4_integrate(Q_of_t: Callable[[float], np.ndarray],
                  p0: np.ndarray,
                  t_grid: np.ndarray) -> np.ndarray:
    """在给定时间网格上用 RK4 积分 dp/dt = Q(t)^T p(t)。"""
    nT = len(t_grid)
    nstate = p0.size
    P = np.zeros((nT, nstate), dtype=float)
    P[0] = p0

    def f(t, p):
        return Q_of_t(t).T @ p

    for k in range(nT - 1):
        t = t_grid[k]
        h = t_grid[k + 1] - t
        p = P[k]

        k1 = f(t, p)
        k2 = f(t + 0.5 * h, p + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, p + 0.5 * h * k2)
        k4 = f(t + h,       p + h * k3)
        p_next = p + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # 数值卫生：裁剪负值并重归一化（防小幅负数和漂移）
        p_next = np.maximum(p_next, 0.0)
        s = p_next.sum()
        if s <= 0.0:
            p_next = p.copy()
            s = p_next.sum()
        if s != 1.0:
            p_next /= s

        P[k + 1] = p_next

    return P

# === CHANGED: E2 到达仅发生在相位2；按“到达视角”在相位2且 j<c 上条件化 ===
def virtual_waiting_time_arrival_E2(P: np.ndarray, c: int, mu: float) -> np.ndarray:
    """
    对每个 t：
      只取相位2的边缘分布 P(phase=2, J=j)，在 j<c 上条件化并计算 E[j/μ]。
    这是 E2 串联系统下的 Palm 加权（λ(t) 抵消）。
    """
    nT = P.shape[0]
    v = np.zeros(nT, dtype=float)

    j = np.arange(0, c + 1, dtype=float)
    w = (j / float(mu))[:c]  # 只在 j<c

    for k in range(nT):
        p2 = P[k, c+1:2*(c+1)]  # 相位2：j=0..c
        den = np.sum(p2[:c])
        num = np.dot(w, p2[:c])
        v[k] = (num / den) if den > 1e-14 else np.nan

    return v


def solve_e2m1c(t_end: float,
                mu: float,
                c: int,
                lam: RateLike,
                steps: int,
                p0_phase: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """主求解器：返回 (t_grid, v(t))。p0_phase 指定 t=0 时系统空且相位=1 或 2。"""
    t_grid = np.linspace(0.0, float(t_end), int(steps) + 1)
    Q_of_t = build_Q_E2M1c(lam, float(mu), int(c))

    # 初始分布：系统空、给定初始相位
    nstate = 2 * (c + 1)
    p0 = np.zeros(nstate, dtype=float)
    if p0_phase == 1:
        p0[0] = 1.0          # (phase=1, j=0)
    else:
        p0[c + 1 + 0] = 1.0  # (phase=2, j=0)

    P = rk4_integrate(Q_of_t, p0, t_grid)
    v = virtual_waiting_time_arrival_E2(P, c, mu)
    return t_grid, v

def plot_vt(t: np.ndarray, v: np.ndarray, c: int, mu: float,
            lam: RateLike,
            save_path: str | None = None) -> None:
    """绘图（英文标签）。save_path 为空则直接 show。"""
    lbl_lam = "lambda(t)" if callable(lam) else str(lam)

    plt.figure(figsize=(8.4, 5.0))
    plt.plot(t, v, label="v(t)", linewidth=2)
    plt.xlabel("Arrival time t")
    plt.ylabel("Virtual waiting time v(t)")
    plt.title(f"E2(t)/M/1/{c} by kfe")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()
    plt.close()

# ===========================
#           主程序
# ===========================
if __name__ == "__main__":
    # 直接使用上面定义的参数与时间函数
    t_grid, v = solve_e2m1c(
        t_end=T_END,
        mu=MU,
        c=C,
        lam=LAMBDA,    # 若希望常数速率，可改成数字，如 2.0
        steps=STEPS,
        p0_phase=1     # 初始相位可改 1 或 2
    )

    plot_vt(t_grid, v, c=C, mu=MU, lam=LAMBDA, save_path=SAVE_PATH)
    print(f"Saved figure to: {SAVE_PATH}")
