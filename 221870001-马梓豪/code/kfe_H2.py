# h2m1c_virtual_wait_cn.py
# -*- coding: utf-8 -*-
"""
用数值方法（RK4 积分 Kolmogorov 前向方程）计算 H2(t)/M/1/c 的“虚拟等待时间” v(t)，
并绘制到达时间 t 对 v(t) 的曲线。到达过程采用两相超指数 H2(t)，服务为指数（单服务台），
容量 c（含在服者），满员丢弃。相位在每次“到达事件”（即便丢弃）后独立重置。
"""

from __future__ import annotations
from typing import Callable, Union, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 保险：重置 matplotlib 全局设置，避免外部环境遗留字体配置影响
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# =====================================================
# 【在此改参数】—— 与你的 H2(t) 配置一致的默认值
# =====================================================
T_END: float = 16.0   # 仿真时间终点
MU: float = 1.0       # 服务率 μ
C: int = 8            # 系统容量 c（含在服者；满员丢弃）
P_MIX: float = 0.6    # 相位重置到组件1的概率 p（常数 0.6）

# H2(t) 两个相位的时间变率函数（与你的脚本一致）
def LAMBDA1(t: float) -> float:
    """组件1危险率 λ1(t) = 2.0 + 0.6*sin(0.5*t)"""
    return 2.0 + 0.6 * np.sin(0.5 * t)

def LAMBDA2(t: float) -> float:
    """组件2危险率 λ2(t) = 0.8 + 0.4*cos(0.3*t)"""
    return 0.8 + 0.4 * np.cos(0.3 * t)

STEPS: int = 2000     # RK4 时间步数（越大越平滑，计算更慢）
SAVE_PATH: str = "kfe_H2.png"  # 输出图片文件名

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

def build_Q_H2M1c(lambda1: RateLike, lambda2: RateLike, p_mix: RateLike,
                  mu: float, c: int) -> Callable[[float], np.ndarray]:
    """
    构造时间相关的生成矩阵 Q(t)。状态为 (phase, j)：
    - phase∈{1,2}（实现中用0/1表示），j∈{0,...,c}。
    规则：
      * 服务完成（j>0）：(phase, j) -> (phase, j-1)，速率 μ；
      * 到达（速率 = 当前相位的 λ_phase(t)）：
          - 若 j<c：j->j+1，且相位重置到1或2（概率 p 与 1-p）；
          - 若 j=c：丢弃（j 保持 c），但相位同样按 p 与 1-p 重置；
    """
    lambda1f = _as_fn(lambda1)
    lambda2f = _as_fn(lambda2)
    pf       = _as_fn(p_mix)

    def Q_of_t(t: float) -> np.ndarray:
        l1 = lambda1f(t)
        l2 = lambda2f(t)
        p  = pf(t)
        p  = 0.0 if p < 0.0 else (1.0 if p > 1.0 else p)  # 概率裁剪

        nstate = 2 * (c + 1)
        Q = np.zeros((nstate, nstate), dtype=float)

        # phase: 0 -> λ1, 1 -> λ2
        for phase in (0, 1):
            lam = l1 if phase == 0 else l2
            for j in range(c + 1):
                idx = phase * (c + 1) + j
                out = 0.0

                # 服务完成
                if j > 0:
                    Q[idx, phase * (c + 1) + (j - 1)] += mu
                    out += mu

                # 到达事件（即便满员也发生，并重置相位）
                target_j = j + 1 if j < c else c
                Q[idx, 0 * (c + 1) + target_j] += p * lam         # 重置到相位1
                Q[idx, 1 * (c + 1) + target_j] += (1.0 - p) * lam # 重置到相位2
                out += lam

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

# === CHANGED: 用“到达视角（Palm 加权）”计算 v(t) ===
def virtual_waiting_time_arrival(P: np.ndarray, c: int, mu: float,
                                 t_grid: np.ndarray,
                                 lambda1: RateLike, lambda2: RateLike) -> np.ndarray:
    """
    v_arr(t) = [ λ1(t) * Σ_{j<c} (j/μ)·P(phase=1, J=j) + λ2(t) * Σ_{j<c} (j/μ)·P(phase=2, J=j) ]
               -------------------------------------------------------------------------------
               [ λ1(t) * Σ_{j<c}        P(phase=1, J=j) + λ2(t) * Σ_{j<c}        P(phase=2, J=j) ]

    只在 j<c（被接纳的状态）上做到达强度加权的条件化；与仿真记录 (t, V(t)) 的“到达视角”一致。
    """
    nT = P.shape[0]
    v = np.zeros(nT, dtype=float)

    # j/μ 权重，仅对 j<c
    j = np.arange(0, c + 1, dtype=float)
    w = (j / float(mu))[:c]

    l1f = _as_fn(lambda1)
    l2f = _as_fn(lambda2)

    for k in range(nT):
        # 状态展开：phase1 的 j=0..c 在前，phase2 的 j=0..c 在后（与你原脚本一致）
        p1 = P[k, :c+1]
        p2 = P[k, c+1:2*(c+1)]

        l1 = float(l1f(t_grid[k]))
        l2 = float(l2f(t_grid[k]))

        num = l1 * np.dot(w, p1[:c]) + l2 * np.dot(w, p2[:c])
        den = l1 * np.sum(p1[:c])   + l2 * np.sum(p2[:c])
        v[k] = (num / den) if den > 1e-14 else np.nan

    return v

def solve_h2m1c(t_end: float,
                mu: float,
                c: int,
                lambda1: RateLike,
                lambda2: RateLike,
                p_mix: RateLike,
                steps: int,
                p0_phase: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """主求解器：返回 (t_grid, v(t))。p0_phase 指定 t=0 时系统空且相位=1 或 2。"""
    t_grid = np.linspace(0.0, float(t_end), int(steps) + 1)
    Q_of_t = build_Q_H2M1c(lambda1, lambda2, p_mix, float(mu), int(c))

    # 初始分布：系统空、给定初始相位
    nstate = 2 * (c + 1)
    p0 = np.zeros(nstate, dtype=float)
    if p0_phase == 1:
        p0[0] = 1.0          # (phase=1, j=0)
    else:
        p0[c + 1 + 0] = 1.0  # (phase=2, j=0)

    P = rk4_integrate(Q_of_t, p0, t_grid)
    v = virtual_waiting_time_arrival(P, c, mu, t_grid, lambda1, lambda2)
    return t_grid, v

def plot_vt(t: np.ndarray, v: np.ndarray, c: int, mu: float,
            lambda1: RateLike, lambda2: RateLike, p_mix: RateLike,
            save_path: str | None = None) -> None:
    """绘图（英文标签）。save_path 为空则直接 show。"""
    lbl_l1 = "lambda1(t)" if callable(lambda1) else str(lambda1)
    lbl_l2 = "lambda2(t)" if callable(lambda2) else str(lambda2)
    lbl_p  = "p(t)"       if callable(p_mix)  else str(p_mix)

    plt.figure(figsize=(8.4, 5.0))
    plt.plot(t, v, label="v(t)", linewidth=2)
    plt.xlabel("Arrival time t")
    plt.ylabel("Virtual waiting time v(t)")
    plt.title(f"H2(t)/M/1/{c} by kfe")
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
    t_grid, v = solve_h2m1c(
        t_end=T_END,
        mu=MU,
        c=C,
        lambda1=LAMBDA1,
        lambda2=LAMBDA2,
        p_mix=P_MIX,   # 也可以改成函数 p(t)，如：lambda t: 0.6 + 0.2*np.sin(2*np.pi*t/15)
        steps=STEPS,
        p0_phase=1     # 初始相位可改 1 或 2
    )
    plot_vt(t_grid, v, c=C, mu=MU, lambda1=LAMBDA1, lambda2=LAMBDA2, p_mix=P_MIX, save_path=SAVE_PATH)
    print(f"Saved figure to: {SAVE_PATH}")
