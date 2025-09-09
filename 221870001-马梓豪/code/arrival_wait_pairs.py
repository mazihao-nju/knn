import numpy as np
import math
from typing import Callable, List, Tuple, Optional

# =======================================================
# (t, V(t)) at arrival epochs (admitted only)
# E2(t)/M/1/c and H2(t)/M/1/c with nonstationary arrivals
# =======================================================
#
# Event-driven simulation:
# - Single server, exponential service with rate mu
# - Finite capacity c (including any in service). Arrivals when j==c are blocked.
# - For each *admitted* arrival at time t, record actual waiting time V(t):
#     V(t) = remaining service time of job in service at t  + sum of service times
#            of all queued jobs ahead of the arriving job.
#
# Nonstationary arrivals:
# - E2(t): interarrival = sum of 2 nonhomogeneous exponential stage completions.
# - H2(t): per arrival choose branch 1 w.p. p (rate λ1(t)) else branch 2 (rate λ2(t)).
#
# Nonhomogeneous exponential stage completion is generated via Ogata's thinning;
# you must supply an envelope maximum lam_max >= sup_t λ(t) for each rate function.

# ---------------- Utilities ----------------

def exp_rv(rate: float, rng: np.random.Generator) -> float:
    return rng.exponential(1.0 / rate) if rate > 0 else math.inf

def thinning_nhexp_next_time(t0: float, lam_t: Callable[[float], float],
                             lam_max: float, rng: np.random.Generator) -> float:
    """Ogata thinning for next event time of NH exp clock λ(t), starting at t0."""
    t = t0
    while True:
        t += rng.exponential(1.0 / lam_max)
        if rng.uniform() <= lam_t(t) / lam_max:
            return t

# ---------------- E2(t) arrivals ----------------

def next_arrival_time_E2(t0: float, lam_t: Callable[[float], float],
                         lam_max: float, rng: np.random.Generator) -> float:
    t1 = thinning_nhexp_next_time(t0, lam_t, lam_max, rng)
    t2 = thinning_nhexp_next_time(t1, lam_t, lam_max, rng)
    return t2

# ---------------- H2(t) arrivals ----------------

def next_arrival_time_H2(t0: float, p: float,
                         lam1_t: Callable[[float], float], lam1_max: float,
                         lam2_t: Callable[[float], float], lam2_max: float,
                         rng: np.random.Generator) -> float:
    if rng.uniform() < p:
        return thinning_nhexp_next_time(t0, lam1_t, lam1_max, rng)
    else:
        return thinning_nhexp_next_time(t0, lam2_t, lam2_max, rng)

# ---------------- Core engine (single server) ----------------

def generate_pairs_E2(t_end: float, mu: float, c: int,
                      lam_t: Callable[[float], float], lam_max: float,
                      seed: int = 0) -> List[Tuple[float, float]]:
    """
    Generate (t, V(t)) for each *admitted* arrival up to time t_end in E2(t)/M/1/c.
    Returns list of (arrival_time, waiting_time).
    """
    rng = np.random.default_rng(seed)

    t = 0.0
    j = 0  # number in system
    pairs: List[Tuple[float, float]] = []

    # service process state
    next_dep = math.inf              # absolute time of next service completion (if j>0)
    queue_times: list[float] = []    # service times of queued jobs (not in service)

    # first arrival
    next_arr = next_arrival_time_E2(t, lam_t, lam_max, rng)

    while True:
        # advance to the next event before t_end
        t_next = min(next_arr, next_dep)
        if t_next > t_end:
            break

        if next_dep <= next_arr:
            # --- Service completion event ---
            t = next_dep
            j -= 1
            if j == 0:
                next_dep = math.inf
            else:
                # start service for head-of-line job
                st = queue_times.pop(0)  # that job's full service time
                next_dep = t + st
        else:
            # --- Arrival event ---
            t = next_arr

            # process all departures strictly before the arrival time
            # (handled by event ordering; here next_dep > t)
            if j < c:
                # compute V(t): actual waiting time for this arrival
                if j == 0:
                    V = 0.0
                else:
                    rem_current = max(0.0, next_dep - t)  # remaining time of job in service
                    V = rem_current + sum(queue_times)

                # record pair
                pairs.append((t, V))

                # admit: update system
                j += 1
                # sample service time for this arriving job
                st_new = exp_rv(mu, rng)

                if j == 1:
                    # server was idle -> this job starts service immediately
                    next_dep = t + st_new
                else:
                    # enqueue
                    queue_times.append(st_new)
            # else: blocked; we skip recording since customer not admitted

            # schedule next arrival
            next_arr = next_arrival_time_E2(t, lam_t, lam_max, rng)

    return pairs


def generate_pairs_H2(t_end: float, mu: float, c: int, p: float,
                      lam1_t: Callable[[float], float], lam1_max: float,
                      lam2_t: Callable[[float], float], lam2_max: float,
                      seed: int = 0) -> List[Tuple[float, float]]:
    """
    Generate (t, V(t)) for each *admitted* arrival up to t_end in H2(t)/M/1/c.
    Returns list of (arrival_time, waiting_time).
    """
    rng = np.random.default_rng(seed)

    t = 0.0
    j = 0
    pairs: List[Tuple[float, float]] = []

    next_dep = math.inf
    queue_times: list[float] = []

    next_arr = next_arrival_time_H2(t, p, lam1_t, lam1_max, lam2_t, lam2_max, rng)

    while True:
        t_next = min(next_arr, next_dep)
        if t_next > t_end:
            break

        if next_dep <= next_arr:
            # service completion
            t = next_dep
            j -= 1
            if j == 0:
                next_dep = math.inf
            else:
                st = queue_times.pop(0)
                next_dep = t + st
        else:
            # arrival
            t = next_arr
            if j < c:
                if j == 0:
                    V = 0.0
                else:
                    rem_current = max(0.0, next_dep - t)
                    V = rem_current + sum(queue_times)
                pairs.append((t, V))

                j += 1
                st_new = exp_rv(mu, rng)
                if j == 1:
                    next_dep = t + st_new
                else:
                    queue_times.append(st_new)
            # schedule next arrival
            next_arr = next_arrival_time_H2(t, p, lam1_t, lam1_max, lam2_t, lam2_max, rng)

    return pairs