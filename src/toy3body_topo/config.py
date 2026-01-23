from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


@dataclass(frozen=True)
class Units:
    # Code units, default G=1.
    G: float = 1.0


@dataclass(frozen=True)
class PhysicalParams:
    # masses in code units
    m1: float
    m2: float
    m3: float

    # optional relativistic strength parameter epsilon = GM_bin/(a0 c^2)
    epsilon: float = 0.0

    # PN warning / merger-cut in units of r_g = G (m_i+m_j) / c^2  (pair-dependent)
    k_warn: float = 50.0
    k_merge: float = 10.0

    @property
    def Mbin(self) -> float:
        return float(self.m1 + self.m2)

    @property
    def c(self) -> Optional[float]:
        if self.epsilon <= 0.0:
            return None
        return float(1.0 / np.sqrt(self.epsilon))


@dataclass(frozen=True)
class SimParams:
    # solve_ivp options
    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12
    max_step: float = float('inf')

    # time control
    t_max: float = 3000.0
    dt_chunk: float = 50.0

    # termination heuristics
    R_int: float = 20.0
    R_end: float = 200.0
    hier_ratio: float = 5.0

    # storage control
    max_store_points: int = 20000
    decimate_to: int = 5000

    # --- numerical safety rails (useful for chaotic scattering grids) ---
    # Collision cut used even in pure Newton runs to prevent pathological
    # near-collisions from stalling the integrator. Set <= 0 to disable.
    r_coll: float = 1e-5

    # Hard wall-clock limit for a single trajectory integration (seconds).
    # Set <= 0 to disable.
    max_walltime_sec: float = 0.0

    # Minimum required advance in time per chunk; if the solver fails to
    # advance, we bail out instead of spinning forever.
    min_t_advance: float = 1e-12


@dataclass(frozen=True)
class ICParams:
    # binary
    a0: float = 1.0
    e0: float = 0.0
    phase: float = 0.0

    # single
    v_inf: float = 1.0
    b: float = 0.0
    R0: float = 100.0

    # global rotation in plane
    theta: float = 0.0


@dataclass(frozen=True)
class LoopParams:
    # close return threshold in shape space (log-dist coordinates)
    eps_close: float = 1e-2
    # minimum loop duration (in time units)
    min_dt: float = 10.0
    # KDTree nearest-neighbor count for candidate close returns (speed/robustness)
    k_nn: int = 16
    # base number of loops to extract in a typical run
    max_loops_base: int = 1
    # if dwell time is long, extract more loops
    max_loops_long: int = 8
    t_long: float = 300.0
    # resample each loop to fixed number of points (controls linking accuracy/cost)
    resample_n: int = 256

    # linking quality gates
    link_round_tol: float = 0.1
    link_min_sep: float = 1e-3


@dataclass(frozen=True)
class OutputParams:
    out_dir: str = "out_runs"
    store_raw: bool = False
    store_raw_every: int = 0  # store every N-th run (0 disables)
    compress_npz: bool = True


@dataclass(frozen=True)
class ParallelParams:
    workers: int = 0  # 0 => use os.cpu_count()
    chunksize: int = 1


@dataclass(frozen=True)
class EnsembleConfig:
    # parameter sweeps
    mu_list: List[float]
    nu_list: List[float]
    epsilon_list: List[float]
    v_inf_list: List[float]
    b_max: float
    n_per_setting: int
    seed0: int = 1234

    # IC controls
    a0: float = 1.0
    R0: float = 100.0
    random_phase: bool = True
    random_theta: bool = True

    sim: SimParams = SimParams()
    loop: LoopParams = LoopParams()
    output: OutputParams = OutputParams()
    parallel: ParallelParams = ParallelParams()


def _dataclass_from_dict(cls, d: Dict[str, Any]):
    # allow passing dict for nested dataclasses
    kwargs = {}
    for f in cls.__dataclass_fields__.values():  # type: ignore
        if f.name not in d:
            continue
        val = d[f.name]
        if f.name == 'max_step' and val is None:
            val = float('inf')
        if hasattr(f.type, "__dataclass_fields__") and isinstance(val, dict):
            kwargs[f.name] = _dataclass_from_dict(f.type, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)  # type: ignore


def load_ensemble_config(path: str) -> EnsembleConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # build nested dataclasses
    if "sim" in d:
        d["sim"] = _dataclass_from_dict(SimParams, d["sim"])
    if "loop" in d:
        d["loop"] = _dataclass_from_dict(LoopParams, d["loop"])
    if "output" in d:
        d["output"] = _dataclass_from_dict(OutputParams, d["output"])
    if "parallel" in d:
        d["parallel"] = _dataclass_from_dict(ParallelParams, d["parallel"])
    return _dataclass_from_dict(EnsembleConfig, d)


def phys_from_mu_nu(mu: float, nu: float, epsilon: float, k_warn: float = 50.0, k_merge: float = 10.0) -> PhysicalParams:
    # normalized M_bin = 1
    m1 = 1.0/(1.0+mu)
    m2 = mu/(1.0+mu)
    m3 = nu
    return PhysicalParams(m1=m1, m2=m2, m3=m3, epsilon=epsilon, k_warn=k_warn, k_merge=k_merge)


def to_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(obj), f, indent=2)
