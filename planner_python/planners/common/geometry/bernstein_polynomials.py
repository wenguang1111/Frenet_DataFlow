"""
Bernstein polynomials for Frenet planning.

Two concrete polynomial types share a common base class:

    BernsteinPolynomialBase
        |
        +-- QuinticBernstein  (degree 5, fully-constrained endpoints)
        |       boundary: s0, v0, a0, sf, vf, af
        |
        +-- QuarticBernstein  (degree 4, free terminal position)
                boundary: s0, v0, a0,     vf, af
                used for velocity-keeping trajectories

Both classes expose the same evaluation API:
    evaluate(t)                  -> position
    calc_first_derivative(t)     -> velocity
    calc_second_derivative(t)    -> acceleration
    calc_third_derivative(t)     -> jerk
    state(t)                     -> (s, v, a, j)

Subclasses only need to implement _compute_control_points() to provide
the position control points; the base class handles everything else
(derivative control points, De Casteljau evaluation, precomputed inv_T).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Base class: shared evaluation machinery
# ---------------------------------------------------------------------------
class BernsteinPolynomialBase(ABC):
    """
    Shared machinery for Bernstein-form polynomial evaluation in Frenet
    planning. Subclasses provide the position control points; this base
    handles derivative chains, hot-path evaluation, and dtype casting.

    Internal storage (all in self.dtype):
        self.control_points     degree n,   n+1 points  (position)
        self.cp_vel             degree n-1, n   points  (physical velocity)
        self.cp_acc             degree n-2, n-1 points  (physical acceleration)
        self.cp_jerk            degree n-3, n-2 points  (physical jerk)
        self.T, self.inv_T      scalar constants
    """

    def __init__(self, T: float, dtype=np.float64):
        if T <= 0.0:
            raise ValueError("T must be positive")
        self.dtype = dtype

        # Constants in float64; cast to dtype at the end.
        T_     = np.float64(T)
        inv_T_ = 1.0 / T_

        # Subclass supplies the position control points (float64 array).
        cp_pos = self._compute_control_points(T_)
        n = len(cp_pos) - 1   # polynomial degree

        # Derivative control points, physical domain.
        # For degree-n Bernstein: d/dt p(t) has degree n-1 with
        # control points (n/T) * diff(cp_pos).
        cp_vel  = (float(n)       * inv_T_) * np.diff(cp_pos)
        cp_acc  = (float(n - 1)   * inv_T_) * np.diff(cp_vel)
        cp_jerk = (float(n - 2)   * inv_T_) * np.diff(cp_acc)

        # Single cast to target dtype.
        self.control_points = cp_pos .astype(dtype)
        self.cp_vel         = cp_vel .astype(dtype)
        self.cp_acc         = cp_acc .astype(dtype)
        self.cp_jerk        = cp_jerk.astype(dtype)

        self.T     = dtype(T_)
        self.inv_T = dtype(inv_T_)

    # ------------------------------------------------------------------
    @abstractmethod
    def _compute_control_points(self, T: np.float64) -> np.ndarray:
        """
        Compute position control points from boundary conditions, in float64.
        Called once during __init__. The base class takes care of casting,
        derivative chains, and inv_T precomputation.

        Returns:
            np.ndarray of shape (n+1,) dtype=float64.
        """
        ...

    # ------------------------------------------------------------------
    # Evaluation API (shared by all subclasses)
    # ------------------------------------------------------------------
    def evaluate(self, t: float) -> float:
        """Position s(t)."""
        tau = self.dtype(t) * self.inv_T
        return self._de_casteljau(self.control_points, tau)

    def calc_first_derivative(self, t: float) -> float:
        """Velocity ds/dt."""
        tau = self.dtype(t) * self.inv_T
        return self._de_casteljau(self.cp_vel, tau)

    def calc_second_derivative(self, t: float) -> float:
        """Acceleration d2s/dt2."""
        tau = self.dtype(t) * self.inv_T
        return self._de_casteljau(self.cp_acc, tau)

    def calc_third_derivative(self, t: float) -> float:
        """Jerk d3s/dt3."""
        tau = self.dtype(t) * self.inv_T
        return self._de_casteljau(self.cp_jerk, tau)

    def state(self, t: float) -> tuple[float, float, float, float]:
        """(s, v, a, j) at time t, sharing the tau computation."""
        tau = self.dtype(t) * self.inv_T
        return (
            self._de_casteljau(self.control_points, tau),
            self._de_casteljau(self.cp_vel,         tau),
            self._de_casteljau(self.cp_acc,         tau),
            self._de_casteljau(self.cp_jerk,        tau),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _de_casteljau(ctrl: np.ndarray, tau) -> float:
        """
        De Casteljau recursion. Each inner step is one FMA:
            b[k] = b[k] + tau * (b[k+1] - b[k])
        """
        dtype = ctrl.dtype
        tau = dtype.type(tau)
        b = ctrl.copy()
        n = len(b)
        for r in range(1, n):
            for k in range(n - r):
                b[k] = b[k] + tau * (b[k + 1] - b[k])
        return b[0]


# ---------------------------------------------------------------------------
# Quintic (degree 5): fully-constrained endpoints
# ---------------------------------------------------------------------------
class QuinticBernstein(BernsteinPolynomialBase):
    """
    Quintic polynomial in Bernstein form over tau = t/T in [0, 1].

    Boundary conditions (6 total):
        s(0)  = s0,  s'(0)  = v0,  s''(0)  = a0
        s(T)  = sf,  s'(T)  = vf,  s''(T)  = af

    Typical use: lane-change, merging, any maneuver with a specified
    terminal position.
    """

    def __init__(
        self,
        s0: float, v0: float, a0: float,
        sf: float, vf: float, af: float,
        T: float,
        dtype=np.float64,
    ):
        # Stash boundary conditions so _compute_control_points can use them.
        self._bc = (s0, v0, a0, sf, vf, af)
        super().__init__(T, dtype=dtype)

    def _compute_control_points(self, T: np.float64) -> np.ndarray:
        s0, v0, a0, sf, vf, af = self._bc
        s0_, v0_, a0_ = map(np.float64, (s0, v0, a0))
        sf_, vf_, af_ = map(np.float64, (sf, vf, af))
        T2 = T * T

        # Closed-form control points from boundary conditions.
        c0 = s0_
        c5 = sf_
        c1 = s0_ + v0_ * T / 5.0
        c4 = sf_ - vf_ * T / 5.0
        c2 = s0_ + 2.0 * v0_ * T / 5.0 + a0_ * T2 / 20.0
        c3 = sf_ - 2.0 * vf_ * T / 5.0 + af_ * T2 / 20.0
        return np.array([c0, c1, c2, c3, c4, c5])


# ---------------------------------------------------------------------------
# Quartic (degree 4): free terminal position
# ---------------------------------------------------------------------------
class QuarticBernstein(BernsteinPolynomialBase):
    """
    Quartic polynomial in Bernstein form over tau = t/T in [0, 1].

    Boundary conditions (5 total):
        s(0)  = s0,  s'(0)  = v0,  s''(0)  = a0
                     s'(T)  = vf,  s''(T)  = af
        (s(T) is unconstrained.)

    Typical use: velocity-keeping, cruising trajectories where only
    the target speed matters, not the exact terminal position.
    """

    def __init__(
        self,
        s0: float, v0: float, a0: float,
        vf: float, af: float,
        T: float,
        dtype=np.float64,
    ):
        self._bc = (s0, v0, a0, vf, af)
        super().__init__(T, dtype=dtype)

    def _compute_control_points(self, T: np.float64) -> np.ndarray:
        s0, v0, a0, vf, af = self._bc
        s0_, v0_, a0_ = map(np.float64, (s0, v0, a0))
        vf_, af_      = map(np.float64, (vf, af))
        T2 = T * T

        # Closed-form control points from boundary conditions.
        # c0, c1, c2 from start conditions; c3, c4 from terminal velocity
        # and acceleration (both expressed relative to c2).
        c0 = s0_
        c1 = s0_ + v0_ * T / 4.0
        c2 = s0_ + v0_ * T / 2.0 + a0_ * T2 / 12.0
        c3 = c2  + vf_ * T / 4.0 - af_ * T2 / 12.0
        c4 = c2  + vf_ * T / 2.0 - af_ * T2 / 12.0
        return np.array([c0, c1, c2, c3, c4])


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("QuinticBernstein: lane-change scenario")
    print("=" * 70)
    quint = QuinticBernstein(
        s0=0.0, v0=10.0, a0=0.0,
        sf=50.0, vf=20.0, af=0.0,
        T=3.0,
    )
    print(f"  Position ctrl pts    : {quint.control_points}")
    print(f"  Velocity ctrl pts    : {quint.cp_vel}")
    print(f"  Acceleration ctrl pts: {quint.cp_acc}")
    print(f"  Jerk ctrl pts        : {quint.cp_jerk}")
    print()
    print(f"  Boundary t=0 : s={quint.evaluate(0):.4f}  "
          f"v={quint.calc_first_derivative(0):.4f}  "
          f"a={quint.calc_second_derivative(0):.4f}")
    print(f"  Boundary t=T : s={quint.evaluate(3):.4f}  "
          f"v={quint.calc_first_derivative(3):.4f}  "
          f"a={quint.calc_second_derivative(3):.4f}")
    s, v, a, j = quint.state(2.1)
    print(f"  State at t=2.1: s={s:.6f}  v={v:.6f}  a={a:.6f}  j={j:.6f}")

    print()
    print("=" * 70)
    print("QuarticBernstein: velocity-keeping scenario")
    print("=" * 70)
    quart = QuarticBernstein(
        s0=0.0, v0=10.0, a0=0.0,
        vf=15.0, af=0.0,
        T=3.0,
    )
    print(f"  Position ctrl pts    : {quart.control_points}")
    print(f"  Velocity ctrl pts    : {quart.cp_vel}")
    print(f"  Acceleration ctrl pts: {quart.cp_acc}")
    print(f"  Jerk ctrl pts        : {quart.cp_jerk}")
    print()
    print(f"  Boundary t=0 : s={quart.evaluate(0):.4f}  "
          f"v={quart.calc_first_derivative(0):.4f}  "
          f"a={quart.calc_second_derivative(0):.4f}")
    print(f"  Boundary t=T : v={quart.calc_first_derivative(3):.4f}  "
          f"a={quart.calc_second_derivative(3):.4f}  "
          f"(s={quart.evaluate(3):.4f}, free)")
    s, v, a, j = quart.state(2.1)
    print(f"  State at t=2.1: s={s:.6f}  v={v:.6f}  a={a:.6f}  j={j:.6f}")
