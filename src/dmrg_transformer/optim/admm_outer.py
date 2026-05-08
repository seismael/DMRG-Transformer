"""ADMM outer loop for inter-layer consensus (FUTURE_WORK.md Option B).

Wraps per-layer DMRG sweeps in an Alternating Direction Method of Multipliers
(ADMM) that introduces consensus variables :math:`z_\\ell` and dual variables
:math:`u_\\ell` enforcing :math:`y_\\ell \\approx z_\\ell`.  The inner DMRG
sweeps use :math:`z_\\ell + u_\\ell` as augmented targets — targets that are
*jointly consistent* across the network instead of arriving via a single
chain of pseudo-inverse pull-backs.

Reference: Boyd, Parikh, Chu (2011), §3–§7.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dmrg_transformer.propagation.target_propagator import TargetPropagator


@dataclass
class ADMMState:
    """Mutable state for one ADMM layer interface.

    ``z`` — consensus output of this layer.
    ``u`` — dual variable enforcing ``y ≈ z``.
    """

    z: torch.Tensor
    u: torch.Tensor
    rho: float


@dataclass
class ADMMReport:
    """Bookkeeping for an ADMM outer-loop run."""

    n_iter: int
    converged: bool
    primal_residuals: list[float] = field(default_factory=list)
    dual_residuals: list[float] = field(default_factory=list)
    mse_history: list[float] = field(default_factory=list)
    final_mse: float = 0.0


class ADMMOuter:
    """Alternating Direction Method of Multipliers for a chain of layers.

    Each layer must expose ``dmrg_step(X, Y, *, lam, ...)`` and ``forward(X)``.
    Linear layers must additionally expose ``to_dense_weight()`` so the z-update
    can pull targets back through them.

    Args:
        layers: ordered list of layers (first = closest to input).
        rho: ADMM penalty parameter (≥ 0).
        tol: stopping tolerance on primal / dual residuals.
        max_iter: maximum outer-loop iterations.
        rho_auto_tune: if ``True``, adapt ``ρ`` via residual balancing
            (Boyd et al. 2011 §3.4.1).
        lam: Tikhonov damping forwarded to inner DMRG sweeps.
        propagator_lam: Tikhonov damping for pseudo-inverse pull-backs
            during the z-update.
    """

    def __init__(
        self,
        layers: list,
        *,
        rho: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 50,
        rho_auto_tune: bool = True,
        lam: float = 1e-5,
        propagator_lam: float = 1e-2,
    ) -> None:
        if rho <= 0:
            raise ValueError(f"rho must be positive, got {rho}")
        if len(layers) < 1:
            raise ValueError("at least one layer required")
        self.layers = list(layers)
        self.L = len(layers)
        self.rho = float(rho)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.rho_auto_tune = bool(rho_auto_tune)
        self.lam = float(lam)
        self._propagator = TargetPropagator(lam=float(propagator_lam))
        self._states: list[ADMMState] = []
        self._mu: float = 10.0     # residual-balance factor for auto-tune
        self._tau: float = 2.0     # step-size for auto-tune

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_states(self, X: torch.Tensor, Y_global: torch.Tensor) -> None:
        """Create consensus / dual tensors at every layer output.

        Interior layer consensus is initialised from the current forward pass.
        The last layer's consensus is initialised to ``Y_global`` so the
        z-update has a driving signal from the start.
        """
        self._states.clear()
        x = X
        for i, layer in enumerate(self.layers):
            y = layer.forward(x)
            if i == self.L - 1:
                z = Y_global.detach().clone()
            else:
                z = y.detach().clone()
            u = torch.zeros_like(z)
            self._states.append(ADMMState(z=z, u=u, rho=self.rho))
            x = y

    # ------------------------------------------------------------------
    # x-update — DMRG sweep per layer
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _x_update(self, X: torch.Tensor, Y_global: torch.Tensor) -> None:
        """Run one DMRG sweep on each layer against its augmented target."""
        x = X
        for i, layer in enumerate(self.layers):
            s = self._states[i]
            T_aug = _blend_target(
                Y_global if i == self.L - 1 else None,
                s.z,
                s.u,
                s.rho,
            )
            layer.dmrg_step(x, T_aug, lam=self.lam)
            x = layer.forward(x)

    # ------------------------------------------------------------------
    # z-update — consensus projection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _z_update(self, X: torch.Tensor, Y_global: torch.Tensor) -> None:
        """Update consensus variables z_ℓ to be jointly consistent.

        For the last layer, z is pulled toward ``Y_global``.
        For interior layers, z is pulled toward the consensus of the next
        layer, mapped back through the next layer's weights via pseudo-inverse
        (linear layers) or DTP pullback (residual / nonlinear layers).
        """
        # -- forward pass: capture all outputs *and* inputs --------------------
        y_vals: list[torch.Tensor] = []
        x_vals: list[torch.Tensor] = [X.detach()]
        x = X
        for layer in self.layers:
            y = layer.forward(x)
            y_vals.append(y.detach())
            x = y
            x_vals.append(x.detach())

        # -- backward z-update ------------------------------------------------
        L = self.L
        for i in range(L - 1, -1, -1):
            s = self._states[i]
            y_i = y_vals[i]
            if i == L - 1:
                # Last layer: reference target = Y_global
                ref = Y_global
            else:
                # Pull z_{i+1} back through layer_{i+1}
                s_next = self._states[i + 1]
                z_next_dual = s_next.z - s_next.u
                x_i = x_vals[i]  # input activation *into* this layer
                ref = self._pullback_through(i + 1, z_next_dual, x_in=x_i)
            gamma = s.rho / (1.0 + s.rho)
            s.z = (1.0 - gamma) * ref + gamma * (y_i + s.u)

    # ------------------------------------------------------------------
    # u-update — dual ascent
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _u_update(self, X: torch.Tensor) -> None:
        """Dual ascent: u_ℓ ← u_ℓ + y_ℓ − z_ℓ."""
        x = X
        for i, layer in enumerate(self.layers):
            y = layer.forward(x)
            s = self._states[i]
            s.u = s.u + y.detach() - s.z
            x = y

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_primal_residual(self, X: torch.Tensor) -> float:
        """Return primal residual ‖y − z‖ across all layers."""
        x = X
        primal_sq = 0.0
        for i, layer in enumerate(self.layers):
            y = layer.forward(x)
            primal_sq += float(torch.sum((y - self._states[i].z) ** 2).item())
            x = y
        return primal_sq**0.5

    @torch.no_grad()
    def _global_mse(self, X: torch.Tensor, Y_global: torch.Tensor) -> float:
        """End-to-end MSE of the layer chain against the global target."""
        x = X
        for layer in self.layers:
            x = layer.forward(x)
        return float(torch.mean((x - Y_global) ** 2).item())

    # ------------------------------------------------------------------
    # Auto-tune ρ via residual balancing (Boyd et al. 2011 §3.4.1)
    # ------------------------------------------------------------------

    def _auto_tune_rho(self, primal: float, dual: float) -> None:
        """Adjust ρ when primal / dual residuals are out of balance."""
        if dual < 1e-12 and primal < 1e-12:
            return
        if primal > self._mu * dual:
            new_rho = self.rho * self._tau
        elif dual > self._mu * primal:
            new_rho = self.rho / self._tau
        else:
            return
        new_rho = max(min(new_rho, 1e4), 1e-6)
        self.rho = new_rho
        for s in self._states:
            s.rho = self.rho

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, X: torch.Tensor, Y_global: torch.Tensor) -> ADMMReport:
        """One full ADMM iteration (x → z → u)."""
        if len(self._states) != self.L:
            self._init_states(X, Y_global)

        # Snapshot z for dual residual
        z_prev = [s.z.detach().clone() for s in self._states]

        self._x_update(X, Y_global)
        self._z_update(X, Y_global)
        self._u_update(X)

        # Dual residual = ρ · ‖z − z_prev‖
        dual_sq = 0.0
        for i, s in enumerate(self._states):
            dual_sq += float(torch.sum((s.z - z_prev[i]) ** 2).item())
        dual = (self.rho * dual_sq) ** 0.5

        # Primal residual = ‖y − z‖
        primal_sq = 0.0
        x = X
        for i, layer in enumerate(self.layers):
            y = layer.forward(x)
            primal_sq += float(torch.sum((y - self._states[i].z) ** 2).item())
            x = y
        primal = primal_sq**0.5

        if self.rho_auto_tune:
            self._auto_tune_rho(primal, dual)

        return ADMMReport(
            n_iter=1,
            converged=False,
            primal_residuals=[primal],
            dual_residuals=[dual],
            mse_history=[self._global_mse(X, Y_global)],
        )

    @torch.no_grad()
    def solve(self, X: torch.Tensor, Y_global: torch.Tensor) -> ADMMReport:
        """Run ADMM to convergence (or ``max_iter``)."""
        self._init_states(X, Y_global)
        report = ADMMReport(n_iter=0, converged=False)
        report.mse_history.append(self._global_mse(X, Y_global))

        for iteration in range(1, self.max_iter + 1):
            result = self.step(X, Y_global)
            report.n_iter = iteration
            report.primal_residuals.extend(result.primal_residuals)
            report.dual_residuals.extend(result.dual_residuals)
            report.mse_history.append(result.mse_history[0])
            report.final_mse = report.mse_history[-1]

            # Convergence check
            primal = result.primal_residuals[0]
            dual = result.dual_residuals[0]
            if primal < self.tol and dual < self.tol:
                report.converged = True
                break

        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pullback_through(
        self,
        layer_idx: int,
        downstream_target: torch.Tensor,
        x_in: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pull a target backward through one layer for the z-update.

        ``x_in`` is the current activation feeding into the layer — required
        by residual / nonlinear layers whose ``pullback_target`` needs the
        operating point for first-order linearisation (DTP).
        """
        layer = self.layers[layer_idx]
        if hasattr(layer, "to_dense_weight"):
            W = layer.to_dense_weight()
            if hasattr(layer, "_bias"):
                downstream_target = downstream_target - layer._bias
            return self._propagator.project_through_linear(W, downstream_target)
        # Fallback for residual / nonlinear layers: use pullback_target.
        if hasattr(layer, "pullback_target"):
            if x_in is None:
                raise ValueError(
                    f"Layer {layer_idx} ({type(layer).__name__}) requires "
                    f"x_in for pullback_target; pass current activation."
                )
            return layer.pullback_target(x_in, downstream_target)
        raise TypeError(
            f"Layer {layer_idx} ({type(layer).__name__}) does not expose "
            f"to_dense_weight() or pullback_target(); cannot pull z-target back."
        )


# ------------------------------------------------------------------
# Free functions
# ------------------------------------------------------------------


def _blend_target(
    Y_global: torch.Tensor | None,
    z: torch.Tensor,
    u: torch.Tensor,
    rho: float,
) -> torch.Tensor:
    """Compute the augmented DMRG target.

    The augmented Lagrangian sub-problem is ::

        min  ‖Y − f(W)‖² + (ρ/2)‖f(W) − z + u‖²

    which is equivalent (up to a constant) to ::

        min  ‖T_aug − f(W)‖²

    where ``T_aug = (Y + (ρ/2)(z−u)) / (1 + ρ/2)``.

    For interior layers where ``Y_global`` is ``None``, the first term
    drops out and ``T_aug = z − u``.
    """
    if Y_global is None:
        return z - u
    alpha = (rho / 2.0) / (1.0 + rho / 2.0)
    return (1.0 - alpha) * Y_global + alpha * (z - u)
