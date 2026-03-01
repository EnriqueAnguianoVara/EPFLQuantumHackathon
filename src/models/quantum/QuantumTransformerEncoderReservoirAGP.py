"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC QLA-TRANSFORMER + BOSON SAMPLING RESERVOIR                          ║
║   AGP (Adiabatic Gauge Potential) variant — Krylov-subspace ITE               ║
║   Adapted for Quandela Perceval / MerLin                                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  PIPELINE (Photonic — AGP variant)                                              ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  x_t ──► PhotonicEncoder ──► BosonSamplingReservoir ──► P(n₁,...,nₘ)          ║
║           (angle-encoded       (fixed random                │                  ║
║            phase shifters)      interferometer)              ▼                  ║
║                                                    AGP-Krylov VarITE           ║
║                                       ┌────────────────────────────────┐       ║
║                                       │  H = −Σ_t |f_t⟩⟨f_t|          │       ║
║                                       │  Lanczos → Krylov basis K_l   │       ║
║                                       │  exp(−H·dτ) ≈ K expm(−T dτ)e₁│       ║
║                                       │  Per-param projection (no QGT) │       ║
║                                       └────────────┬───────────────────┘       ║
║                                                     │ compressed f(θ*)         ║
║                                                     ▼                          ║
║                                       PhotonicAttention (Q,K,V)                ║
║                                       trainable interferometers on             ║
║                                       compressed features                      ║
║                                                     │                          ║
║                                                     ▼                          ║
║                                       Classical Readout ──► ŷ_{t+h}           ║
║                                                                                 ║
║  KEY DIFFERENCE vs VarQITE-SVD (QGT) FILE                                      ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  VarQITE-SVD (QGT):                                                            ║
║    • Computes P×P quantum geometric tensor (Fubini-Study metric)               ║
║    • Solves linear system (A + λI)δθ = C  — O(P³) per step                    ║
║    • First-order Euler integration of imaginary time                           ║
║    • Requires Tikhonov regularisation for ill-conditioned QGT                  ║
║                                                                                 ║
║  AGP-Krylov (this file):                                                        ║
║    • Builds l-dimensional Krylov subspace via Lanczos                          ║
║    • Computes matrix exponential in small l×l subspace (l ≪ P)                 ║
║    • Per-parameter projection — O(P·d) instead of O(P³)                        ║
║    • Counter-diabatic (CD) acceleration: the Krylov exponential                ║
║      implicitly includes all orders of the AGP within K_l,                     ║
║      equivalent to variational CD driving                                      ║
║    • No QGT computation or inversion required                                  ║
║                                                                                 ║
║  MATHEMATICAL BASIS                                                             ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  The Adiabatic Gauge Potential A_λ generates exact adiabatic transport:        ║
║      ∂_λ|ψ₀(λ)⟩ = A_λ|ψ₀(λ)⟩                                                ║
║                                                                                 ║
║  For imaginary-time VarITE with H = −Σ_t f_t f_tᵀ:                            ║
║    1. ITE target: F = −(H − E)f  with  E = fᵀHf / fᵀf                        ║
║    2. Lanczos builds orthonormal Krylov basis K_l from {F, HF, …, H^l F}      ║
║    3. Krylov exponential: exp(−Hτ)f ≈ ‖F‖ · K_l · expm(−T_l τ) · e₁          ║
║       where T_l is the l×l tridiagonal Lanczos matrix                          ║
║    4. This is equivalent to the l-th order variational AGP / CD driving:       ║
║       A^(l) = Σ_{k=0}^l b_k (ad_H)^k F                                       ║
║       where (ad_H)^k = [H,[H,…[H,·]…]] (nested commutators)                  ║
║    5. Parameter update: δθ_i = ⟨∂_if, direction⟩ / ⟨∂_if, ∂_if⟩              ║
║       (per-parameter projection, no P×P matrix inversion)                      ║
║                                                                                 ║
║  References:                                                                    ║
║    • Sels & Polkovnikov, Phys. Rev. A 95, 023621 (2017)                       ║
║    • Claeys et al., Phys. Rev. Lett. 123, 090602 (2019)                       ║
║    • Park & Killoran, Quantum 8, 1457 (2024) — photonic VarQITE              ║
║                                                                                 ║
║  Requirements: pip install perceval-quandela merlinquantum torch numpy         ║
║                         matplotlib scipy openpyxl pandas scikit-learn          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import perceval as pcvl
from scipy.linalg import expm as scipy_expm
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from merlin import LexGrouping, QuantumLayer, MeasurementStrategy, ComputationSpace
    from merlin.builder import CircuitBuilder
    HAS_MERLIN = True
except ImportError:
    HAS_MERLIN = False

# Try to re-use shared photonic components from the QGT variant.
# The module name contains hyphens, so we use importlib.
_SHARED_IMPORTED = False
try:
    import importlib
    _qgt_mod = importlib.import_module(
        "src.models.quantum.QuantumTransformerEncoderReservoirVarQITE-SVD"
    )
    PhotonicEncoder = _qgt_mod.PhotonicEncoder
    BosonSamplingReservoir = _qgt_mod.BosonSamplingReservoir
    PhotonicFeatureExtractor = _qgt_mod.PhotonicFeatureExtractor
    PhotonicAttentionHead = _qgt_mod.PhotonicAttentionHead
    PhotonicAttentionLayer = _qgt_mod.PhotonicAttentionLayer
    _SHARED_IMPORTED = True
except Exception:
    _SHARED_IMPORTED = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicAGPConfig:
    """Central configuration for the Photonic QLA-Transformer (AGP variant)."""

    # ── Photonic circuit ───────────────────────────────────────────────────────
    n_modes:           int   = 8      # optical modes
    n_photons:         int   = 4      # photons injected
    n_res_layers:      int   = 3      # reservoir interferometer depth
    reservoir_seed:    int   = 42

    # ── AGP-Krylov ITE (replaces VarQITE-QGT) ────────────────────────────────
    ansatz_layers:     int   = 2      # depth of trainable ansatz interferometer
    ite_steps:         int   = 8      # imaginary-time evolution steps per fwd
    ite_dtau:          float = 0.08   # imaginary time step Δτ
    agp_krylov_order:  int   = 6      # Krylov subspace dimension l
    agp_reg:           float = 1e-4   # per-parameter projection regularisation
    agp_lanczos_reorth: bool = True   # full Lanczos re-orthogonalisation
    param_shift:       float = np.pi / 2  # parameter-shift rule angle

    # ── Attention ─────────────────────────────────────────────────────────────
    n_heads:           int   = 2
    n_attn_layers:     int   = 2

    # ── Sequence ──────────────────────────────────────────────────────────────
    seq_len:           int   = 10
    horizon:           int   = 1
    n_features:        int   = 5

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs:          int   = 12
    lr_readout:        float = 0.01
    lr_attn:           float = 0.01
    attn_shift:        float = np.pi / 2
    batch_attn:        int   = 4
    batch_size:        int   = 16

    # ── Feature grouping ──────────────────────────────────────────────────────
    n_fock_features:   int   = 20

    @property
    def ansatz_n_params(self) -> int:
        """Trainable phase-shifters in the ITE ansatz."""
        return self.n_modes * self.ansatz_layers

    @property
    def feature_dim(self) -> int:
        return self.n_fock_features * self.n_heads

    @property
    def attn_n_params(self) -> int:
        return self.n_modes * self.n_attn_layers


# ═══════════════════════════════════════════════════════════════════════════════
# 1–3. PHOTONIC ENCODER, RESERVOIR, FEATURE EXTRACTOR
#      Re-used from QuantumTransformerEncoderReservoirVarQITE-SVD.py
#      (imported above).  If running standalone, fall back to local copies.
# ═══════════════════════════════════════════════════════════════════════════════

if not _SHARED_IMPORTED:
    # ── Local fallback copies (identical to QGT variant) ──────────────────────

    class PhotonicEncoder:
        """Maps x ∈ ℝ^d → phase-shifter configuration on optical modes."""

        def __init__(self, n_modes: int, n_features: int):
            self.n_modes = n_modes
            self.n_features = min(n_features, n_modes)

        def build_encoding_circuit(self, x: np.ndarray) -> pcvl.Circuit:
            circ = pcvl.Circuit(self.n_modes)
            for i in range(0, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())
            for i in range(self.n_features):
                circ.add(i, pcvl.PS(float(x[i]) * np.pi))
            for i in range(1, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())
            return circ

    class BosonSamplingReservoir:
        """Fixed random photonic interferometer (never trained)."""

        def __init__(self, n_modes: int, n_layers: int, seed: int = 42):
            self.n_modes = n_modes
            rng = np.random.default_rng(seed)
            self._circuit = self._build(rng, n_layers)

        def _build(self, rng, n_layers) -> pcvl.Circuit:
            circ = pcvl.Circuit(self.n_modes)
            for layer in range(n_layers):
                for i in range(self.n_modes):
                    circ.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))
                start = layer % 2
                for i in range(start, self.n_modes - 1, 2):
                    circ.add(i, pcvl.BS())
                for i in range(self.n_modes):
                    circ.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))
            return circ

        @property
        def circuit(self) -> pcvl.Circuit:
            return self._circuit

    class PhotonicFeatureExtractor:
        """Extracts Fock-state probability features from photonic circuits."""

        def __init__(self, n_modes: int, n_photons: int, n_features: int = 20):
            self.n_modes = n_modes
            self.n_photons = n_photons
            self.n_features = n_features
            state_list = [0] * n_modes
            for i in range(min(n_photons, n_modes)):
                state_list[i] = 1
            self.input_state = pcvl.BasicState(state_list)

        def extract(self, circuit: pcvl.Circuit) -> np.ndarray:
            backend = pcvl.BackendFactory.get_backend("SLOS")
            backend.set_circuit(circuit)
            backend.set_input_state(self.input_state)
            probs = backend.prob_distribution()

            features = np.zeros(self.n_features)
            mode_occ = np.zeros(self.n_modes)
            for state, prob in probs.items():
                for mode in range(self.n_modes):
                    mode_occ[mode] += float(prob) * state[mode]

            n_occ = min(self.n_modes, self.n_features)
            features[:n_occ] = mode_occ[:n_occ]

            if self.n_features > n_occ:
                prob_list = sorted(
                    [(float(p), s) for s, p in probs.items()],
                    key=lambda x: x[0], reverse=True,
                )
                for j, (p, _) in enumerate(prob_list):
                    idx = n_occ + j
                    if idx >= self.n_features:
                        break
                    features[idx] = p
            return features

    class PhotonicAttentionHead:
        """One photonic attention head (Q/K/V interferometers)."""

        def __init__(self, n_modes, n_layers, n_photons, n_features, head_id):
            self.n_modes = n_modes
            self.n_layers = n_layers
            self.P = n_modes * n_layers
            rng = np.random.default_rng(seed=head_id * 31337 + 17)
            self.params_q = rng.uniform(0, 2 * np.pi, self.P)
            self.params_k = rng.uniform(0, 2 * np.pi, self.P)
            self.params_v = rng.uniform(0, 2 * np.pi, self.P)
            self.extractor = PhotonicFeatureExtractor(n_modes, n_photons, n_features)

        def _build_proj_circuit(self, params):
            circ = pcvl.Circuit(self.n_modes)
            for layer in range(self.n_layers):
                for i in range(self.n_modes):
                    circ.add(i, pcvl.PS(float(params[layer * self.n_modes + i])))
                start = layer % 2
                for i in range(start, self.n_modes - 1, 2):
                    circ.add(i, pcvl.BS())
            return circ

        def _project(self, base_circuit, params):
            proj = self._build_proj_circuit(params)
            full = pcvl.Circuit(self.n_modes)
            full.add(0, base_circuit)
            full.add(0, proj)
            return self.extractor.extract(full)

        def project(self, base_circuit, which):
            p = {"q": self.params_q, "k": self.params_k,
                 "v": self.params_v}[which]
            return self._project(base_circuit, p)

        def get_params(self):
            return np.concatenate([self.params_q, self.params_k, self.params_v])

        def set_params(self, flat):
            n = self.P
            self.params_q = flat[0:n].copy()
            self.params_k = flat[n:2*n].copy()
            self.params_v = flat[2*n:3*n].copy()

    class PhotonicAttentionLayer:
        """Multi-head photonic attention."""

        def __init__(self, cfg):
            self.cfg = cfg
            self.heads = [
                PhotonicAttentionHead(
                    cfg.n_modes, cfg.n_attn_layers,
                    cfg.n_photons, cfg.n_fock_features, h,
                )
                for h in range(cfg.n_heads)
            ]

        def forward(self, compressed_circuits):
            T = len(compressed_circuits)
            d = self.cfg.n_fock_features
            head_outs = []
            for head in self.heads:
                Qs = np.array([head.project(c, "q") for c in compressed_circuits])
                Ks = np.array([head.project(c, "k") for c in compressed_circuits])
                Vs = np.array([head.project(c, "v") for c in compressed_circuits])
                scores = (Qs @ Ks.T) / np.sqrt(d)
                mask = np.triu(np.full((T, T), -1e9), k=1)
                scores += mask
                scores -= scores.max(axis=-1, keepdims=True)
                exp_s = np.exp(scores)
                alpha = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-12)
                head_outs.append(alpha @ Vs)
            return np.concatenate(head_outs, axis=-1)

        def get_all_params(self):
            return np.concatenate([h.get_params() for h in self.heads])

        def set_all_params(self, flat):
            n = len(self.heads[0].get_params())
            for i, head in enumerate(self.heads):
                head.set_params(flat[i * n: (i + 1) * n])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AGP-KRYLOV VARIATIONAL IMAGINARY-TIME EVOLUTION (PHOTONIC)
#
#    Replaces VarQITE-SVD (QGT / McLachlan) with the Adiabatic Gauge
#    Potential approach.
#
#    Core algorithm per imaginary-time step:
#      1.  f(θ)         ← feature vector from photonic circuit
#      2.  H_eff f      ← −Σ_t f_t (f_t · f)   (matrix-free H action)
#      3.  E            ← fᵀ H_eff f / fᵀf      (Rayleigh quotient)
#      4.  F            ← −(H_eff − E) f         (ITE tangent / residual)
#      5.  Lanczos(F,H) → K_l, T_l               (l-dim Krylov basis)
#      6.  direction    ← ‖F‖ · K_l expm(−T_l Δτ) e₁   (Krylov exponential)
#      7.  δθ_i         ← ⟨∂_i f, direction⟩ / ⟨∂_i f, ∂_i f⟩  (no QGT)
#
#    Computational complexity per step:
#      • Lanczos:   O(l · T · d)   — l Krylov vecs, T trajectory len, d feat dim
#      • Gradients: O(P · d)       — P param-shift evaluations
#      • QGT path:  O(P² · d + P³) — full QGT build + solve   ← AVOIDED
#
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicAGPVarITE:
    """
    AGP-based Variational Imaginary-Time Evolution on photonic feature
    trajectories from a boson-sampling reservoir.

    Instead of computing the P×P quantum geometric tensor (QGT) and solving

        (A + λI) δθ = C                     [McLachlan / QGT approach]

    this module:

      1. **Lanczos** — builds an l-dimensional orthonormal Krylov subspace
         K_l = span{F, HF, H²F, …} from the ITE residual F and the
         effective Hamiltonian H = −Σ_t f_t f_tᵀ.

      2. **Krylov exponential** — approximates the exact ITE propagator
         exp(−H Δτ) f  ≈  ‖F‖ · K_l · expm(−T_l Δτ) · e₁
         where T_l is the l×l tridiagonal Lanczos matrix.
         This is equivalent to including all orders 0…l of the variational
         Adiabatic Gauge Potential (counter-diabatic driving) within K_l.

      3. **Per-parameter projection** — maps the Krylov-enhanced direction
         back to parameter space via diagonal projection:
             δθ_i = ⟨∂_i f, direction⟩ / (⟨∂_i f, ∂_i f⟩ + ε)
         Cost: O(P·d) vs O(P³) for QGT inversion.

    Advantages
    ----------
    • **No QGT computation or inversion** — avoids O(P²) finite-difference
      evaluations and O(P³) linear solve.
    • **Higher-order ITE** — Krylov exponential captures non-linear dynamics
      that first-order Euler (McLachlan) misses.
    • **Natural regularisation** — Krylov subspace is inherently
      low-rank; no Tikhonov parameter needed.
    • **Counter-diabatic acceleration** — the matrix exponential in the
      Krylov basis implicitly includes the AGP to all orders reachable
      within K_l, accelerating convergence to the dominant eigenvector.
    """

    def __init__(self, cfg: PhotonicAGPConfig):
        self.cfg = cfg
        self.n_modes = cfg.n_modes
        self.n_photons = cfg.n_photons
        self.P = cfg.ansatz_n_params
        self.s = cfg.param_shift
        self.extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features,
        )

    # ─── ansatz ───────────────────────────────────────────────────────────────

    def _build_ansatz(self, params: np.ndarray) -> pcvl.Circuit:
        """Trainable photonic ansatz: [PS(θ_i) + BS ladder] × n_layers."""
        circ = pcvl.Circuit(self.n_modes)
        for i in range(0, self.n_modes - 1, 2):
            circ.add(i, pcvl.BS())
        for layer in range(self.cfg.ansatz_layers):
            for i in range(self.n_modes):
                idx = layer * self.n_modes + i
                circ.add(i, pcvl.PS(float(params[idx])))
            start = layer % 2
            for i in range(start, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())
        return circ

    def _get_features(self, params: np.ndarray,
                      base_circuit: pcvl.Circuit) -> np.ndarray:
        """Compose base_circuit + ansatz(θ) → extract feature vector."""
        ansatz = self._build_ansatz(params)
        full = pcvl.Circuit(self.n_modes)
        full.add(0, base_circuit)
        full.add(0, ansatz)
        return self.extractor.extract(full)

    # ─── effective Hamiltonian (matrix-free) ──────────────────────────────────

    @staticmethod
    def _apply_H(v: np.ndarray,
                 reservoir_features: List[np.ndarray]) -> np.ndarray:
        """
        Apply H_eff = −Σ_t f_t f_tᵀ  to vector v  (matrix-free).

        Cost: O(T·d)  where T = len(reservoir_features), d = len(v).
        """
        result = np.zeros_like(v)
        for f_t in reservoir_features:
            result -= f_t * np.dot(f_t, v)
        return result

    # ─── Lanczos tridiagonalisation ───────────────────────────────────────────

    def _lanczos(
        self,
        v0: np.ndarray,
        reservoir_features: List[np.ndarray],
        order: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Lanczos algorithm with optional full re-orthogonalisation.

        Starting vector v0 is normalised internally.

        Returns
        -------
        Q : (d, l) orthonormal Krylov basis vectors
        alpha : (l,) diagonal of tridiagonal T
        beta  : (l−1,) off-diagonal of T
        """
        d = len(v0)
        l = min(order, d)

        Q = np.zeros((d, l))
        alpha = np.zeros(l)
        beta = np.zeros(max(l - 1, 0))

        norm0 = np.linalg.norm(v0)
        if norm0 < 1e-15:
            return Q[:, :1], alpha[:1], beta[:0]

        Q[:, 0] = v0 / norm0
        actual_l = l  # may shrink if lucky breakdown

        for j in range(l):
            # Lanczos step: w = H q_j − β_{j−1} q_{j−1}
            w = self._apply_H(Q[:, j], reservoir_features)
            alpha[j] = np.dot(Q[:, j], w)
            w -= alpha[j] * Q[:, j]
            if j > 0:
                w -= beta[j - 1] * Q[:, j - 1]

            # Full re-orthogonalisation (numerically safer)
            if self.cfg.agp_lanczos_reorth:
                for k in range(j + 1):
                    w -= np.dot(Q[:, k], w) * Q[:, k]

            b = np.linalg.norm(w)
            if j < l - 1:
                beta[j] = b
                if b < 1e-14:
                    actual_l = j + 1
                    break
                Q[:, j + 1] = w / b

        return Q[:, :actual_l], alpha[:actual_l], beta[:max(actual_l - 1, 0)]

    # ─── Krylov exponential (core AGP step) ───────────────────────────────────

    @staticmethod
    def _krylov_expm(
        Q: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        norm_v0: float,
        dtau: float,
    ) -> np.ndarray:
        """
        Approximate  exp(−H·Δτ) v₀  in the Krylov subspace.

            exp(−H Δτ) v₀  ≈  ‖v₀‖ · Q · expm(−T Δτ) · e₁

        where T is the l×l symmetric tridiagonal Lanczos matrix.

        This is the mathematical equivalent of the variational AGP / CD
        driving to order l: the Krylov basis spans exactly the subspace
        that the first l nested commutators [H,[H,…[H,F]…]] generate.
        """
        l = len(alpha)
        if l == 0:
            return np.zeros(Q.shape[0])

        # Build tridiagonal T
        T = np.diag(alpha)
        for j in range(len(beta)):
            T[j, j + 1] = beta[j]
            T[j + 1, j] = beta[j]

        # Matrix exponential in small l×l space
        exp_T = scipy_expm(-T * dtau)

        # First basis vector coefficient
        e0 = np.zeros(l)
        e0[0] = 1.0
        coeffs = norm_v0 * (exp_T @ e0)

        return Q @ coeffs

    # ─── per-parameter projection ─────────────────────────────────────────────

    def _project_to_params(
        self,
        direction: np.ndarray,
        params: np.ndarray,
        base_circuit: pcvl.Circuit,
    ) -> np.ndarray:
        """
        Map a feature-space direction to parameter updates via per-parameter
        projection (diagonal approximation of the normal equations).

            δθ_i = ⟨∂_i f, direction⟩ / (‖∂_i f‖² + ε)

        This avoids the O(P²) QGT computation and O(P³) inversion:
        • QGT approach builds the full ∂f/∂θ_i · ∂f/∂θ_j  matrix → O(P²·d + P³)
        • Per-parameter projection is O(P·d)

        The approximation is exact when ∂_i f are orthogonal (or nearly so)
        and remains a good descent direction otherwise thanks to the
        Krylov-enhanced target direction.
        """
        dtheta = np.zeros(self.P)
        reg = self.cfg.agp_reg

        for i in range(self.P):
            p_plus = params.copy();  p_plus[i] += self.s
            p_minus = params.copy(); p_minus[i] -= self.s
            f_plus = self._get_features(p_plus, base_circuit)
            f_minus = self._get_features(p_minus, base_circuit)
            df_i = (f_plus - f_minus) / (2.0 * np.sin(self.s))

            norm_sq = np.dot(df_i, df_i)
            dtheta[i] = np.dot(df_i, direction) / (norm_sq + reg)

        return dtheta

    # ─── single ITE step (AGP-Krylov) ────────────────────────────────────────

    def _agp_step(
        self,
        params: np.ndarray,
        reservoir_features: List[np.ndarray],
        base_circuit: pcvl.Circuit,
    ) -> np.ndarray:
        """
        One AGP-Krylov imaginary-time step.

        1. Compute current features f(θ)
        2. Compute ITE residual F = −(H − E)f
        3. Lanczos → Krylov basis K_l + tridiagonal T_l
        4. Krylov exponential → target direction in feature space
        5. Per-parameter projection → δθ

        Returns
        -------
        delta_theta : (P,) parameter update vector
        """
        f = self._get_features(params, base_circuit)

        # ── H_eff action and energy ──────────────────────────────────────────
        Hf = self._apply_H(f, reservoir_features)
        f_norm_sq = np.dot(f, f) + 1e-15
        E = np.dot(f, Hf) / f_norm_sq       # Rayleigh quotient

        # ── ITE residual (target drift in feature space) ─────────────────────
        F = -(Hf - E * f)                    # projected ITE tangent
        norm_F = np.linalg.norm(F)

        if norm_F < 1e-14:
            # Already at eigenstate — no update needed
            return np.zeros(self.P)

        # ── Lanczos → Krylov subspace ────────────────────────────────────────
        Q, alpha, beta = self._lanczos(
            F, reservoir_features, self.cfg.agp_krylov_order,
        )

        # ── Krylov exponential (AGP / CD-enhanced direction) ─────────────────
        direction = self._krylov_expm(Q, alpha, beta, norm_F, self.cfg.ite_dtau)

        # ── Per-parameter projection (no QGT!) ──────────────────────────────
        delta_theta = self._project_to_params(direction, params, base_circuit)

        return delta_theta

    # ─── full ITE run ─────────────────────────────────────────────────────────

    def run(
        self,
        reservoir_features: List[np.ndarray],
        base_circuit: pcvl.Circuit,
        init_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run AGP-Krylov ITE over multiple steps.

        Args
        ----
        reservoir_features : T feature vectors from reservoir
        base_circuit       : composed encoder + reservoir circuit
        init_params        : initial θ; random if None

        Returns
        -------
        params_final : converged ansatz parameters θ*
        features     : feature vector at θ*  (shape n_fock_features,)
        """
        cfg = self.cfg
        rng = np.random.default_rng(
            seed=int(abs(reservoir_features[0][0]) * 1e6) % (2**31),
        )
        params = (
            init_params.copy()
            if init_params is not None
            else rng.uniform(0, 2 * np.pi, self.P)
        )

        for step in range(cfg.ite_steps):
            delta = self._agp_step(params, reservoir_features, base_circuit)
            params = params + cfg.ite_dtau * delta

        features = self._get_features(params, base_circuit)
        return params, features


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PHOTONIC ATTENTION
#    Re-used from QGT variant (imported or locally defined above).
#    The attention layer is identical — only the ITE compression changes.
# ═══════════════════════════════════════════════════════════════════════════════

# (PhotonicAttentionHead and PhotonicAttentionLayer already available)


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. MerLin ATTENTION (optional — differentiable QuantumLayer heads)
# ═══════════════════════════════════════════════════════════════════════════════

if HAS_MERLIN:

    class MerLinAttentionHead(nn.Module):
        """Attention head using MerLin QuantumLayer for differentiable Q/K/V."""

        def __init__(self, n_modes: int, n_photons: int,
                     input_dim: int, output_dim: int):
            super().__init__()
            self.n_modes = n_modes
            self.n_photons = n_photons
            self.output_dim = output_dim
            self.q_layer = self._build_layer(input_dim, "Q")
            self.k_layer = self._build_layer(input_dim, "K")
            self.v_layer = self._build_layer(input_dim, "V")

        def _build_layer(self, input_dim: int, name: str) -> nn.Module:
            builder = CircuitBuilder(n_modes=self.n_modes)
            builder.add_entangling_layer(trainable=True, name=f"{name}_U1")
            builder.add_angle_encoding(
                modes=list(range(min(input_dim, self.n_modes))),
                name=f"{name}_input",
            )
            builder.add_rotations(trainable=True, name=f"{name}_theta")
            builder.add_superpositions(depth=1)
            core = QuantumLayer(
                input_size=input_dim,
                builder=builder,
                n_photons=self.n_photons,
                dtype=torch.float32,
            )
            return nn.Sequential(core, LexGrouping(core.output_size, self.output_dim))

        def forward(self, x: torch.Tensor):
            return self.q_layer(x), self.k_layer(x), self.v_layer(x)

    class MerLinAttentionLayer(nn.Module):
        """Multi-head photonic attention via MerLin QuantumLayers."""

        def __init__(self, cfg: PhotonicAGPConfig, input_dim: int):
            super().__init__()
            self.cfg = cfg
            self.heads = nn.ModuleList([
                MerLinAttentionHead(
                    cfg.n_modes, cfg.n_photons, input_dim, cfg.n_fock_features,
                )
                for _ in range(cfg.n_heads)
            ])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            T = x.shape[0]
            d = self.cfg.n_fock_features
            outs = []
            for head in self.heads:
                Q, K, V = head(x)
                scores = (Q @ K.T) / np.sqrt(d)
                mask = torch.triu(torch.full((T, T), -1e9), diagonal=1)
                scores += mask
                outs.append(torch.softmax(scores, dim=-1) @ V)
            return torch.cat(outs, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FULL PHOTONIC QLA-TRANSFORMER (AGP VARIANT)
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicAGPTransformer(nn.Module):
    """
    Full Photonic Quantum Linear Algebra Transformer — AGP variant.

    Architecture identical to PhotonicQLATransformer except the
    VarQITE-SVD compression (section 4) is replaced by AGP-Krylov ITE.

    Forward pass
    ------------
    1. PhotonicEncoder          → phase-encoded photonic circuit per time step
    2. BosonSamplingReservoir   → fixed random interferometer (QRC)
    3. **AGP-Krylov VarITE**    → compressed features (dominant subspace)
    4. PhotonicAttention        → multi-head causal attention
    5. Classical readout        → prediction ŷ_{t+h}

    Where step 3 uses:
      • Lanczos tridiagonalisation (l ≪ P)
      • Krylov exponential ≈ variational CD driving to order l
      • Per-parameter projection (no QGT)
    """

    def __init__(self, cfg: PhotonicAGPConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PhotonicEncoder(cfg.n_modes, cfg.n_features)
        self.reservoir = BosonSamplingReservoir(
            cfg.n_modes, cfg.n_res_layers, cfg.reservoir_seed,
        )
        self.agp_ite = PhotonicAGPVarITE(cfg)
        self.feature_extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features,
        )

        # Photonic attention (Perceval path)
        self.attention = PhotonicAttentionLayer(cfg)

        # Classical readout
        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

        # Warm-start cache for ITE params
        self._ite_cache: Optional[np.ndarray] = None

    def _build_reservoir_circuit(self, x: np.ndarray) -> pcvl.Circuit:
        enc = self.encoder.build_encoding_circuit(x)
        res = self.reservoir.circuit
        full = pcvl.Circuit(self.cfg.n_modes)
        full.add(0, enc)
        full.add(0, res)
        return full

    def _reservoir_features(self, sequence: np.ndarray):
        circuits, features = [], []
        for t in range(len(sequence)):
            circ = self._build_reservoir_circuit(sequence[t])
            feat = self.feature_extractor.extract(circ)
            circuits.append(circ)
            features.append(feat)
        return circuits, features

    def _agp_compress(
        self,
        circuits: List[pcvl.Circuit],
        features: List[np.ndarray],
    ) -> Tuple[List[pcvl.Circuit], np.ndarray]:
        """
        Run AGP-Krylov ITE on causal windows of reservoir features.

        For token t: context = features[0:t+1] (causal).
        """
        compressed = []
        init_p = self._ite_cache

        for t in range(len(circuits)):
            context = features[:t + 1]
            params, _ = self.agp_ite.run(
                context, circuits[t], init_params=init_p,
            )
            ansatz = self.agp_ite._build_ansatz(params)
            comp = pcvl.Circuit(self.cfg.n_modes)
            comp.add(0, circuits[t])
            comp.add(0, ansatz)
            compressed.append(comp)
            init_p = params

        self._ite_cache = init_p
        return compressed, init_p

    def forward_np(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Full forward pass (numpy input → torch output).

        Args:
            sequence: (T, n_features)

        Returns:
            prediction: (horizon,) tensor
        """
        circuits, features = self._reservoir_features(sequence)
        comp_circuits, _ = self._agp_compress(circuits, features)
        attn_out = self.attention.forward(comp_circuits)
        last = torch.tensor(attn_out[-1], dtype=torch.float32)
        return self.readout(last)

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        return self.forward_np(sequence).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. MerLin HYBRID (AGP variant)
# ═══════════════════════════════════════════════════════════════════════════════

class MerLinAGPHybridModel(nn.Module):
    """
    Hybrid model:
      • Perceval reservoir for feature extraction
      • AGP-Krylov ITE for compression
      • MerLin QuantumLayer for differentiable attention
      • PyTorch readout
    """

    def __init__(self, cfg: PhotonicAGPConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PhotonicEncoder(cfg.n_modes, cfg.n_features)
        self.reservoir = BosonSamplingReservoir(
            cfg.n_modes, cfg.n_res_layers, cfg.reservoir_seed,
        )
        self.feature_extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features,
        )
        self.agp_ite = PhotonicAGPVarITE(cfg)

        if HAS_MERLIN:
            self.attention = MerLinAttentionLayer(cfg, cfg.n_fock_features)
        else:
            self.attention = None

        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )
        self._ite_cache = None

    def _extract_features(self, sequence: np.ndarray) -> np.ndarray:
        all_features, all_circuits = [], []
        for t in range(len(sequence)):
            enc = self.encoder.build_encoding_circuit(sequence[t])
            res = self.reservoir.circuit
            circ = pcvl.Circuit(self.cfg.n_modes)
            circ.add(0, enc)
            circ.add(0, res)
            all_features.append(self.feature_extractor.extract(circ))
            all_circuits.append(circ)

        compressed = []
        init_p = self._ite_cache
        for t in range(len(sequence)):
            context = all_features[:t + 1]
            params, comp_feat = self.agp_ite.run(
                context, all_circuits[t], init_params=init_p,
            )
            compressed.append(comp_feat)
            init_p = params

        self._ite_cache = init_p
        return np.array(compressed)

    def forward(self, sequence: np.ndarray) -> torch.Tensor:
        features = self._extract_features(sequence)
        features_t = torch.tensor(features, dtype=torch.float32)

        if self.attention is not None:
            attn_out = self.attention(features_t)
        else:
            attn_out = features_t.repeat(1, self.cfg.n_heads)

        return self.readout(attn_out[-1])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DATA LOADING (identical to QGT variant)
# ═══════════════════════════════════════════════════════════════════════════════

def load_swaption_data(
    train_path: str,
    cfg: PhotonicAGPConfig,
    target_columns: Optional[List[str]] = None,
    n_pca_features: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """Load swaption data, PCA-reduce, and create sequences."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = pd.read_excel(train_path)
    date_col = df.columns[0]
    data = df.drop(columns=[date_col]).ffill().bfill()
    values = data.values.astype(np.float32)

    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    pca = PCA(n_components=n_pca_features)
    values_pca = pca.fit_transform(values_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    v_min = values_pca.min(axis=0, keepdims=True)
    v_max = values_pca.max(axis=0, keepdims=True)
    values_norm = (values_pca - v_min) / (v_max - v_min + 1e-9)

    X, y = [], []
    for i in range(len(values_norm) - cfg.seq_len - cfg.horizon + 1):
        X.append(values_norm[i: i + cfg.seq_len])
        y.append(values_norm[i + cfg.seq_len + cfg.horizon - 1])

    X, y = np.array(X), np.array(y)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, (scaler, pca, v_min, v_max)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: PhotonicAGPTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: PhotonicAGPConfig,
    verbose: bool = True,
) -> List[float]:
    """
    Hybrid training: Adam on readout + parameter-shift on attention.
    (Identical structure to QGT variant — only the internal ITE differs.)
    """
    optimizer = optim.Adam(model.readout.parameters(), lr=cfg.lr_readout)
    loss_fn = nn.MSELoss()
    losses = []
    N = len(X_train)
    s = cfg.attn_shift
    rng = np.random.default_rng(0)

    for epoch in range(cfg.n_epochs):
        epoch_loss = 0.0
        perm = rng.permutation(N)

        for i in perm[:cfg.batch_size]:
            seq = X_train[i]
            tgt = torch.tensor(y_train[i][:cfg.horizon], dtype=torch.float32)
            optimizer.zero_grad()
            pred = model.forward_np(seq)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Parameter-shift on attention
        q_params = model.attention.get_all_params()
        P = len(q_params)
        q_grad = np.zeros(P)
        sample_idx = rng.choice(N, min(cfg.batch_attn, N), replace=False)
        shift_idx = rng.choice(P, min(P, 12), replace=False)

        for idx in sample_idx:
            seq = X_train[idx]
            tgt = float(y_train[idx][0])
            for j in shift_idx:
                p_plus = q_params.copy(); p_plus[j] += s
                model.attention.set_all_params(p_plus)
                loss_plus = (model.forward_np(seq).item() - tgt) ** 2

                p_minus = q_params.copy(); p_minus[j] -= s
                model.attention.set_all_params(p_minus)
                loss_minus = (model.forward_np(seq).item() - tgt) ** 2

                q_grad[j] += (loss_plus - loss_minus) / (2.0 * np.sin(s))

        model.attention.set_all_params(q_params)
        q_grad /= max(len(sample_idx), 1)
        q_params -= cfg.lr_attn * q_grad
        model.attention.set_all_params(q_params)

        avg = epoch_loss / min(cfg.batch_size, N)
        losses.append(avg)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs}"
                  f"  |  MSE: {avg:.6f}"
                  f"  |  ‖∇attn‖: {np.linalg.norm(q_grad):.5f}")

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    losses: List[float],
    cfg: PhotonicAGPConfig,
    save_path: str = "photonic_agp_results.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0a0a18")
    colors = {"bg": "#12122a", "spine": "#333", "true": "#4dd0e1",
              "pred": "#ff7043", "loss": "#ce93d8"}

    for ax in axes:
        ax.set_facecolor(colors["bg"])
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor(colors["spine"])

    # Panel 1: Forecast
    t = np.arange(len(y_true))
    axes[0].plot(t, y_true, color=colors["true"], lw=2, label="Ground truth")
    axes[0].plot(t, y_pred, color=colors["pred"], lw=2, ls="--",
                 label="AGP-Krylov QLAT")
    axes[0].fill_between(t, y_true, y_pred, alpha=0.12, color=colors["pred"])
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    axes[0].text(0.97, 0.05,
                 f"MSE={mse:.5f}\nMAE={mae:.5f}\nR²={r2:.4f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 color="white", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="#111", ec="#555"))
    axes[0].set_title("AGP-Krylov Forecast", color="white", fontsize=12)
    axes[0].set_xlabel("Time step", color="white")
    axes[0].set_ylabel("Normalised value", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=9)

    # Panel 2: Training loss
    axes[1].plot(losses, color=colors["loss"], lw=2)
    axes[1].set_title("Training Loss (MSE)", color="white", fontsize=12)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss", color="white")
    if losses and min(losses) > 0:
        axes[1].set_yscale("log")

    # Panel 3: Architecture
    ax3 = axes[2]
    ax3.axis("off")
    diagram = [
        ("Input x_t (swaption features)",       0.92),
        ("↓",                                    0.86),
        ("Photonic Encoder",                     0.80),
        ("angle → PS on modes + BS ring",        0.74),
        ("↓",                                    0.68),
        ("Boson Sampling Reservoir [FIXED]",     0.62),
        ("random PS + BS interferometer",        0.56),
        ("↓",                                    0.50),
        ("AGP-Krylov VarITE",                    0.44),
        ("Lanczos → K_l, expm(−T Δτ)",          0.38),
        ("per-param projection (NO QGT)",        0.32),
        ("↓",                                    0.26),
        ("Photonic Attention (Q,K,V heads)",     0.20),
        ("↓",                                    0.14),
        ("Classical Readout → ŷ_{t+h}",          0.08),
    ]
    for i, (label, y_pos) in enumerate(diagram):
        weight = "bold" if i in {0, 2, 5, 8, 12, 14} else "normal"
        color_ = ("#4dd0e1" if i == 0 else
                  "#a5d6a7" if i in {5, 6} else
                  "#ce93d8" if i in {8, 9, 10} else
                  "#ff7043" if i in {12, 13} else
                  "#fff59d" if i == 14 else "white")
        ax3.text(0.5, y_pos, label, ha="center", va="center",
                 color=color_, fontsize=8.5, fontweight=weight,
                 fontfamily="monospace", transform=ax3.transAxes)
    ax3.set_title("Model Architecture (AGP)", color="white", fontsize=12)

    arch = (f"modes={cfg.n_modes}  |  photons={cfg.n_photons}  |  "
            f"krylov_l={cfg.agp_krylov_order}  |  heads={cfg.n_heads}")
    fig.suptitle(f"Photonic QLA-Transformer + AGP-Krylov ITE\n{arch}",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN — SWAPTION PRICING
# ═══════════════════════════════════════════════════════════════════════════════

def main(train_path: str = "train.xlsx"):
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Photonic QLA-Transformer + Boson Sampling Reservoir            ║")
    print("║  *** AGP-Krylov variant (no QGT) ***                           ║")
    print("║  Lanczos · Krylov expm · per-param projection · Perceval       ║")
    print("║  Swaption Pricing — Quandela EPFL Hackathon 2026               ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    cfg = PhotonicAGPConfig(
        n_modes          = 8,
        n_photons        = 4,
        n_res_layers     = 3,
        ansatz_layers    = 2,
        ite_steps        = 8,
        ite_dtau         = 0.08,
        agp_krylov_order = 6,
        agp_reg          = 1e-4,
        n_heads          = 2,
        n_attn_layers    = 2,
        seq_len          = 10,
        horizon          = 1,
        n_epochs         = 20,
        lr_readout       = 5e-3,
        lr_attn          = 0.03,
        batch_attn       = 4,
        batch_size       = 16,
        n_features       = 5,
        n_fock_features  = 20,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/4] Loading swaption data…")
    X_train, y_train, X_test, y_test, transforms = load_swaption_data(
        train_path, cfg, n_pca_features=cfg.n_features,
    )
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/4] Building photonic model (AGP-Krylov)…")
    model = PhotonicAGPTransformer(cfg)

    n_ite = cfg.ansatz_n_params
    n_attn = len(model.attention.get_all_params())
    n_readout = sum(p.numel() for p in model.readout.parameters())
    print(f"      Photonic modes             : {cfg.n_modes}")
    print(f"      Photons                    : {cfg.n_photons}")
    print(f"      Reservoir layers           : {cfg.n_res_layers} (fixed, seed={cfg.reservoir_seed})")
    print(f"      ITE ansatz params          : {n_ite}  (AGP-Krylov per fwd)")
    print(f"      Krylov subspace order      : {cfg.agp_krylov_order}")
    print(f"      Attention Q,K,V params     : {n_attn}  (param-shift trained)")
    print(f"      Readout params             : {n_readout}  (Adam trained)")
    print(f"      ITE steps per token        : {cfg.ite_steps}")
    print(f"      Fock feature dim           : {cfg.n_fock_features}")
    print(f"      Feature dim after attention: {cfg.feature_dim}")
    print(f"      MerLin available           : {HAS_MERLIN}")
    print(f"      QGT used                   : NO (AGP-Krylov)\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[3/4] Training…")
    losses = train_model(model, X_train, y_train, cfg, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set…")
    preds = []
    for i in range(len(X_test)):
        preds.append(model.predict(X_test[i])[0])
    preds = np.array(preds)

    y_true = y_test[:, 0]
    y_hat = preds
    mse = float(np.mean((y_true - y_hat)**2))
    mae = float(np.mean(np.abs(y_true - y_hat)))
    ss_res = np.sum((y_true - y_hat)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    print(f"      MSE : {mse:.6f}")
    print(f"      MAE : {mae:.6f}")
    print(f"      R²  : {r2:.4f}")

    plot_results(y_true, y_hat, losses, cfg, save_path="photonic_agp_results.png")
    print("\n✓ Done.")
    return model, losses, preds, y_test


if __name__ == "__main__":
    model, losses, preds, y_test = main("train.xlsx")
