"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║         QUANTUM LINEAR ALGEBRA TRANSFORMER + QRC                               ║
║         VarQITE-SVD  ·  Quantum Geometric Tensor  ·  McLachlan Principle       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  PIPELINE                                                                       ║
║  ──────────────────────────────────────────────────────────────────────────    ║
║                                                                                 ║
║  x_t ──► QuantumEncoder ──► QuantumReservoir ──► |ψ_t⟩                        ║
║                                (fixed)               │                         ║
║                                                       ▼                        ║
║                                              VarQITE-SVD                       ║
║                                  ┌──────────────────────────────┐              ║
║                                  │  H = -Σ_t |ψ_t⟩⟨ψ_t|        │              ║
║                                  │  McLachlan: A·dθ/dτ = C      │              ║
║                                  │  A_ij = Re[QGT_ij]           │              ║
║                                  │  C_i  = -Re[⟨∂_iψ|H|ψ⟩]    │              ║
║                                  │  solve (A+λI)δθ = C          │              ║
║                                  └────────────┬─────────────────┘              ║
║                                               │ compressed |φ(θ*)⟩             ║
║                                               ▼                                ║
║                                    QuantumAttention (Q,K,V)                    ║
║                                    param-shift on ⟨Z_i⟩ features              ║
║                                               │                                ║
║                                               ▼                                ║
║                                    Classical Readout ──► x̂_{t+h}              ║
║                                                                                 ║
║  MATH                                                                           ║
║  ──────────────────────────────────────────────────────────────────────────    ║
║                                                                                 ║
║  VarQITE ground state of H = -Σ_t |ψ_t⟩⟨ψ_t| is the state that maximises     ║
║  the total overlap with all reservoir states — i.e. the dominant principal     ║
║  component of {|ψ_t⟩}. This is quantum PCA without eigendecomposition.        ║
║                                                                                 ║
║  QGT (Fubini-Study metric):                                                    ║
║    A_ij = Re[⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩]                              ║
║  Computed via parameter-shift on statevectors (exact, no sampling noise).      ║
║                                                                                 ║
║  Energy gradient (for C vector):                                               ║
║    ⟨∂_iψ|H|ψ⟩ = -Σ_t ⟨∂_iψ|ψ_t⟩⟨ψ_t|ψ⟩                                     ║
║  H applied as: H|v⟩ = -Σ_t ⟨ψ_t|v⟩·|ψ_t⟩  (rank-T operator)                 ║
║                                                                                 ║
║  Training                                                                       ║
║  ──────────────────────────────────────────────────────────────────────────    ║
║  · Reservoir params  → FIXED (QRC philosophy)                                  ║
║  · VarQITE ansatz    → Re-optimised each forward pass (τ steps)                ║
║  · Attention Q,K,V   → Parameter-shift rule                                    ║
║  · Readout           → Adam (PyTorch autograd)                                  ║
║                                                                                 ║
║  Requirements:  pip install qiskit qiskit-aer torch numpy matplotlib scipy    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from scipy.linalg import solve as scipy_solve
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QLAConfig:
    """Central configuration — every hyperparameter in one place."""

    # ── Quantum circuit ────────────────────────────────────────────────────────
    n_qubits:          int   = 4      # qubits per circuit
    n_res_layers:      int   = 3      # reservoir depth
    reservoir_seed:    int   = 42

    # ── VarQITE ───────────────────────────────────────────────────────────────
    varqite_layers:    int   = 2      # depth of VarQITE ansatz U(θ)
    varqite_steps:     int   = 12     # imaginary time steps per forward pass
    varqite_dtau:      float = 0.08   # imaginary time step size Δτ
    varqite_reg:       float = 1e-3   # Tikhonov reg λ for (A + λI) inversion
    varqite_shift:     float = np.pi / 2   # param-shift angle for QGT/C

    # ── Attention ─────────────────────────────────────────────────────────────
    n_heads:           int   = 2
    n_attn_layers:     int   = 2

    # ── Sequence ──────────────────────────────────────────────────────────────
    seq_len:           int   = 6
    horizon:           int   = 1

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs:          int   = 25
    lr_readout:        float = 5e-3
    lr_attn:           float = 0.05
    attn_shift:        float = np.pi / 2
    batch_attn:        int   = 4     # samples per epoch for quantum gradient

    @property
    def varqite_n_params(self) -> int:
        """Params in the VarQITE ansatz: RY+RZ per qubit per layer."""
        return self.n_qubits * 2 * self.varqite_layers

    @property
    def feature_dim(self) -> int:
        """Feature dimension after attention concatenation."""
        return self.n_qubits * self.n_heads

    @property
    def attn_n_params(self) -> int:
        """Trainable params per Q/K/V circuit in one head."""
        return self.n_qubits * 2 * self.n_attn_layers


# ═══════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM ENCODER
#    Angle encoding with harmonic re-uploading.
#    Each qubit i receives frequency (i+1)/n of the input signal.
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumEncoder:
    """
    Maps x ∈ [0,1] → n-qubit quantum state.

    H^⊗n  →  ∏_i [RY(x·π·(i+1)/n) · RZ(x·π/(i+1))]  →  CNOT ring

    Rationale:
      · Hadamards create uniform superposition (all basis states present)
      · Harmonic angles give each qubit a different Fourier mode of x
      · CNOT ring injects entanglement so features are non-separable
    """
    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def encode(self, x: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Enc")
        qc.h(range(self.n))
        for i in range(self.n):
            qc.ry(float(x) * np.pi * (i + 1) / self.n, i)
            qc.rz(float(x) * np.pi / (i + 1),          i)
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n - 1, 0)
        return qc


# ═══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM RESERVOIR
#    Fixed random unitary — brick-wall CZ + random U3 + long-range CX.
#    Never trained (core QRC philosophy).
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumReservoir:
    """
    Fixed random quantum reservoir circuit.

    Per layer:
        [U3(θ_i, φ_i, λ_i) on each qubit i]
        [CZ on even pairs  (brick-wall)]
        [CX(0, n-1)        (long-range)]

    The reservoir maps input-encoded states into an exponentially large
    Hilbert space, providing the rich feature bank that VarQITE then mines.
    """
    def __init__(self, n_qubits: int, n_layers: int, seed: int = 42):
        self.n = n_qubits
        rng = np.random.default_rng(seed)
        self._circuit = self._build(rng, n_layers)

    def _build(self, rng: np.random.Generator, n_layers: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Reservoir")
        for layer in range(n_layers):
            for i in range(self.n):
                θ, φ, λ = rng.uniform(0, 2 * np.pi, 3)
                qc.u(θ, φ, λ, i)
            start = layer % 2
            for i in range(start, self.n - 1, 2):
                qc.cz(i, i + 1)
            if self.n > 2:
                qc.cx(0, self.n - 1)
        return qc

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VarQITE-SVD MODULE
#    ─────────────────────────────────────────────────────────────────────────
#    THEORY:
#      Given T reservoir states {|ψ_t⟩}, define the density-like Hamiltonian:
#          H = -Σ_t |ψ_t⟩⟨ψ_t|
#
#      The ground state of H is the state |φ⟩ that maximises:
#          E(φ) = Σ_t |⟨ψ_t|φ⟩|²
#      i.e. the state with maximum overlap to the entire reservoir trajectory
#      = dominant principal component = top right singular vector of
#        the reservoir state matrix M = [|ψ_0⟩, ..., |ψ_{T-1}⟩].
#
#      VarQITE optimises a parametrised ansatz U(θ)|0⟩ by evolving under:
#          dθ_i/dτ = Σ_j A^{-1}_{ij} C_j
#
#      where:
#          A_ij = Re[⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩]   (Fubini-Study QGT)
#          C_i  = -Re[⟨∂_iψ|H|ψ⟩]                          (energy gradient)
#
#      Both computed exactly via parameter-shift on Statevector objects.
#
#      Applying H: H|v⟩ = -Σ_t ⟨ψ_t|v⟩·|ψ_t⟩   (rank-T, O(T·2^n) ops)
# ═══════════════════════════════════════════════════════════════════════════════

class VarQITESVD:
    """
    VarQITE-based SVD / quantum PCA module.

    Finds the dominant subspace of a set of reservoir statevectors by
    minimising the energy of H = -Σ_t |ψ_t⟩⟨ψ_t| via imaginary time evolution
    with the McLachlan variational principle.

    Output: expectation values ⟨Z_i⟩ of the converged ansatz state = feature vec.
    """

    def __init__(self, cfg: QLAConfig):
        self.cfg = cfg
        self.n   = cfg.n_qubits
        self.P   = cfg.varqite_n_params
        self.s   = cfg.varqite_shift
        # Pauli-Z observables
        self._obs = [
            SparsePauliOp.from_list(
                [("I" * i + "Z" + "I" * (self.n - i - 1), 1.0)]
            )
            for i in range(self.n)
        ]

    # ─── Ansatz ───────────────────────────────────────────────────────────────

    def _ansatz(self, params: np.ndarray) -> QuantumCircuit:
        """Hardware-efficient ansatz: [RY RZ CNOT ladder] × n_layers."""
        qc = QuantumCircuit(self.n)
        qc.h(range(self.n))          # Start from uniform superposition
        for layer in range(self.cfg.varqite_layers):
            for i in range(self.n):
                idx = (layer * self.n + i) * 2
                qc.ry(float(params[idx]),     i)
                qc.rz(float(params[idx + 1]), i)
            for i in range(self.n - 1):
                qc.cx(i, i + 1)
        return qc

    def _get_sv(self, params: np.ndarray) -> np.ndarray:
        """Statevector |ψ(θ)⟩ as numpy array."""
        qc = self._ansatz(params)
        return Statevector.from_instruction(qc).data   # shape (2^n,), complex

    # ─── Hamiltonian action ───────────────────────────────────────────────────

    @staticmethod
    def _apply_H(v: np.ndarray,
                 reservoir_svs: List[np.ndarray]) -> np.ndarray:
        """
        H|v⟩  =  -Σ_t ⟨ψ_t|v⟩ · |ψ_t⟩

        Exact application of the rank-T Hamiltonian. No matrix assembly needed.
        """
        result = np.zeros_like(v, dtype=complex)
        for psi_t in reservoir_svs:
            overlap = psi_t.conj() @ v          # ⟨ψ_t|v⟩  (complex scalar)
            result -= overlap * psi_t
        return result

    # ─── QGT matrix A and energy gradient C ──────────────────────────────────

    def _compute_A_C(
        self,
        params: np.ndarray,
        reservoir_svs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute:
          A_ij = Re[⟨∂_iψ|∂_jψ⟩ - ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩]   (P×P)
          C_i  = -Re[⟨∂_iψ|H|ψ⟩]                          (P,)

        Parameter-shift rule for statevector derivatives:
          ∂_i|ψ⟩ ≈ (|ψ(θ+s·eᵢ)⟩ - |ψ(θ-s·eᵢ)⟩) / (2 sin s)
          with s = π/2  →  denominator = 2.
        """
        psi     = self._get_sv(params)              # (2^n,)
        H_psi   = self._apply_H(psi, reservoir_svs) # (2^n,)

        # Compute all derivative states  ∂_i|ψ⟩  for i=0,...,P-1
        d_psi = np.zeros((self.P, len(psi)), dtype=complex)
        for i in range(self.P):
            p_plus  = params.copy(); p_plus[i]  += self.s
            p_minus = params.copy(); p_minus[i] -= self.s
            sv_plus  = self._get_sv(p_plus)
            sv_minus = self._get_sv(p_minus)
            d_psi[i] = (sv_plus - sv_minus) / (2.0 * np.sin(self.s))

        # QGT  A_ij = Re[⟨∂_iψ|∂_jψ⟩  -  ⟨∂_iψ|ψ⟩⟨ψ|∂_jψ⟩]
        # Vectorised: d_psi @ d_psi†  then subtract outer product
        inner_dd  = d_psi.conj() @ d_psi.T          # (P,P) complex
        inner_dpsi = d_psi.conj() @ psi              # (P,)  complex   ⟨∂_iψ|ψ⟩
        outer_term = np.outer(inner_dpsi, inner_dpsi.conj())  # (P,P)
        A = (inner_dd - outer_term).real             # (P,P) real

        # Energy gradient  C_i = -Re[⟨∂_iψ|H|ψ⟩]
        C = -(d_psi.conj() @ H_psi).real             # (P,)

        return A, C

    # ─── Main VarQITE loop ────────────────────────────────────────────────────

    def run(
        self,
        reservoir_svs: List[np.ndarray],
        init_params:   np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run VarQITE imaginary time evolution.

        Args:
            reservoir_svs : T statevectors from the QRC (each shape 2^n)
            init_params   : initial θ; if None → random init

        Returns:
            params_final  : converged ansatz parameters θ*
            features      : ⟨Z_i⟩ values at θ*  (shape n_qubits,)
        """
        cfg = self.cfg
        rng = np.random.default_rng(seed=int(reservoir_svs[0][0].real * 1e6) % (2**31))
        params = (init_params.copy() if init_params is not None
                  else rng.uniform(0, 2 * np.pi, self.P))

        energies = []
        for step in range(cfg.varqite_steps):
            A, C = self._compute_A_C(params, reservoir_svs)

            # Tikhonov regularisation: solve (A + λI)δθ = C
            A_reg = A + cfg.varqite_reg * np.eye(self.P)
            try:
                delta_theta = scipy_solve(A_reg, C, assume_a="pos")
            except np.linalg.LinAlgError:
                # Fallback: pseudoinverse
                delta_theta = np.linalg.lstsq(A_reg, C, rcond=None)[0]

            params = params + cfg.varqite_dtau * delta_theta

            # Track energy for diagnostics
            psi = self._get_sv(params)
            E   = sum(-abs(psi_t.conj() @ psi) ** 2 for psi_t in reservoir_svs)
            energies.append(float(E.real))

        # Extract features: ⟨Z_i⟩ = expectation value of Pauli-Z on each qubit
        psi_final = Statevector(self._get_sv(params))
        features  = np.array([
            float(psi_final.expectation_value(obs).real)
            for obs in self._obs
        ])
        return params, features

    def get_energy_landscape(
        self,
        reservoir_svs: List[np.ndarray],
        params: np.ndarray,
    ) -> float:
        """Energy at current params (for diagnostics)."""
        psi = self._get_sv(params)
        return float(sum(
            -abs(psi_t.conj() @ psi) ** 2 for psi_t in reservoir_svs
        ).real)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM ATTENTION (Q, K, V CIRCUITS)
#    Operates on the VarQITE-compressed features.
#    Q, K, V are parametrised circuits applied to the VarQITE ansatz state.
#    Inner products computed as dot products of ⟨Z_i⟩ feature vectors.
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumAttentionHead:
    """
    One quantum attention head.

    Q, K, V each apply a parametrised circuit U_{Q/K/V}(θ) to the VarQITE
    compressed state |φ(θ*)⟩, then extract ⟨Z_i⟩ as feature vectors.

    Attention score:
        a_{ij} = softmax( f_Q(i)·f_K(j)ᵀ / √d )  with causal mask
    Output:
        out_i  = Σ_j a_{ij} · f_V(j)

    Trained via parameter-shift rule on the readout loss.
    """

    def __init__(self, n_qubits: int, n_layers: int, head_id: int):
        self.n = n_qubits
        self.n_layers = n_layers
        self.P = n_qubits * 2 * n_layers
        rng = np.random.default_rng(seed=head_id * 31337 + 17)
        self.params_q = rng.uniform(0, 2 * np.pi, self.P)
        self.params_k = rng.uniform(0, 2 * np.pi, self.P)
        self.params_v = rng.uniform(0, 2 * np.pi, self.P)
        self._obs = [
            SparsePauliOp.from_list(
                [("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)]
            )
            for i in range(n_qubits)
        ]

    def _proj_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n)
        for layer in range(self.n_layers):
            for i in range(self.n):
                idx = (layer * self.n + i) * 2
                qc.ry(float(params[idx]),     i)
                qc.rz(float(params[idx + 1]), i)
            for i in range(self.n - 1):
                qc.cx(i, i + 1)
        return qc

    def _measure(self, base_sv: np.ndarray, proj_params: np.ndarray) -> np.ndarray:
        """Apply projection circuit to base statevector → ⟨Z_i⟩ feature vec."""
        proj   = self._proj_circuit(proj_params)
        state  = Statevector(base_sv).evolve(proj)
        return np.array([float(state.expectation_value(o).real) for o in self._obs])

    def project(self, base_sv: np.ndarray, which: str) -> np.ndarray:
        p = {"q": self.params_q, "k": self.params_k, "v": self.params_v}[which]
        return self._measure(base_sv, p)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.params_q, self.params_k, self.params_v])

    def set_params(self, flat: np.ndarray):
        n = self.P
        self.params_q = flat[0:n].copy()
        self.params_k = flat[n:2*n].copy()
        self.params_v = flat[2*n:3*n].copy()


class QuantumAttentionLayer:
    """
    Multi-head quantum attention over VarQITE-compressed states.

    For each head h, and each token t with compressed statevector |φ_t⟩:
        Q_t^h = U_Q^h |φ_t⟩  → ⟨Z_i⟩ features
        K_t^h = U_K^h |φ_t⟩  → ⟨Z_i⟩ features
        V_t^h = U_V^h |φ_t⟩  → ⟨Z_i⟩ features

    Causal-masked attention:
        scores_{ij} = Q_i · K_j / √d,  mask upper triangle to -∞
        α_{ij}      = softmax(scores_i)
        out_t^h     = Σ_j α_{tj} · V_j^h

    Heads concatenated → (T, n_heads × n_qubits)
    """

    def __init__(self, cfg: QLAConfig):
        self.cfg   = cfg
        self.heads = [
            QuantumAttentionHead(cfg.n_qubits, cfg.n_attn_layers, h)
            for h in range(cfg.n_heads)
        ]

    def forward(self, compressed_svs: List[np.ndarray]) -> np.ndarray:
        """
        Args:
            compressed_svs: T statevectors (each 2^n_qubits) from VarQITE
        Returns:
            (T, feature_dim) attention output
        """
        T = len(compressed_svs)
        d = self.cfg.n_qubits
        head_outs = []

        for head in self.heads:
            Qs = np.array([head.project(sv, "q") for sv in compressed_svs])  # (T, d)
            Ks = np.array([head.project(sv, "k") for sv in compressed_svs])  # (T, d)
            Vs = np.array([head.project(sv, "v") for sv in compressed_svs])  # (T, d)

            scores = (Qs @ Ks.T) / np.sqrt(d)                                # (T, T)
            mask   = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + mask
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s  = np.exp(scores)
            alpha  = exp_s / exp_s.sum(axis=-1, keepdims=True)               # (T, T)
            out    = alpha @ Vs                                               # (T, d)
            head_outs.append(out)

        return np.concatenate(head_outs, axis=-1)                            # (T, feat_dim)

    def get_all_params(self) -> np.ndarray:
        return np.concatenate([h.get_params() for h in self.heads])

    def set_all_params(self, flat: np.ndarray):
        n = len(self.heads[0].get_params())
        for i, head in enumerate(self.heads):
            head.set_params(flat[i * n: (i + 1) * n])


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FULL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class QuantumLinearAlgebraTransformer(nn.Module):
    """
    Full Quantum Linear Algebra Transformer + QRC.

    Forward pass:
        1. Encode sequence x_0,...,x_{T-1} via QuantumEncoder + QuantumReservoir
           → reservoir statevectors {|ψ_t⟩}
        2. Run VarQITE-SVD on {|ψ_t⟩}
           → per-token compressed statevectors {|φ_t(θ*)⟩}
        3. Multi-head quantum attention on {|φ_t⟩}
           → (T, feature_dim) tensor
        4. Classical readout on last token → prediction x̂_{t+h}

    Note: VarQITE runs τ_steps × 2P statevector evaluations per token per
    forward pass. This is computationally intensive but theoretically exact.
    """

    def __init__(self, cfg: QLAConfig):
        super().__init__()
        self.cfg      = cfg
        self.encoder  = QuantumEncoder(cfg.n_qubits)
        self.reservoir = QuantumReservoir(cfg.n_qubits, cfg.n_res_layers,
                                          cfg.reservoir_seed)
        self.varqite  = VarQITESVD(cfg)
        self.attention = QuantumAttentionLayer(cfg)

        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

        # Cache VarQITE init params (warm-started between calls for speed)
        self._varqite_cache: np.ndarray | None = None

    # ─── Quantum forward ──────────────────────────────────────────────────────

    def _reservoir_states(self, sequence: np.ndarray) -> List[np.ndarray]:
        """Encode + reservoir → list of statevectors."""
        states = []
        for x in sequence:
            enc  = self.encoder.encode(float(x))
            res  = self.reservoir.circuit
            circ = enc.compose(res)
            sv   = Statevector.from_instruction(circ).data   # (2^n,) complex
            states.append(sv)
        return states

    def _varqite_compress(
        self,
        reservoir_svs: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Run VarQITE independently for each token position.

        For token t: reservoir context = {|ψ_0⟩,...,|ψ_t⟩} (causal — only past).
        This gives each token access to the dominant subspace of its history.

        Returns:
            compressed_svs : list of T statevectors |φ_t(θ_t*)⟩
            final_params   : last token's converged params (for warm-start)
        """
        compressed = []
        init_p = self._varqite_cache  # warm start

        for t in range(len(reservoir_svs)):
            context = reservoir_svs[:t + 1]   # causal context
            params, _ = self.varqite.run(context, init_params=init_p)
            # Store the compressed statevector for attention
            compressed.append(self.varqite._get_sv(params))
            init_p = params  # warm-start next token from previous solution

        # Update cache
        self._varqite_cache = init_p
        return compressed, init_p

    def forward_np(
        self,
        sequence: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray, torch.Tensor]:
        """
        Full forward pass.

        Returns:
            reservoir_svs   : raw reservoir statevectors (for diagnostics)
            compressed_svs  : VarQITE-compressed statevectors
            prediction      : (horizon,) tensor
        """
        # 1. Reservoir
        res_svs = self._reservoir_states(sequence)

        # 2. VarQITE-SVD compression (causal)
        comp_svs, _ = self._varqite_compress(res_svs)

        # 3. Multi-head quantum attention
        attn_out = self.attention.forward(comp_svs)         # (T, feat_dim)
        last     = torch.tensor(attn_out[-1], dtype=torch.float32)  # last token

        # 4. Classical readout
        pred = self.readout(last)
        return res_svs, comp_svs, pred

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        _, _, pred = self.forward_np(sequence)
        return pred.numpy()

    # ─── Diagnostics ─────────────────────────────────────────────────────────

    def varqite_energy(self, sequence: np.ndarray) -> float:
        """Energy of VarQITE ground state for a sequence (lower = better)."""
        res_svs = self._reservoir_states(sequence)
        comp_svs, params = self._varqite_compress(res_svs)
        return self.varqite.get_energy_landscape(res_svs, params)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HYBRID TRAINER
#    Component        Method
#    ────────────     ─────────────────────────────────────────────────────────
#    Reservoir        FIXED
#    VarQITE ansatz   Re-optimised each forward (not externally trained)
#    Attention Q,K,V  Parameter-shift rule  (quantum natural gradient approx.)
#    Readout          Adam
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    model:    QuantumLinearAlgebraTransformer,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    cfg:      QLAConfig,
    verbose:  bool = True,
) -> List[float]:
    """
    Hybrid training loop.

    Per epoch:
      Step 1 — Full pass through training set, update readout with Adam.
      Step 2 — Stochastic parameter-shift on attention Q,K,V params
               over cfg.batch_attn random samples.
    """
    optimizer = optim.Adam(model.readout.parameters(), lr=cfg.lr_readout)
    loss_fn   = nn.MSELoss()
    losses    = []
    N         = len(X_train)
    s         = cfg.attn_shift
    rng       = np.random.default_rng(0)

    for epoch in range(cfg.n_epochs):

        # ── Step 1: Readout with Adam ─────────────────────────────────────────
        epoch_loss = 0.0
        perm = rng.permutation(N)
        for i in perm:
            seq  = X_train[i]
            tgt  = torch.tensor([y_train[i]], dtype=torch.float32)
            optimizer.zero_grad()
            _, _, pred = model.forward_np(seq)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Step 2: Parameter-shift on attention quantum params ───────────────
        q_params = model.attention.get_all_params()
        P        = len(q_params)
        q_grad   = np.zeros(P)

        sample_idx = rng.choice(N, cfg.batch_attn, replace=False)
        shift_idx  = rng.choice(P, min(P, 16), replace=False)

        for idx in sample_idx:
            seq = X_train[idx]
            tgt = float(y_train[idx])

            for j in shift_idx:
                # +shift
                p_plus = q_params.copy(); p_plus[j] += s
                model.attention.set_all_params(p_plus)
                _, _, pr_plus = model.forward_np(seq)
                loss_plus = (pr_plus.item() - tgt) ** 2

                # -shift
                p_minus = q_params.copy(); p_minus[j] -= s
                model.attention.set_all_params(p_minus)
                _, _, pr_minus = model.forward_np(seq)
                loss_minus = (pr_minus.item() - tgt) ** 2

                q_grad[j] += (loss_plus - loss_minus) / (2.0 * np.sin(s))

        # Restore and update
        model.attention.set_all_params(q_params)
        q_grad  /= len(sample_idx)
        q_params -= cfg.lr_attn * q_grad
        model.attention.set_all_params(q_params)

        avg_loss = epoch_loss / N
        losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs}"
                  f"  |  MSE: {avg_loss:.6f}"
                  f"  |  ‖∇attn‖: {np.linalg.norm(q_grad):.5f}")

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DATA
# ═══════════════════════════════════════════════════════════════════════════════

def make_dataset(
    n_samples: int,
    seq_len:   int,
    horizon:   int = 1,
    noise:     float = 0.03,
    seed:      int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-frequency sine series, normalised to [0,1]."""
    rng  = np.random.default_rng(seed)
    T    = n_samples + seq_len + horizon
    t    = np.linspace(0, 8 * np.pi, T)
    s    = (np.sin(t)
            + 0.5  * np.sin(2.3 * t + 0.5)
            + 0.25 * np.sin(5.1 * t + 1.2)
            + noise * rng.standard_normal(T))
    s    = (s - s.min()) / (s.max() - s.min() + 1e-9)
    X, y = [], []
    for i in range(n_samples):
        X.append(s[i: i + seq_len])
        y.append(s[i + seq_len + horizon - 1])
    return np.array(X), np.array(y)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    losses:  List[float],
    cfg:     QLAConfig,
    save_path: str = "qla_transformer_results.png",
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#0a0a18")
    colors = {"bg": "#12122a", "spine": "#333", "true": "#4dd0e1",
              "pred": "#ff7043", "loss": "#ce93d8", "energy": "#a5d6a7"}

    for ax in axes:
        ax.set_facecolor(colors["bg"])
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor(colors["spine"])

    # ── Panel 1: Forecast ─────────────────────────────────────────────────────
    t = np.arange(len(y_true))
    axes[0].plot(t, y_true, color=colors["true"],  lw=2,   label="Ground truth")
    axes[0].plot(t, y_pred, color=colors["pred"],  lw=2, ls="--",
                 label="QLAT+QRC forecast")
    axes[0].fill_between(t, y_true, y_pred, alpha=0.12, color=colors["pred"])
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)
    axes[0].text(0.97, 0.05,
                 f"MSE={mse:.5f}\nMAE={mae:.5f}\nR²={r2:.4f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 color="white", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="#111", ec="#555"))
    axes[0].set_title("Forecast vs Ground Truth", color="white", fontsize=12)
    axes[0].set_xlabel("Time step", color="white")
    axes[0].set_ylabel("Normalised value", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=9)

    # ── Panel 2: Training loss ─────────────────────────────────────────────────
    axes[1].plot(losses, color=colors["loss"], lw=2)
    axes[1].set_title("Training Loss (MSE)", color="white", fontsize=12)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss (log scale)", color="white")
    axes[1].set_yscale("log")

    # ── Panel 3: Architecture diagram ─────────────────────────────────────────
    ax3 = axes[2]
    ax3.axis("off")
    diagram = [
        ("Input x_t",                          0.85),
        ("↓",                                  0.77),
        ("Quantum Encoder",                    0.70),
        ("angle encoding + CNOT ring",         0.64),
        ("↓",                                  0.57),
        ("Quantum Reservoir  [FIXED]",         0.50),
        ("random U3 + brick-wall CZ",          0.44),
        ("↓",                                  0.37),
        ("VarQITE-SVD",                        0.30),
        ("H=-Σ|ψ_t⟩⟨ψ_t| · QGT · McLachlan",  0.24),
        ("↓",                                  0.17),
        ("Quantum Attention  (Q,K,V heads)",   0.10),
        ("↓",                                  0.04),
        ("Classical Readout → x̂_{t+h}",        -0.02),
    ]
    box_items  = {0, 2, 5, 8, 11, 13}
    for i, (label, y_pos) in enumerate(diagram):
        weight = "bold" if i in {0, 2, 5, 8, 11, 13} else "normal"
        color_ = ("#4dd0e1" if i == 0 else
                  "#a5d6a7" if i in {5, 6} else
                  "#ce93d8" if i in {8, 9} else
                  "#ff7043" if i in {11, 12} else
                  "#fff59d" if i == 13 else "white")
        ax3.text(0.5, y_pos, label, ha="center", va="center",
                 color=color_, fontsize=8.5, fontweight=weight,
                 fontfamily="monospace",
                 transform=ax3.transAxes)

    ax3.set_title("Model Architecture", color="white", fontsize=12)

    arch = (f"n_qubits={cfg.n_qubits}  |  res_layers={cfg.n_res_layers}  |  "
            f"varqite_steps={cfg.varqite_steps}  |  heads={cfg.n_heads}")
    fig.suptitle(f"Quantum Linear Algebra Transformer + QRC\n{arch}",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Quantum Linear Algebra Transformer + QRC                       ║")
    print("║  VarQITE-SVD  ·  QGT  ·  McLachlan  ·  Param-Shift  ·  Adam   ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    cfg = QLAConfig(
        n_qubits        = 4,
        n_res_layers    = 3,
        varqite_layers  = 2,
        varqite_steps   = 12,
        varqite_dtau    = 0.08,
        varqite_reg     = 1e-3,
        n_heads         = 2,
        n_attn_layers   = 2,
        seq_len         = 6,
        horizon         = 1,
        n_epochs        = 25,
        lr_readout      = 5e-3,
        lr_attn         = 0.05,
        batch_attn      = 4,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/4] Generating dataset…")
    X_train, y_train = make_dataset(40, cfg.seq_len, cfg.horizon, seed=0)
    X_test,  y_test  = make_dataset(15, cfg.seq_len, cfg.horizon, seed=99)
    print(f"      Train {X_train.shape}  |  Test {X_test.shape}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/4] Building model…")
    model = QuantumLinearAlgebraTransformer(cfg)
    n_varqite = cfg.varqite_n_params
    n_attn    = len(model.attention.get_all_params())
    n_readout = sum(p.numel() for p in model.readout.parameters())
    print(f"      VarQITE ansatz params      : {n_varqite}  (re-optimised each fwd)")
    print(f"      Reservoir params           : —  (fixed, seed={cfg.reservoir_seed})")
    print(f"      Attention Q,K,V params     : {n_attn}  (param-shift trained)")
    print(f"      Readout params             : {n_readout}  (Adam trained)")
    print(f"      VarQITE steps per token    : {cfg.varqite_steps}")
    print(f"      QGT size per step          : {n_varqite}×{n_varqite}")
    print(f"      Feature dim after attention: {cfg.feature_dim}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[3/4] Training (VarQITE warm-start + param-shift attention + Adam)…")
    losses = train(model, X_train, y_train, cfg, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set…")
    preds = np.array([model.predict(X_test[i])[0] for i in range(len(X_test))])
    mse   = float(np.mean((preds - y_test)**2))
    mae   = float(np.mean(np.abs(preds - y_test)))
    r2    = float(1 - np.sum((y_test - preds)**2) /
                      np.sum((y_test - y_test.mean())**2))
    print(f"      MSE : {mse:.6f}")
    print(f"      MAE : {mae:.6f}")
    print(f"      R²  : {r2:.4f}")

    plot_results(y_test, preds, losses, cfg,
                 save_path="qla_transformer_results.png")
    print("\n✓ Done.")
    return model, losses, preds, y_test


if __name__ == "__main__":
    model, losses, preds, y_test = main()
