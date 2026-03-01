"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║       QUANTUM TRANSFORMER + PHOTONIC QRC                                       ║
║       Boson Sampling Reservoir  ·  MZI Attention  ·  Perceval / MerLin        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  TRANSLATION FROM QISKIT → PERCEVAL                                            ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║                                                                                 ║
║  Qiskit concept          │  Perceval / photonic equivalent                     ║
║  ─────────────────────── │ ──────────────────────────────────────────────────  ║
║  |0⟩, |1⟩  qubit        │  |n₁,n₂,...,nₘ⟩  Fock state (mode occupancies)    ║
║  RY(θ), RZ(φ)  gate     │  PS(φ) phase shifter + BS(θ) beam splitter         ║
║  CNOT  gate              │  No direct equiv. — use post-selected KLM           ║
║  Random U3 reservoir     │  Haar-random unitary U(m) → Boson Sampling         ║
║  Expectation ⟨Z_i⟩       │  Output photon probability distribution p(s)       ║
║  Param-shift on θ        │  Param-shift on φ of PS (same math, same shift)    ║
║                                                                                 ║
║  ARCHITECTURE                                                                   ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║                                                                                 ║
║  x_t ──► PhotonicEncoder ──► BosonSamplingReservoir ──► feature_t             ║
║          PS(x·π·i/m) +        Haar-random U(m)           prob. vector         ║
║          MZI entangler         |1,1,...,1,0,...,0⟩         p(s) ∈ ℝᵈ          ║
║                                                                  │              ║
║                                                ┌─────────────────┘              ║
║                                                ▼                                ║
║                                     PhotonicAttention (Q,K,V)                  ║
║                                     MZI mesh per projection                    ║
║                                     causal-masked softmax                       ║
║                                                │                                ║
║                                                ▼                                ║
║                                     Classical Readout → x̂_{t+h}               ║
║                                                                                 ║
║  WHY BOSON SAMPLING FOR QRC                                                     ║
║  ─────────────────────────────────────────────────────────────────────────────  ║
║  · A Haar-random m-mode interferometer + k photons samples from a              ║
║    distribution that is #P-hard to simulate classically for m~50              ║
║  · The output distribution p(s) lives in a C(m+k-1,k)-dimensional space —     ║
║    exponentially richer than the m-dimensional qubit expectation vector        ║
║  · No entangling gates needed: linear optics + single photons suffices         ║
║  · Directly implementable on Quandela's photonic chip                          ║
║                                                                                 ║
║  Requirements:                                                                  ║
║    pip install perceval-quandela torch numpy matplotlib scipy                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import perceval as pcvl
from perceval.components import Unitary, PS, BS
from scipy.stats import unitary_group
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicConfig:
    """Central configuration for the Photonic Quantum Transformer."""

    # ── Photonic circuit ───────────────────────────────────────────────────────
    n_modes:       int   = 6      # optical modes  m
    n_photons:     int   = 2      # input photons  k  → features = C(m+k-1,k)
    reservoir_seed: int  = 42     # fixed random seed for Haar unitary

    # ── Attention ─────────────────────────────────────────────────────────────
    n_heads:       int   = 2      # attention heads
    n_attn_layers: int   = 2      # MZI layers in Q/K/V projection circuit

    # ── Sequence ──────────────────────────────────────────────────────────────
    seq_len:  int = 6
    horizon:  int = 1

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs:     int   = 25
    lr_readout:   float = 5e-3
    lr_attn:      float = 0.05
    param_shift:  float = np.pi / 2
    batch_attn:   int   = 4

    @property
    def feature_dim_per_head(self) -> int:
        """Dimension of the Boson Sampling output distribution.
        = number of k-photon Fock states in m modes = C(m+k-1, k)
        """
        from math import comb
        return comb(self.n_modes + self.n_photons - 1, self.n_photons)

    @property
    def feature_dim(self) -> int:
        return self.feature_dim_per_head * self.n_heads

    @property
    def n_attn_params_per_proj(self) -> int:
        """PS parameters per Q/K/V projection: n_modes × n_attn_layers."""
        return self.n_modes * self.n_attn_layers

    @property
    def input_fock_state(self) -> pcvl.BasicState:
        """Standard input: k photons in first k modes, rest vacuum."""
        occ = [1] * self.n_photons + [0] * (self.n_modes - self.n_photons)
        return pcvl.BasicState(occ)


# ══════════════════════════════════════════════════════════════════════════════
# UTILS: Fock state enumeration and probability extraction
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_fock_states(n_modes: int, n_photons: int) -> List[pcvl.BasicState]:
    """
    Enumerate all Fock states with exactly n_photons in n_modes.
    Count = C(n_modes + n_photons - 1, n_photons).

    Example: n_modes=3, n_photons=2 →
        |2,0,0⟩, |1,1,0⟩, |1,0,1⟩, |0,2,0⟩, |0,1,1⟩, |0,0,2⟩
    """
    states = []
    # Generate all multisets of size n_photons from range(n_modes)
    for combo in combinations_with_replacement(range(n_modes), n_photons):
        occ = [0] * n_modes
        for idx in combo:
            occ[idx] += 1
        states.append(pcvl.BasicState(occ))
    return states


def extract_prob_vector(
    circuit:       pcvl.Circuit,
    input_state:   pcvl.BasicState,
    output_states: List[pcvl.BasicState],
) -> np.ndarray:
    """
    Run exact Boson Sampling simulation and return probability vector.

    Uses Perceval's SLOS (Sparse Linear Optical Simulator) backend for
    exact permanent computation. This is the gold standard for small m.

    Args:
        circuit       : Perceval circuit (m modes)
        input_state   : Fock input |k photons⟩
        output_states : list of output Fock states to query

    Returns:
        prob_vec : (len(output_states),) probability vector, sums to ≤1
                   (≤1 because we don't query all states in general)
    """
    proc = pcvl.Processor("SLOS", circuit)
    proc.with_input(input_state)
    # Use Analyzer for exact probabilities over a specific state list
    analyzer = pcvl.algorithm.Analyzer(
        proc,
        input_states  = [input_state],
        output_states = output_states,
    )
    analyzer.compute(normalize=True)
    # Extract row corresponding to our input state
    probs = np.array([
        float(analyzer.distribution[input_state].get(s, 0.0))
        for s in output_states
    ])
    # Numerical safety: clip negatives from floating point
    probs = np.clip(probs, 0.0, None)
    # Renormalise to sum=1 (may not sum to 1 if output_states is a subset)
    total = probs.sum()
    if total > 1e-12:
        probs /= total
    return probs


# ══════════════════════════════════════════════════════════════════════════════
# 1. PHOTONIC ENCODER
#    Encodes scalar x ∈ [0,1] into an m-mode photonic state by applying
#    phase shifts before the reservoir.
#
#    Circuit structure:
#        PS(x·π·(i+1)/m) on each mode i        ← harmonic phase diversity
#        followed by a shallow MZI entangler    ← creates input correlations
#
#    Why phase shifts for encoding:
#      A PS(φ) on mode i multiplies amplitude aᵢ† → e^{iφ} aᵢ†.
#      Different harmonics give each mode a different frequency of x,
#      analogous to qubit angle encoding.
# ══════════════════════════════════════════════════════════════════════════════

class PhotonicEncoder:
    """
    Encodes x ∈ [0,1] into phase-modulated input modes.

    The encoder circuit is prepended to the reservoir circuit.
    """
    def __init__(self, n_modes: int):
        self.m = n_modes

    def build(self, x: float) -> pcvl.Circuit:
        """Build encoding circuit for value x."""
        qc = pcvl.Circuit(self.m, name="Encoder")
        # Harmonic phase shifts — data re-uploading in photonic form
        for i in range(self.m):
            phi = float(x) * np.pi * (i + 1) / self.m
            qc.add(i, PS(phi))
        # Shallow MZI entangler — couple modes pairwise
        for i in range(0, self.m - 1, 2):
            qc.add((i, i + 1), BS())   # 50/50 beam splitter
        if self.m > 2:
            for i in range(1, self.m - 1, 2):
                qc.add((i, i + 1), BS())
        return qc


# ══════════════════════════════════════════════════════════════════════════════
# 2. BOSON SAMPLING RESERVOIR
#    Fixed Haar-random unitary interferometer acting as the nonlinear reservoir.
#
#    WHY THIS IS POWERFUL:
#      · A Haar-random unitary on m modes mixes all input modes completely
#      · With k photons, the output distribution lives in C(m+k-1,k) dimensions
#      · For m=6, k=2: 21-dimensional feature space from a 6-mode circuit
#      · The permanent of submatrices of U governs the probabilities —
#        this is #P-hard to compute classically for large m
#      · The reservoir is FIXED: never trained (QRC philosophy)
# ══════════════════════════════════════════════════════════════════════════════

class BosonSamplingReservoir:
    """
    Fixed Haar-random unitary interferometer for photonic QRC.

    The reservoir maps an encoded m-mode Fock state through a random
    unitary U ∈ U(m), producing an output probability distribution
    that serves as the feature vector.
    """
    def __init__(self, n_modes: int, seed: int = 42):
        self.m = n_modes
        rng    = np.random.default_rng(seed)
        # Sample Haar-random unitary via scipy
        self._U = unitary_group.rvs(n_modes, random_state=int(rng.integers(0, 2**31)))
        self._circuit = self._build()

    def _build(self) -> pcvl.Circuit:
        """Build Perceval circuit from the Haar-random unitary matrix."""
        qc = pcvl.Circuit(self.m, name="BosonReservoir")
        qc.add(0, Unitary(pcvl.Matrix(self._U)))
        return qc

    @property
    def circuit(self) -> pcvl.Circuit:
        return self._circuit.copy()

    def unitary(self) -> np.ndarray:
        return self._U.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 3. PHOTONIC ATTENTION HEAD
#    Each head has 3 parametrized circuits: Q, K, V.
#    Each projection circuit is a MZI mesh (PS + BS ladder).
#    Features = output photon probability distribution after projection.
#
#    PARAMETER-SHIFT FOR PHOTONIC CIRCUITS:
#      A PS(φ) gate contributes e^{iφ} to the unitary.
#      Shifting φ → φ ± π/2 gives:
#          ∂p/∂φ = (p(φ+π/2) - p(φ-π/2)) / (2 sin(π/2)) = (p+ - p-) / 2
#      This is IDENTICAL to the qubit parameter-shift rule.
# ══════════════════════════════════════════════════════════════════════════════

class PhotonicAttentionHead:
    """
    One multi-head photonic attention head.

    Q, K, V projections are MZI meshes (n_modes PS + BS per layer).
    Features are output Boson Sampling probability vectors.
    """

    def __init__(self, cfg: PhotonicConfig, head_id: int):
        self.cfg     = cfg
        self.m       = cfg.n_modes
        self.k       = cfg.n_photons
        self.P       = cfg.n_attn_params_per_proj
        self.n_layers = cfg.n_attn_layers
        self.head_id  = head_id

        # Pre-enumerate output Fock states (same for all projections)
        self._out_states = enumerate_fock_states(self.m, self.k)
        self._input_state = cfg.input_fock_state

        # Initialise params
        rng = np.random.default_rng(seed=head_id * 31337 + 17)
        self.params_q = rng.uniform(0, 2 * np.pi, self.P)
        self.params_k = rng.uniform(0, 2 * np.pi, self.P)
        self.params_v = rng.uniform(0, 2 * np.pi, self.P)

    # ── Build projection circuit ──────────────────────────────────────────────

    def _build_proj(self, params: np.ndarray) -> pcvl.Circuit:
        """
        Build MZI projection circuit from phase parameters.

        Structure per layer:
          PS(φᵢ) on each mode i    → local phase diversity
          BS on alternating pairs  → mode mixing
        """
        qc = pcvl.Circuit(self.m)
        for layer in range(self.n_layers):
            # Phase shifters
            for i in range(self.m):
                phi = float(params[layer * self.m + i])
                qc.add(i, PS(phi))
            # Beam splitters — brick wall
            start = layer % 2
            for i in range(start, self.m - 1, 2):
                qc.add((i, i + 1), BS())
        return qc

    # ── Measure features ──────────────────────────────────────────────────────

    def _measure(self, reservoir_circuit: pcvl.Circuit,
                 proj_params: np.ndarray) -> np.ndarray:
        """
        Apply encoder state → reservoir → projection → measure probs.

        Note: reservoir_circuit already contains encoder + reservoir.
        We compose it with the projection circuit.
        """
        proj    = self._build_proj(proj_params)
        full    = reservoir_circuit.copy() // proj
        probs   = extract_prob_vector(full, self._input_state, self._out_states)
        return probs

    def project(self, reservoir_circuit: pcvl.Circuit, which: str) -> np.ndarray:
        """Project reservoir state with Q / K / V circuit."""
        p = {"q": self.params_q, "k": self.params_k, "v": self.params_v}[which]
        return self._measure(reservoir_circuit, p)

    def project_with_params(self, reservoir_circuit: pcvl.Circuit,
                             params: np.ndarray) -> np.ndarray:
        return self._measure(reservoir_circuit, params)

    # ── Param management ──────────────────────────────────────────────────────

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.params_q, self.params_k, self.params_v])

    def set_params(self, flat: np.ndarray):
        n = self.P
        self.params_q = flat[0:n].copy()
        self.params_k = flat[n:2*n].copy()
        self.params_v = flat[2*n:3*n].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 4. PHOTONIC ATTENTION LAYER
#    Multi-head causal attention over Boson Sampling feature vectors.
# ══════════════════════════════════════════════════════════════════════════════

class PhotonicAttentionLayer:
    """
    Multi-head photonic attention.

    For each head h and each token t:
        reservoir_circuit_t = encoder(x_t) ∘ U_reservoir

        Q_t^h = prob_vector( reservoir_t → U_Q^h )
        K_t^h = prob_vector( reservoir_t → U_K^h )
        V_t^h = prob_vector( reservoir_t → U_V^h )

    Causal-masked attention scores:
        score_{ij} = Q_i · K_j / √d    (dot product of prob vectors)
        α_{ij}     = softmax(scores_i)  (causal masked)
        out_t^h    = Σ_j α_{tj} · V_j^h

    Heads concatenated → (T, feature_dim) output.
    """

    def __init__(self, cfg: PhotonicConfig):
        self.cfg   = cfg
        self.heads = [PhotonicAttentionHead(cfg, h) for h in range(cfg.n_heads)]

    def forward(self, reservoir_circuits: List[pcvl.Circuit]) -> np.ndarray:
        T = len(reservoir_circuits)
        d = self.cfg.feature_dim_per_head
        head_outs = []

        for head in self.heads:
            Qs = np.array([head.project(rc, "q") for rc in reservoir_circuits])  # (T,d)
            Ks = np.array([head.project(rc, "k") for rc in reservoir_circuits])  # (T,d)
            Vs = np.array([head.project(rc, "v") for rc in reservoir_circuits])  # (T,d)

            scores = (Qs @ Ks.T) / np.sqrt(d)                                   # (T,T)
            mask   = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + mask
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s  = np.exp(scores)
            alpha  = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-12)        # (T,T)
            out    = alpha @ Vs                                                  # (T,d)
            head_outs.append(out)

        return np.concatenate(head_outs, axis=-1)                               # (T,feat_dim)

    def get_all_params(self) -> np.ndarray:
        return np.concatenate([h.get_params() for h in self.heads])

    def set_all_params(self, flat: np.ndarray):
        n = len(self.heads[0].get_params())
        for i, h in enumerate(self.heads):
            h.set_params(flat[i * n: (i + 1) * n])


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL MODEL: PhotonicQuantumTransformer
# ══════════════════════════════════════════════════════════════════════════════

class PhotonicQuantumTransformer(nn.Module):
    """
    Full Photonic Quantum Transformer for time series forecasting.

    Pipeline:
        1. For each x_t in sequence:
               encoder(x_t) ∘ U_reservoir → reservoir_circuit_t
        2. Multi-head photonic attention over {reservoir_circuit_t}
               → (T, feature_dim) attention output
        3. Classical readout on last token → x̂_{t+h}

    Photonic QRC:
        The reservoir circuit is FIXED (Haar-random unitary).
        The encoder adds x-dependent phase shifts before U_reservoir.
        Features = output photon probability distribution p(s) ∈ ℝ^{C(m+k-1,k)}.
    """

    def __init__(self, cfg: PhotonicConfig):
        super().__init__()
        self.cfg       = cfg
        self.encoder   = PhotonicEncoder(cfg.n_modes)
        self.reservoir = BosonSamplingReservoir(cfg.n_modes, cfg.reservoir_seed)
        self.attention = PhotonicAttentionLayer(cfg)

        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

    # ── Build reservoir circuits ───────────────────────────────────────────────

    def _reservoir_circuits(self, sequence: np.ndarray) -> List[pcvl.Circuit]:
        """
        For each x_t, build: encoder(x_t) ∘ U_reservoir

        Returns list of T Perceval circuits ready for Boson Sampling.
        """
        circuits = []
        for x in sequence:
            enc   = self.encoder.build(float(x))
            res   = self.reservoir.circuit
            full  = enc // res    # Perceval composition operator
            circuits.append(full)
        return circuits

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward_np(self, sequence: np.ndarray) -> Tuple[List[pcvl.Circuit], torch.Tensor]:
        """
        Args:
            sequence: (T,) normalised time series window
        Returns:
            reservoir_circuits : list of T composed circuits (for diagnostics)
            prediction         : (horizon,) tensor
        """
        res_circuits = self._reservoir_circuits(sequence)
        attn_out     = self.attention.forward(res_circuits)        # (T, feat_dim)
        last         = torch.tensor(attn_out[-1], dtype=torch.float32)
        pred         = self.readout(last)
        return res_circuits, pred

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        _, pred = self.forward_np(sequence)
        return pred.numpy()

    # ── Reservoir features (diagnostics) ──────────────────────────────────────

    def get_reservoir_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Raw Boson Sampling output probabilities for each token.
        Returns (T, C(m+k-1,k)) matrix — the reservoir feature bank.
        """
        out_states   = enumerate_fock_states(self.cfg.n_modes, self.cfg.n_photons)
        input_state  = self.cfg.input_fock_state
        circuits     = self._reservoir_circuits(sequence)
        features     = []
        for circ in circuits:
            p = extract_prob_vector(circ, input_state, out_states)
            features.append(p)
        return np.array(features)


# ══════════════════════════════════════════════════════════════════════════════
# 6. HYBRID TRAINER
#    Reservoir      → FIXED
#    Attention Q,K,V → Parameter-shift rule on PS phases
#    Readout        → Adam
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model:    PhotonicQuantumTransformer,
    X_train:  np.ndarray,
    y_train:  np.ndarray,
    cfg:      PhotonicConfig,
    verbose:  bool = True,
) -> List[float]:
    """
    Hybrid photonic training loop.

    PARAMETER-SHIFT FOR PS GATES:
        ∂L/∂φⱼ = (L(φⱼ+π/2) - L(φⱼ-π/2)) / (2·sin(π/2))
                = (L+ - L-) / 2

    This is exact for any PS gate in the circuit (no approximation).
    """
    optimizer = optim.Adam(model.readout.parameters(), lr=cfg.lr_readout)
    loss_fn   = nn.MSELoss()
    losses    = []
    N         = len(X_train)
    s         = cfg.param_shift
    rng       = np.random.default_rng(0)

    for epoch in range(cfg.n_epochs):

        # ── Step 1: Readout — Adam ─────────────────────────────────────────────
        epoch_loss = 0.0
        perm = rng.permutation(N)
        for i in perm:
            seq  = X_train[i]
            tgt  = torch.tensor([y_train[i]], dtype=torch.float32)
            optimizer.zero_grad()
            _, pred = model.forward_np(seq)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Step 2: Attention phases — Parameter-shift ─────────────────────────
        q_params   = model.attention.get_all_params()
        P          = len(q_params)
        q_grad     = np.zeros(P)
        sample_idx = rng.choice(N, cfg.batch_attn, replace=False)
        shift_idx  = rng.choice(P, min(P, 16), replace=False)

        for idx in sample_idx:
            seq = X_train[idx]
            tgt = float(y_train[idx])
            for j in shift_idx:
                p_plus = q_params.copy(); p_plus[j] += s
                model.attention.set_all_params(p_plus)
                _, pr_plus  = model.forward_np(seq)
                l_plus  = (pr_plus.item() - tgt) ** 2

                p_minus = q_params.copy(); p_minus[j] -= s
                model.attention.set_all_params(p_minus)
                _, pr_minus = model.forward_np(seq)
                l_minus = (pr_minus.item() - tgt) ** 2

                q_grad[j] += (l_plus - l_minus) / (2.0 * np.sin(s))

        # Restore + update
        model.attention.set_all_params(q_params)
        q_grad  /= max(len(sample_idx), 1)
        q_params -= cfg.lr_attn * q_grad
        model.attention.set_all_params(q_params)

        avg_loss = epoch_loss / N
        losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs}"
                  f"  |  MSE: {avg_loss:.6f}"
                  f"  |  ‖∇attn‖: {np.linalg.norm(q_grad):.5f}")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# 7. DATA
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(
    n_samples: int,
    seq_len:   int,
    horizon:   int = 1,
    noise:     float = 0.03,
    seed:      int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
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


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    losses:  List[float],
    res_features: np.ndarray,
    cfg:     PhotonicConfig,
    save_path: str = "photonic_qt_qrc_results.png",
):
    """
    4-panel figure:
      1. Forecast vs ground truth
      2. Training loss
      3. Boson Sampling reservoir feature heatmap (T × feature_dim)
      4. Architecture schematic
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#080818")
    bg    = "#10102a"
    spine = "#333"
    for ax in axes:
        ax.set_facecolor(bg)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor(spine)

    # ── Panel 1: Forecast ──────────────────────────────────────────────────────
    t = np.arange(len(y_true))
    axes[0].plot(t, y_true, color="#4dd0e1", lw=2, label="Ground truth")
    axes[0].plot(t, y_pred, color="#ff7043", lw=2, ls="--", label="Photonic QT-QRC")
    axes[0].fill_between(t, y_true, y_pred, alpha=0.12, color="#ff7043")
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2  = 1 - np.sum((y_true-y_pred)**2) / (np.sum((y_true-y_true.mean())**2)+1e-9)
    axes[0].text(0.97, 0.05,
                 f"MSE={mse:.5f}\nMAE={mae:.5f}\nR²={r2:.4f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 color="white", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="#111", ec="#555"))
    axes[0].set_title("Forecast vs Ground Truth", color="white", fontsize=11)
    axes[0].set_xlabel("Time step", color="white")
    axes[0].set_ylabel("Normalised value", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=8)

    # ── Panel 2: Loss ──────────────────────────────────────────────────────────
    axes[1].plot(losses, color="#ce93d8", lw=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Training Loss (MSE)", color="white", fontsize=11)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss (log)", color="white")

    # ── Panel 3: Reservoir feature heatmap ────────────────────────────────────
    from math import comb
    d = comb(cfg.n_modes + cfg.n_photons - 1, cfg.n_photons)
    if res_features.shape[1] > 0:
        im = axes[2].imshow(res_features.T, aspect="auto", cmap="plasma",
                            interpolation="nearest")
        plt.colorbar(im, ax=axes[2], label="Probability")
    axes[2].set_title(f"Boson Sampling Reservoir\nfeatures ({d} Fock states)",
                      color="white", fontsize=10)
    axes[2].set_xlabel("Token (time step)", color="white")
    axes[2].set_ylabel("Fock state index", color="white")
    axes[2].tick_params(labelcolor="white")

    # ── Panel 4: Architecture schematic ───────────────────────────────────────
    axes[3].axis("off")
    lines = [
        ("x_t",                              0.92, "#4dd0e1", True),
        ("↓",                                0.85, "white",   False),
        ("PhotonicEncoder",                  0.79, "#80cbc4", True),
        ("PS(x·π·i/m) + MZI entangler",     0.73, "#80cbc4", False),
        ("↓",                                0.66, "white",   False),
        ("Boson Sampling Reservoir [FIXED]", 0.60, "#a5d6a7", True),
        (f"Haar U({cfg.n_modes}) · |{cfg.n_photons} photons⟩",
                                             0.54, "#a5d6a7", False),
        (f"features ∈ ℝ^{d}  (#P-hard)",    0.48, "#a5d6a7", False),
        ("↓",                                0.41, "white",   False),
        ("Photonic Attention (Q,K,V)",       0.35, "#ffcc80", True),
        ("PS phase-shift + BS MZI mesh",     0.29, "#ffcc80", False),
        ("causal-masked softmax",            0.23, "#ffcc80", False),
        ("↓",                                0.16, "white",   False),
        ("Classical Readout → x̂_{t+h}",      0.10, "#ff8a65", True),
    ]
    for text, y, color, bold in lines:
        axes[3].text(0.5, y, text, ha="center", va="center",
                     color=color, fontsize=8,
                     fontweight="bold" if bold else "normal",
                     fontfamily="monospace",
                     transform=axes[3].transAxes)
    axes[3].set_title("Photonic Architecture", color="white", fontsize=11)

    arch = (f"modes={cfg.n_modes}  |  photons={cfg.n_photons}  |  "
            f"heads={cfg.n_heads}  |  features={cfg.feature_dim}")
    fig.suptitle(f"Photonic Quantum Transformer + Boson Sampling QRC  (Perceval / MerLin)\n{arch}",
                 color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from math import comb
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║  Photonic Quantum Transformer + Boson Sampling QRC                  ║")
    print("║  Perceval / MerLin (Quandela)   ·   SLOS exact simulation          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    cfg = PhotonicConfig(
        n_modes        = 6,
        n_photons      = 2,
        reservoir_seed = 42,
        n_heads        = 2,
        n_attn_layers  = 2,
        seq_len        = 6,
        horizon        = 1,
        n_epochs       = 25,
        lr_readout     = 5e-3,
        lr_attn        = 0.05,
        batch_attn     = 4,
    )
    feat_dim = comb(cfg.n_modes + cfg.n_photons - 1, cfg.n_photons)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/5] Generating dataset…")
    X_train, y_train = make_dataset(40, cfg.seq_len, cfg.horizon, seed=0)
    X_test,  y_test  = make_dataset(15, cfg.seq_len, cfg.horizon, seed=99)
    print(f"      Train {X_train.shape}  |  Test {X_test.shape}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/5] Building model…")
    model = PhotonicQuantumTransformer(cfg)
    n_attn    = len(model.attention.get_all_params())
    n_readout = sum(p.numel() for p in model.readout.parameters())
    print(f"      Modes: {cfg.n_modes}  |  Photons: {cfg.n_photons}")
    print(f"      Boson Sampling features per token : {feat_dim}  (C({cfg.n_modes}+{cfg.n_photons}-1,{cfg.n_photons}))")
    print(f"      Reservoir params                  : —  (Haar U({cfg.n_modes}), fixed)")
    print(f"      Attention phase params (trainable): {n_attn}  (PS phases)")
    print(f"      Readout params                    : {n_readout}  (Adam)")
    print(f"      Total feature dim after attention : {cfg.feature_dim}\n")

    # ── Reservoir features (diagnostics) ──────────────────────────────────────
    print("[3/5] Computing reservoir features for first training sample…")
    res_feat_sample = model.get_reservoir_features(X_train[0])
    print(f"      Reservoir feature matrix shape: {res_feat_sample.shape}")
    print(f"      Feature range: [{res_feat_sample.min():.4f}, {res_feat_sample.max():.4f}]")
    print(f"      Row sums (should be ≈1.0): {res_feat_sample.sum(axis=1).round(4)}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[4/5] Training (Boson Sampling reservoir + param-shift MZI attention + Adam)…")
    losses = train(model, X_train, y_train, cfg, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[5/5] Evaluating on test set…")
    preds = np.array([model.predict(X_test[i])[0] for i in range(len(X_test))])
    mse   = float(np.mean((preds - y_test)**2))
    mae   = float(np.mean(np.abs(preds - y_test)))
    r2    = float(1 - np.sum((y_test-preds)**2) /
                      (np.sum((y_test-y_test.mean())**2) + 1e-9))
    print(f"      MSE : {mse:.6f}")
    print(f"      MAE : {mae:.6f}")
    print(f"      R²  : {r2:.4f}")

    # Reservoir features for all test samples (for heatmap)
    res_feat_test = model.get_reservoir_features(X_test[0])

    plot_results(y_test, preds, losses, res_feat_test, cfg,
                 save_path="photonic_qt_qrc_results.png")
    print("\n✓ Done.\n")
    print("Next step: replace Processor('SLOS', ...) with Processor('Quandela', ...)")
    print("for execution on Quandela's photonic hardware via the MerLin cloud API.")
    return model, losses, preds, y_test


if __name__ == "__main__":
    model, losses, preds, y_test = main()
