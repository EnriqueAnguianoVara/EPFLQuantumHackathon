"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║   PHOTONIC QLA-TRANSFORMER + BOSON SAMPLING RESERVOIR                          ║
║   Adapted for Quandela Perceval / MerLin                                       ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                 ║
║  PIPELINE (Photonic)                                                            ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  x_t ──► PhotonicEncoder ──► BosonSamplingReservoir ──► P(n₁,...,nₘ)          ║
║           (angle-encoded       (fixed random                │                  ║
║            phase shifters)      interferometer)              ▼                  ║
║                                                    ClassicalVarQITE-SVD        ║
║                                       ┌────────────────────────────────┐       ║
║                                       │  H = -Σ_t |f_t⟩⟨f_t|          │       ║
║                                       │  McLachlan on feature vectors  │       ║
║                                       │  A_ij = Re[QGT_ij] (finite    │       ║
║                                       │    difference on photonic     │       ║
║                                       │    circuit outputs)            │       ║
║                                       │  solve (A+λI)δθ = C            │       ║
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
║  KEY ADAPTATIONS FROM QUBIT → PHOTONIC                                         ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  • Qiskit gates (RY, RZ, CX, CZ)  →  Perceval (BS, PS, GenericInterferometer)║
║  • Pauli-Z expectation ⟨Z_i⟩      →  Mode occupation probabilities P(n_i)     ║
║  • Statevector simulation          →  Perceval probability computation         ║
║  • Qubit reservoir (random U3+CZ)  →  Boson sampling reservoir                ║
║    (fixed random interferometer with photon-count histograms)                  ║
║  • VarQITE on statevectors         →  VarQITE on photonic output              ║
║    distributions (feature vectors from Fock probabilities)                     ║
║  • Quantum attention circuits      →  Perceval trainable interferometers       ║
║    with QuantumLayer + LexGrouping                                             ║
║                                                                                 ║
║  MATH (unchanged principle)                                                     ║
║  ──────────────────────────────────────────────────────────────────────────     ║
║                                                                                 ║
║  VarQITE finds the dominant principal component of the reservoir feature       ║
║  trajectory {f_t} by evolving a parametrised photonic ansatz under:            ║
║      dθ_i/dτ = Σ_j A^{-1}_{ij} C_j                                           ║
║  where A is the QGT (Fubini-Study metric) and C is the energy gradient,       ║
║  both computed via finite-difference / parameter-shift on the photonic         ║
║  circuit output probabilities.                                                  ║
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
from scipy.linalg import solve as scipy_solve
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
from merlin import LexGrouping, QuantumLayer, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder
HAS_MERLIN = True


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhotonicQLAConfig:
    """Central configuration for the Photonic QLA-Transformer."""

    # ── Photonic circuit ───────────────────────────────────────────────────────
    n_modes:           int   = 8      # optical modes
    n_photons:         int   = 4      # photons injected
    n_res_layers:      int   = 3      # reservoir interferometer depth
    reservoir_seed:    int   = 42

    # ── VarQITE (on feature vectors from photonic outputs) ────────────────────
    varqite_layers:    int   = 2      # depth of VarQITE ansatz interferometer
    varqite_steps:     int   = 8      # imaginary time steps per forward pass
    varqite_dtau:      float = 0.08   # imaginary time step Δτ
    varqite_reg:       float = 1e-3   # Tikhonov λ
    varqite_shift:     float = np.pi / 2

    # ── Attention ─────────────────────────────────────────────────────────────
    n_heads:           int   = 2
    n_attn_layers:     int   = 2

    # ── Sequence (for swaption time series) ───────────────────────────────────
    seq_len:           int   = 10     # lookback window (trading days)
    horizon:           int   = 1      # prediction horizon
    n_features:        int   = 5      # input features per time step

    # ── Training ──────────────────────────────────────────────────────────────
    n_epochs:          int   = 12
    lr_readout:        float = 0.01
    lr_attn:           float = 0.01
    attn_shift:        float = np.pi / 2
    batch_attn:        int   = 4
    batch_size:        int   = 16     # mini-batch for readout training

    # ── Feature grouping ──────────────────────────────────────────────────────
    n_fock_features:   int   = 20     # LexGrouping output dimension

    @property
    def varqite_n_params(self) -> int:
        """Params in the VarQITE photonic ansatz: 2 PS per mode pair per layer."""
        # For a GenericInterferometer with n_modes modes and varqite_layers layers
        # Each MZI has 2 phase shifters; there are n_modes*(n_modes-1)/2 MZIs per layer
        # Simplified: we use n_modes phase shifters per layer
        return self.n_modes * self.varqite_layers

    @property
    def feature_dim(self) -> int:
        """Feature dimension after attention concatenation."""
        return self.n_fock_features * self.n_heads

    @property
    def attn_n_params(self) -> int:
        """Trainable params per Q/K/V interferometer in one head."""
        return self.n_modes * self.n_attn_layers


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PHOTONIC ENCODER
#    Maps real-valued features → phase shifts on optical modes.
#    Uses angle encoding: each feature modulates a phase shifter.
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicEncoder:
    """
    Maps a feature vector x ∈ ℝ^d → photonic circuit phase configuration.

    Architecture:
        Initial BS layer (50:50 mixing) → Phase encoding x_i·π on mode i
        → Entangling BS cascade

    The encoded features are then passed through the reservoir.
    In the photonic paradigm, this replaces qubit angle encoding with
    phase-shifter angle encoding on optical modes.
    """
    def __init__(self, n_modes: int, n_features: int):
        self.n_modes = n_modes
        self.n_features = min(n_features, n_modes)

    def build_encoding_circuit(self, x: np.ndarray) -> pcvl.Circuit:
        """Build a Perceval circuit that encodes features as phase shifts."""
        circ = pcvl.Circuit(self.n_modes)

        # Initial beam splitter layer (creates superposition)
        for i in range(0, self.n_modes - 1, 2):
            circ.add(i, pcvl.BS())

        # Phase encoding: each feature → phase shifter on a mode
        for i in range(self.n_features):
            phase = float(x[i]) * np.pi
            circ.add(i, pcvl.PS(phase))

        # Entangling BS layer (odd pairs)
        for i in range(1, self.n_modes - 1, 2):
            circ.add(i, pcvl.BS())

        return circ


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BOSON SAMPLING RESERVOIR
#    Fixed random interferometer — the photonic analog of the qubit reservoir.
#    Photons propagate through a random unitary (Haar-random interferometer).
#    Output: photon-count distribution / Fock-state probabilities.
#    NEVER TRAINED — core QRC philosophy.
# ═══════════════════════════════════════════════════════════════════════════════

class BosonSamplingReservoir:
    """
    Fixed random photonic reservoir using Perceval's GenericInterferometer.

    The reservoir maps encoded photonic states through a fixed random
    interferometer. The output distribution over Fock states provides
    an exponentially rich feature space (boson sampling hardness).

    This is the photonic equivalent of the random U3+CZ qubit reservoir.
    """
    def __init__(self, n_modes: int, n_layers: int, seed: int = 42):
        self.n_modes = n_modes
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        self._circuit = self._build(rng)

    def _build(self, rng: np.random.Generator) -> pcvl.Circuit:
        """Build a fixed random interferometer circuit."""
        circ = pcvl.Circuit(self.n_modes)

        for layer in range(self.n_layers):
            # Random phase shifters on each mode
            for i in range(self.n_modes):
                phase = float(rng.uniform(0, 2 * np.pi))
                circ.add(i, pcvl.PS(phase))

            # Beam splitters on even pairs
            start = layer % 2
            for i in range(start, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())

            # Additional random phases
            for i in range(self.n_modes):
                phase = float(rng.uniform(0, 2 * np.pi))
                circ.add(i, pcvl.PS(phase))

        return circ

    @property
    def circuit(self) -> pcvl.Circuit:
        return self._circuit


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PHOTONIC FEATURE EXTRACTOR
#    Computes Fock-state probability distribution from a photonic circuit.
#    This replaces ⟨Z_i⟩ expectation values with photon-count features.
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicFeatureExtractor:
    """
    Extracts feature vectors from photonic circuit outputs.

    Given a composed circuit (encoder + reservoir/ansatz), computes
    the probability distribution over output Fock states.

    Output features:
    - Mode occupation probabilities (analog of ⟨Z_i⟩)
    - Top-k Fock state probabilities (richer features)
    """
    def __init__(self, n_modes: int, n_photons: int, n_features: int = 20):
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_features = n_features

        # Build input state: photons in first n_photons modes
        state_list = [0] * n_modes
        for i in range(min(n_photons, n_modes)):
            state_list[i] = 1
        self.input_state = pcvl.BasicState(state_list)

    def extract(self, circuit: pcvl.Circuit) -> np.ndarray:
        """
        Run the circuit and extract feature vector.

        Returns:
            features: (n_features,) numpy array of probabilities
        """
        # Set up the Perceval backend
        backend = pcvl.BackendFactory.get_backend("SLOS")
        backend.set_circuit(circuit)
        backend.set_input_state(self.input_state)

        # Get output probability distribution
        probs = backend.prob_distribution()

        # Convert to feature vector
        features = np.zeros(self.n_features)

        # Mode occupation expectations (like ⟨n_i⟩ — analog of ⟨Z_i⟩)
        mode_occ = np.zeros(self.n_modes)
        for state, prob in probs.items():
            for mode in range(self.n_modes):
                mode_occ[mode] += float(prob) * state[mode]

        # Fill features: first n_modes slots = mode occupations
        n_occ = min(self.n_modes, self.n_features)
        features[:n_occ] = mode_occ[:n_occ]

        # Fill remaining slots with top Fock-state probabilities
        if self.n_features > n_occ:
            prob_list = sorted(
                [(float(p), s) for s, p in probs.items()],
                key=lambda x: x[0],
                reverse=True
            )
            for j, (p, _) in enumerate(prob_list):
                idx = n_occ + j
                if idx >= self.n_features:
                    break
                features[idx] = p

        return features


# ═══════════════════════════════════════════════════════════════════════════════
# 4. VarQITE-SVD MODULE (PHOTONIC VERSION)
#    Same mathematical principle as qubit VarQITE, but:
#    - Ansatz = trainable Perceval interferometer
#    - Features = photonic output probabilities instead of ⟨Z_i⟩
#    - QGT/C computed via finite-difference on photonic features
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicVarQITESVD:
    """
    VarQITE-based SVD / PCA on photonic feature trajectories.

    Instead of operating on statevectors directly (impossible for photonic
    circuits with sampling), we operate on the feature vectors f(θ)
    extracted from the photonic circuit outputs.

    H = -Σ_t |f_t⟩⟨f_t|  (defined on classical feature space)

    The ansatz is a trainable photonic interferometer that transforms
    the input state. Its parameters θ are evolved via McLachlan's principle
    on the extracted feature vectors.

    This is mathematically equivalent to finding the dominant principal
    component of the feature trajectory — a quantum-inspired PCA where
    the features themselves come from a boson sampling reservoir.
    """

    def __init__(self, cfg: PhotonicQLAConfig):
        self.cfg = cfg
        self.n_modes = cfg.n_modes
        self.n_photons = cfg.n_photons
        self.P = cfg.varqite_n_params
        self.s = cfg.varqite_shift
        self.extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features
        )

    def _build_ansatz(self, params: np.ndarray) -> pcvl.Circuit:
        """
        Build a trainable photonic ansatz circuit.

        Architecture: [PS(θ_i) on each mode + BS ladder] × n_layers
        This is the photonic analog of the [RY RZ CNOT] qubit ansatz.
        """
        circ = pcvl.Circuit(self.n_modes)

        # Initial beam splitter layer
        for i in range(0, self.n_modes - 1, 2):
            circ.add(i, pcvl.BS())

        for layer in range(self.cfg.varqite_layers):
            # Phase shifters (trainable)
            for i in range(self.n_modes):
                idx = layer * self.n_modes + i
                circ.add(i, pcvl.PS(float(params[idx])))

            # Beam splitter entangling layer
            start = layer % 2
            for i in range(start, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())

        return circ

    def _get_features(self, params: np.ndarray,
                      base_circuit: pcvl.Circuit) -> np.ndarray:
        """
        Compose base circuit (encoder+reservoir) with ansatz(θ),
        then extract feature vector.
        """
        ansatz = self._build_ansatz(params)
        full_circuit = pcvl.Circuit(self.n_modes)
        full_circuit.add(0, base_circuit)
        full_circuit.add(0, ansatz)
        return self.extractor.extract(full_circuit)

    def _compute_A_C(
        self,
        params: np.ndarray,
        reservoir_features: List[np.ndarray],
        base_circuit: pcvl.Circuit,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute QGT matrix A and energy gradient C via finite-difference
        on photonic feature vectors.

        A_ij ≈ (∂f/∂θ_i)ᵀ (∂f/∂θ_j) - (∂f/∂θ_i·f)(f·∂f/∂θ_j)
        C_i  = -Σ_t (∂f/∂θ_i)ᵀ f_t · (f_t·f)

        where f = f(θ) is the feature vector from the ansatz circuit.
        """
        f = self._get_features(params, base_circuit)

        # Compute all derivatives ∂f/∂θ_i via parameter-shift
        d_f = np.zeros((self.P, len(f)))
        for i in range(self.P):
            p_plus = params.copy(); p_plus[i] += self.s
            p_minus = params.copy(); p_minus[i] -= self.s
            f_plus = self._get_features(p_plus, base_circuit)
            f_minus = self._get_features(p_minus, base_circuit)
            d_f[i] = (f_plus - f_minus) / (2.0 * np.sin(self.s))

        # QGT: A_ij = (∂_if)·(∂_jf) - (∂_if·f)(f·∂_jf)
        inner_dd = d_f @ d_f.T                          # (P, P)
        inner_df = d_f @ f                               # (P,)
        outer_term = np.outer(inner_df, inner_df)        # (P, P)
        norm_sq = np.dot(f, f)
        if norm_sq > 1e-12:
            outer_term /= norm_sq
        A = inner_dd - outer_term

        # Energy gradient: C_i = -Σ_t (∂_if · f_t)(f_t · f)
        C = np.zeros(self.P)
        for f_t in reservoir_features:
            overlap = np.dot(f_t, f)
            d_overlap = d_f @ f_t            # (P,) — ∂_i(f_t · f)
            C -= d_overlap * overlap

        return A, C

    def run(
        self,
        reservoir_features: List[np.ndarray],
        base_circuit: pcvl.Circuit,
        init_params: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run VarQITE imaginary time evolution on photonic features.

        Args:
            reservoir_features: T feature vectors from the reservoir
            base_circuit: composed encoder+reservoir circuit
            init_params: initial θ; if None → random

        Returns:
            params_final: converged ansatz parameters θ*
            features: extracted feature vector at θ* (shape n_fock_features,)
        """
        cfg = self.cfg
        rng = np.random.default_rng(
            seed=int(abs(reservoir_features[0][0]) * 1e6) % (2**31)
        )
        params = (init_params.copy() if init_params is not None
                  else rng.uniform(0, 2 * np.pi, self.P))

        for step in range(cfg.varqite_steps):
            A, C = self._compute_A_C(params, reservoir_features, base_circuit)

            # Tikhonov: solve (A + λI)δθ = C
            A_reg = A + cfg.varqite_reg * np.eye(self.P)
            try:
                delta_theta = scipy_solve(A_reg, C, assume_a="pos")
            except np.linalg.LinAlgError:
                delta_theta = np.linalg.lstsq(A_reg, C, rcond=None)[0]

            params = params + cfg.varqite_dtau * delta_theta

        # Extract final features
        features = self._get_features(params, base_circuit)
        return params, features


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PHOTONIC ATTENTION (Q, K, V INTERFEROMETERS)
#    Each head applies trainable interferometers to the VarQITE-compressed
#    features. Output: mode-occupation features used for attention scores.
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicAttentionHead:
    """
    One photonic attention head.

    Q, K, V each have a separate trainable interferometer.
    Applied to the VarQITE-compressed features, producing
    new feature vectors for attention computation.

    In photonic terms:
    - Each projection is a parametrised interferometer
    - Features are photonic output probabilities
    - Attention scores = dot products of photonic feature vectors
    """

    def __init__(self, n_modes: int, n_layers: int,
                 n_photons: int, n_features: int, head_id: int):
        self.n_modes = n_modes
        self.n_layers = n_layers
        self.n_photons = n_photons
        self.n_features = n_features
        self.P = n_modes * n_layers  # params per Q/K/V

        rng = np.random.default_rng(seed=head_id * 31337 + 17)
        self.params_q = rng.uniform(0, 2 * np.pi, self.P)
        self.params_k = rng.uniform(0, 2 * np.pi, self.P)
        self.params_v = rng.uniform(0, 2 * np.pi, self.P)

        self.extractor = PhotonicFeatureExtractor(
            n_modes, n_photons, n_features
        )

    def _build_proj_circuit(self, params: np.ndarray) -> pcvl.Circuit:
        """Trainable interferometer for Q/K/V projection."""
        circ = pcvl.Circuit(self.n_modes)
        for layer in range(self.n_layers):
            for i in range(self.n_modes):
                idx = layer * self.n_modes + i
                circ.add(i, pcvl.PS(float(params[idx])))
            start = layer % 2
            for i in range(start, self.n_modes - 1, 2):
                circ.add(i, pcvl.BS())
        return circ

    def _project(self, base_circuit: pcvl.Circuit,
                 params: np.ndarray) -> np.ndarray:
        """Apply projection interferometer and extract features."""
        proj = self._build_proj_circuit(params)
        full = pcvl.Circuit(self.n_modes)
        full.add(0, base_circuit)
        full.add(0, proj)
        return self.extractor.extract(full)

    def project(self, base_circuit: pcvl.Circuit,
                which: str) -> np.ndarray:
        p = {"q": self.params_q, "k": self.params_k,
             "v": self.params_v}[which]
        return self._project(base_circuit, p)

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.params_q, self.params_k, self.params_v])

    def set_params(self, flat: np.ndarray):
        n = self.P
        self.params_q = flat[0:n].copy()
        self.params_k = flat[n:2*n].copy()
        self.params_v = flat[2*n:3*n].copy()


class PhotonicAttentionLayer:
    """
    Multi-head photonic attention over VarQITE-compressed circuits.

    For each head h and each token t with compressed circuit c_t:
        Q_t^h = U_Q^h(c_t) → features
        K_t^h = U_K^h(c_t) → features
        V_t^h = U_V^h(c_t) → features

    Causal-masked scaled dot-product attention:
        scores = Q·Kᵀ / √d,  mask future positions
        output = softmax(scores) · V

    Heads concatenated → (T, n_heads × n_features)
    """

    def __init__(self, cfg: PhotonicQLAConfig):
        self.cfg = cfg
        self.heads = [
            PhotonicAttentionHead(
                cfg.n_modes, cfg.n_attn_layers,
                cfg.n_photons, cfg.n_fock_features, h
            )
            for h in range(cfg.n_heads)
        ]

    def forward(self, compressed_circuits: List[pcvl.Circuit]) -> np.ndarray:
        """
        Args:
            compressed_circuits: T circuits (encoder+reservoir+VarQITE ansatz)
        Returns:
            (T, feature_dim) attention output
        """
        T = len(compressed_circuits)
        d = self.cfg.n_fock_features
        head_outs = []

        for head in self.heads:
            Qs = np.array([head.project(c, "q") for c in compressed_circuits])
            Ks = np.array([head.project(c, "k") for c in compressed_circuits])
            Vs = np.array([head.project(c, "v") for c in compressed_circuits])

            scores = (Qs @ Ks.T) / np.sqrt(d)
            mask = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + mask
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores)
            alpha = exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-12)
            out = alpha @ Vs
            head_outs.append(out)

        return np.concatenate(head_outs, axis=-1)

    def get_all_params(self) -> np.ndarray:
        return np.concatenate([h.get_params() for h in self.heads])

    def set_all_params(self, flat: np.ndarray):
        n = len(self.heads[0].get_params())
        for i, head in enumerate(self.heads):
            head.set_params(flat[i * n: (i + 1) * n])


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. MerLin-BASED ATTENTION (OPTIONAL — uses QuantumLayer for gradient)
#     If MerLin is available, we can use its differentiable QuantumLayer
#     for the attention projections, getting PyTorch autograd for free.
# ═══════════════════════════════════════════════════════════════════════════════

class MerLinAttentionHead(nn.Module):
    """
    Attention head using MerLin's QuantumLayer for differentiable photonic Q/K/V.

    Uses CircuitBuilder to create trainable interferometers that project
    VarQITE-compressed features into Q, K, V subspaces.
    """

    def __init__(self, n_modes: int, n_photons: int,
                 input_dim: int, output_dim: int):
        super().__init__()
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.output_dim = output_dim

        # Build Q, K, V as separate QuantumLayers using CircuitBuilder
        self.q_layer = self._build_layer(input_dim, "Q")
        self.k_layer = self._build_layer(input_dim, "K")
        self.v_layer = self._build_layer(input_dim, "V")

    def _build_layer(self, input_dim: int, name: str) -> nn.Module:
        """Build a trainable QuantumLayer with LexGrouping readout."""
        builder = CircuitBuilder(n_modes=self.n_modes)
        builder.add_entangling_layer(trainable=True, name=f"{name}_U1")
        builder.add_angle_encoding(
            modes=list(range(min(input_dim, self.n_modes))),
            name=f"{name}_input"
        )
        builder.add_rotations(trainable=True, name=f"{name}_theta")
        builder.add_superpositions(depth=1)

        core = QuantumLayer(
            input_size=input_dim,
            builder=builder,
            n_photons=self.n_photons,
            dtype=torch.float32,
        )
        return nn.Sequential(
            core,
            LexGrouping(core.output_size, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (T, input_dim) features from VarQITE
        Returns:
            Q, K, V: each (T, output_dim)
        """
        return self.q_layer(x), self.k_layer(x), self.v_layer(x)


class MerLinAttentionLayer(nn.Module):
    """
    Multi-head photonic attention using MerLin QuantumLayers.
    Fully differentiable through PyTorch autograd.
    """

    def __init__(self, cfg: PhotonicQLAConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        self.heads = nn.ModuleList([
            MerLinAttentionHead(
                cfg.n_modes, cfg.n_photons,
                input_dim, cfg.n_fock_features
            )
            for _ in range(cfg.n_heads)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, input_dim) features
        Returns:
            (T, feature_dim) attention output
        """
        T = x.shape[0]
        d = self.cfg.n_fock_features
        head_outs = []

        for head in self.heads:
            Q, K, V = head(x)

            scores = (Q @ K.T) / np.sqrt(d)
            mask = torch.triu(torch.full((T, T), -1e9), diagonal=1)
            scores = scores + mask
            alpha = torch.softmax(scores, dim=-1)
            out = alpha @ V
            head_outs.append(out)

        return torch.cat(head_outs, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FULL PHOTONIC QLA-TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PhotonicQLATransformer(nn.Module):
    """
    Full Photonic Quantum Linear Algebra Transformer.

    Forward pass:
        1. Encode each time step's features via PhotonicEncoder
        2. Pass through BosonSamplingReservoir (fixed)
           → per-timestep photonic circuits + Fock probability features
        3. Run VarQITE-SVD on feature trajectory
           → compressed feature vectors (dominant subspace)
        4. Multi-head photonic attention on compressed features
        5. Classical readout → prediction ŷ_{t+h}
    """

    def __init__(self, cfg: PhotonicQLAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PhotonicEncoder(cfg.n_modes, cfg.n_features)
        self.reservoir = BosonSamplingReservoir(
            cfg.n_modes, cfg.n_res_layers, cfg.reservoir_seed
        )
        self.varqite = PhotonicVarQITESVD(cfg)
        self.feature_extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features
        )

        # Attention (pure Perceval path)
        self.attention = PhotonicAttentionLayer(cfg)

        # Classical readout head
        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

        # Cache for VarQITE warm-start
        self._varqite_cache: Optional[np.ndarray] = None

    def _build_reservoir_circuit(self, x: np.ndarray) -> pcvl.Circuit:
        """Encode features + pass through reservoir → composed circuit."""
        enc = self.encoder.build_encoding_circuit(x)
        res = self.reservoir.circuit
        full = pcvl.Circuit(self.cfg.n_modes)
        full.add(0, enc)
        full.add(0, res)
        return full

    def _reservoir_features(self, sequence: np.ndarray) -> Tuple[
        List[pcvl.Circuit], List[np.ndarray]
    ]:
        """
        Encode + reservoir → list of (circuit, feature_vector) per time step.
        """
        circuits = []
        features = []
        for t in range(len(sequence)):
            circ = self._build_reservoir_circuit(sequence[t])
            feat = self.feature_extractor.extract(circ)
            circuits.append(circ)
            features.append(feat)
        return circuits, features

    def _varqite_compress(
        self,
        circuits: List[pcvl.Circuit],
        features: List[np.ndarray],
    ) -> Tuple[List[pcvl.Circuit], np.ndarray]:
        """
        Run VarQITE on causal windows of reservoir features.

        For token t: context = features[0:t+1] (causal).
        Uses the last circuit as base for ansatz composition.

        Returns:
            compressed_circuits: T circuits with VarQITE ansatz appended
            final_params: last token's converged params (warm start)
        """
        compressed = []
        init_p = self._varqite_cache

        for t in range(len(circuits)):
            context_features = features[:t + 1]
            base_circuit = circuits[t]

            params, _ = self.varqite.run(
                context_features, base_circuit, init_params=init_p
            )

            # Build compressed circuit = base + VarQITE ansatz
            ansatz = self.varqite._build_ansatz(params)
            comp_circ = pcvl.Circuit(self.cfg.n_modes)
            comp_circ.add(0, base_circuit)
            comp_circ.add(0, ansatz)
            compressed.append(comp_circ)

            init_p = params

        self._varqite_cache = init_p
        return compressed, init_p

    def forward_np(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            sequence: (T, n_features) array of input features

        Returns:
            prediction: (horizon,) tensor
        """
        # 1. Reservoir
        circuits, features = self._reservoir_features(sequence)

        # 2. VarQITE-SVD compression
        comp_circuits, _ = self._varqite_compress(circuits, features)

        # 3. Photonic attention
        attn_out = self.attention.forward(comp_circuits)
        last = torch.tensor(attn_out[-1], dtype=torch.float32)

        # 4. Classical readout
        pred = self.readout(last)
        return pred

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        pred = self.forward_np(sequence)
        return pred.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. MerLin HYBRID MODEL (alternative — uses QuantumLayer for attention)
# ═══════════════════════════════════════════════════════════════════════════════

class MerLinHybridModel(nn.Module):
    """
    Hybrid model using:
    - Perceval BosonSamplingReservoir for feature extraction
    - Classical VarQITE-SVD for compression
    - MerLin QuantumLayer for differentiable attention
    - PyTorch readout

    This combines the best of both: photonic reservoir richness
    with MerLin's autograd-compatible quantum layers for training.
    """

    def __init__(self, cfg: PhotonicQLAConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PhotonicEncoder(cfg.n_modes, cfg.n_features)
        self.reservoir = BosonSamplingReservoir(
            cfg.n_modes, cfg.n_res_layers, cfg.reservoir_seed
        )
        self.feature_extractor = PhotonicFeatureExtractor(
            cfg.n_modes, cfg.n_photons, cfg.n_fock_features
        )
        self.varqite = PhotonicVarQITESVD(cfg)

        # MerLin attention (differentiable)
        if HAS_MERLIN:
            self.attention = MerLinAttentionLayer(cfg, cfg.n_fock_features)
        else:
            # Fallback: classical attention
            self.attention = None

        # Classical readout
        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

        self._varqite_cache = None

    def _extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Full quantum feature extraction pipeline:
        encode → reservoir → VarQITE compression → feature vectors.

        Returns:
            (T, n_fock_features) numpy array
        """
        # Reservoir features
        all_features = []
        all_circuits = []
        for t in range(len(sequence)):
            enc = self.encoder.build_encoding_circuit(sequence[t])
            res = self.reservoir.circuit
            circ = pcvl.Circuit(self.cfg.n_modes)
            circ.add(0, enc)
            circ.add(0, res)
            feat = self.feature_extractor.extract(circ)
            all_features.append(feat)
            all_circuits.append(circ)

        # VarQITE compression (causal)
        compressed_features = []
        init_p = self._varqite_cache

        for t in range(len(sequence)):
            context = all_features[:t + 1]
            params, comp_feat = self.varqite.run(
                context, all_circuits[t], init_params=init_p
            )
            compressed_features.append(comp_feat)
            init_p = params

        self._varqite_cache = init_p
        return np.array(compressed_features)

    def forward(self, sequence: np.ndarray) -> torch.Tensor:
        """
        Args:
            sequence: (T, n_features)
        Returns:
            prediction: (horizon,) tensor
        """
        # Quantum feature extraction (numpy/Perceval)
        features = self._extract_features(sequence)
        features_t = torch.tensor(features, dtype=torch.float32)

        # MerLin attention (PyTorch autograd)
        if self.attention is not None:
            attn_out = self.attention(features_t)
        else:
            # Fallback: use features directly, concatenate for multi-head
            attn_out = features_t.repeat(1, self.cfg.n_heads)

        last = attn_out[-1]
        return self.readout(last)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DATA LOADING (Swaption Dataset)
# ═══════════════════════════════════════════════════════════════════════════════

def load_swaption_data(
    train_path: str,
    cfg: PhotonicQLAConfig,
    target_columns: Optional[List[str]] = None,
    n_pca_features: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Load and prepare swaption data for the QLA-Transformer.

    The training data has ~225 columns (tenor/maturity combos) × ~494 dates.
    We reduce dimensionality via PCA to n_pca_features for the quantum circuit.

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    df = pd.read_excel(train_path)

    # Drop date column, handle missing values
    date_col = df.columns[0]
    dates = df[date_col]
    data = df.drop(columns=[date_col])

    # Forward-fill missing values
    data = data.ffill().bfill()
    values = data.values.astype(np.float32)

    # Standardize
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    # PCA to reduce to n_pca_features
    pca = PCA(n_components=n_pca_features)
    values_pca = pca.fit_transform(values_scaled)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Normalize PCA features to [0, 1] for phase encoding
    v_min = values_pca.min(axis=0, keepdims=True)
    v_max = values_pca.max(axis=0, keepdims=True)
    values_norm = (values_pca - v_min) / (v_max - v_min + 1e-9)

    # Create sequences
    X, y = [], []
    seq_len = cfg.seq_len
    horizon = cfg.horizon

    for i in range(len(values_norm) - seq_len - horizon + 1):
        X.append(values_norm[i: i + seq_len])
        y.append(values_norm[i + seq_len + horizon - 1])

    X = np.array(X)
    y = np.array(y)

    # Train/test split (80/20, no shuffle for time series)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target dim: {y_train.shape[1]}")

    return X_train, y_train, X_test, y_test, (scaler, pca, v_min, v_max)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(
    model: PhotonicQLATransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: PhotonicQLAConfig,
    verbose: bool = True,
) -> List[float]:
    """
    Hybrid training loop.

    Step 1: Readout trained with Adam (PyTorch autograd).
    Step 2: Attention Q/K/V trained with parameter-shift (photonic gradient).
    """
    optimizer = optim.Adam(model.readout.parameters(), lr=cfg.lr_readout)
    loss_fn = nn.MSELoss()
    losses = []
    N = len(X_train)
    s = cfg.attn_shift
    rng = np.random.default_rng(0)

    for epoch in range(cfg.n_epochs):

        # Step 1: Readout with Adam
        epoch_loss = 0.0
        perm = rng.permutation(N)

        for i in perm[:cfg.batch_size]:  # Mini-batch for speed
            seq = X_train[i]
            tgt = torch.tensor(y_train[i][:cfg.horizon], dtype=torch.float32)
            optimizer.zero_grad()
            pred = model.forward_np(seq)
            loss = loss_fn(pred, tgt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Step 2: Parameter-shift on attention photonic params
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
                pr_plus = model.forward_np(seq)
                loss_plus = (pr_plus.item() - tgt) ** 2

                p_minus = q_params.copy(); p_minus[j] -= s
                model.attention.set_all_params(p_minus)
                pr_minus = model.forward_np(seq)
                loss_minus = (pr_minus.item() - tgt) ** 2

                q_grad[j] += (loss_plus - loss_minus) / (2.0 * np.sin(s))

        model.attention.set_all_params(q_params)
        q_grad /= max(len(sample_idx), 1)
        q_params -= cfg.lr_attn * q_grad
        model.attention.set_all_params(q_params)

        avg_loss = epoch_loss / min(cfg.batch_size, N)
        losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs}"
                  f"  |  MSE: {avg_loss:.6f}"
                  f"  |  ‖∇attn‖: {np.linalg.norm(q_grad):.5f}")

    return losses


# ═══════════════════════════════════════════════════════════════════════════════
# 9. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    losses: List[float],
    cfg: PhotonicQLAConfig,
    save_path: str = "photonic_qla_results.png",
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
                 label="Photonic QLAT forecast")
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
    axes[0].set_title("Forecast vs Ground Truth", color="white", fontsize=12)
    axes[0].set_xlabel("Time step", color="white")
    axes[0].set_ylabel("Normalised value", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white", fontsize=9)

    # Panel 2: Training loss
    axes[1].plot(losses, color=colors["loss"], lw=2)
    axes[1].set_title("Training Loss (MSE)", color="white", fontsize=12)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss", color="white")
    if min(losses) > 0:
        axes[1].set_yscale("log")

    # Panel 3: Architecture
    ax3 = axes[2]
    ax3.axis("off")
    diagram = [
        ("Input x_t (swaption features)",       0.90),
        ("↓",                                    0.83),
        ("Photonic Encoder",                     0.76),
        ("angle → PS on modes + BS ring",        0.70),
        ("↓",                                    0.63),
        ("Boson Sampling Reservoir [FIXED]",     0.56),
        ("random PS + BS interferometer",        0.50),
        ("↓",                                    0.43),
        ("VarQITE-SVD (on Fock features)",       0.36),
        ("H=-Σ|f_t⟩⟨f_t| · QGT · McLachlan",   0.30),
        ("↓",                                    0.23),
        ("Photonic Attention (Q,K,V heads)",     0.16),
        ("↓",                                    0.10),
        ("Classical Readout → ŷ_{t+h}",          0.04),
    ]
    for i, (label, y_pos) in enumerate(diagram):
        weight = "bold" if i in {0, 2, 5, 8, 11, 13} else "normal"
        color_ = ("#4dd0e1" if i == 0 else
                  "#a5d6a7" if i in {5, 6} else
                  "#ce93d8" if i in {8, 9} else
                  "#ff7043" if i in {11, 12} else
                  "#fff59d" if i == 13 else "white")
        ax3.text(0.5, y_pos, label, ha="center", va="center",
                 color=color_, fontsize=8.5, fontweight=weight,
                 fontfamily="monospace", transform=ax3.transAxes)
    ax3.set_title("Model Architecture", color="white", fontsize=12)

    arch = (f"modes={cfg.n_modes}  |  photons={cfg.n_photons}  |  "
            f"res_layers={cfg.n_res_layers}  |  heads={cfg.n_heads}")
    fig.suptitle(f"Photonic QLA-Transformer + Boson Sampling\n{arch}",
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
    print("║  VarQITE-SVD · QGT · McLachlan · Perceval · MerLin             ║")
    print("║  Swaption Pricing — Quandela EPFL Hackathon 2026               ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    cfg = PhotonicQLAConfig(
        n_modes         = 8,
        n_photons       = 4,
        n_res_layers    = 3,
        varqite_layers  = 2,
        varqite_steps   = 8,
        varqite_dtau    = 0.08,
        varqite_reg     = 1e-3,
        n_heads         = 2,
        n_attn_layers   = 2,
        seq_len         = 10,
        horizon         = 1,
        n_epochs        = 20,
        lr_readout      = 5e-3,
        lr_attn         = 0.03,
        batch_attn      = 4,
        batch_size      = 16,
        n_features      = 5,
        n_fock_features = 20,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    print("[1/4] Loading swaption data…")
    X_train, y_train, X_test, y_test, transforms = load_swaption_data(
        train_path, cfg, n_pca_features=cfg.n_features
    )
    print()

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[2/4] Building photonic model…")
    model = PhotonicQLATransformer(cfg)

    n_varqite = cfg.varqite_n_params
    n_attn = len(model.attention.get_all_params())
    n_readout = sum(p.numel() for p in model.readout.parameters())
    print(f"      Photonic modes             : {cfg.n_modes}")
    print(f"      Photons                    : {cfg.n_photons}")
    print(f"      Reservoir layers           : {cfg.n_res_layers} (fixed, seed={cfg.reservoir_seed})")
    print(f"      VarQITE ansatz params      : {n_varqite}  (re-optimised each fwd)")
    print(f"      Attention Q,K,V params     : {n_attn}  (param-shift trained)")
    print(f"      Readout params             : {n_readout}  (Adam trained)")
    print(f"      VarQITE steps per token    : {cfg.varqite_steps}")
    print(f"      Fock feature dim           : {cfg.n_fock_features}")
    print(f"      Feature dim after attention: {cfg.feature_dim}")
    print(f"      MerLin available           : {HAS_MERLIN}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[3/4] Training…")
    losses = train_model(model, X_train, y_train, cfg, verbose=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set…")
    preds = []
    for i in range(len(X_test)):
        p = model.predict(X_test[i])
        preds.append(p[0])
    preds = np.array(preds)

    # Compare first PCA component for metrics
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

    plot_results(y_true, y_hat, losses, cfg,
                 save_path="photonic_qla_results.png")
    print("\n✓ Done.")
    return model, losses, preds, y_test


if __name__ == "__main__":
    model, losses, preds, y_test = main("train.xlsx")
