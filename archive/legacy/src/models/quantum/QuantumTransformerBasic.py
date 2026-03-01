import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QTConfig:
    """Central configuration for the Quantum Transformer."""
    # Quantum circuit dimensions
    n_qubits:    int = 4    # qubits per circuit (more → richer but slower)
    n_res_layers: int = 3   # reservoir depth (more → more chaotic dynamics)
    n_heads:     int = 2    # attention heads  (each has its own Q,K,V circuit)
    n_attn_layers: int = 2  # depth of Q/K/V projection circuits

    # Sequence / forecasting
    seq_len:   int = 6      # input window length
    horizon:   int = 1      # forecast steps ahead

    # Reservoir randomness seed (fixed → deterministic reservoir)
    reservoir_seed: int = 42

    # Training hyperparameters
    n_epochs:    int = 30
    lr_readout:  float = 5e-3
    lr_quantum:  float = 0.05
    batch_quantum: int = 4   # samples used for quantum gradient per epoch
    param_shift:   float = np.pi / 2  # standard shift value

    @property
    def feature_dim(self) -> int:
        """Output feature dimension per token after attention concatenation."""
        return self.n_qubits * self.n_heads

    @property
    def n_quantum_params_per_head(self) -> int:
        """Trainable params per projection (Q or K or V) in one head."""
        return self.n_qubits * 2 * self.n_attn_layers   # RY + RZ per qubit per layer


# ══════════════════════════════════════════════════════════════════════════════
# 1. QUANTUM ENCODER
#    Encodes a scalar x ∈ [0,1] into an n-qubit quantum state.
#    Strategy: Hadamard → data re-uploading with phase diversity → CNOT ring
# ══════════════════════════════════════════════════════════════════════════════

class QuantumEncoder:
    """
    Maps a scalar time-series value x ∈ [0,1] to an n-qubit quantum state.

    Circuit structure (depth ~ 2·n_qubits):
        H^⊗n  →  ∏_i [RY(x·π·(i+1)/n)  RZ(x·π/(i+1))]  →  CNOT ring
    
    The increasing coefficients give each qubit a different Fourier component
    of the input (data re-uploading principle).
    """
    def __init__(self, n_qubits: int):
        self.n = n_qubits

    def encode(self, x: float) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Enc")
        # Superposition
        qc.h(range(self.n))
        # Data re-uploading with harmonic diversity
        for i in range(self.n):
            qc.ry(float(x) * np.pi * (i + 1) / self.n, i)
            qc.rz(float(x) * np.pi / (i + 1),          i)
        # Entanglement ring
        for i in range(self.n - 1):
            qc.cx(i, i + 1)
        qc.cx(self.n - 1, 0)
        return qc


# ══════════════════════════════════════════════════════════════════════════════
# 2. QUANTUM RESERVOIR
#    Fixed (non-trainable) random unitary circuit.
#    Creates an exponentially large feature space via chaotic quantum dynamics.
#    Brick-wall CZ layout maximises entanglement spread per layer.
# ══════════════════════════════════════════════════════════════════════════════

class QuantumReservoir:
    """
    Fixed random quantum circuit acting as the reservoir.

    Circuit structure per layer:
        [U3(θ,φ,λ) on each qubit]  →  [brick-wall CZ]  →  [long-range CX]

    The reservoir is built once and never updated — classical QRC philosophy.
    Increasing n_layers makes the dynamics more chaotic (better separation).
    """
    def __init__(self, n_qubits: int, n_layers: int, seed: int = 42):
        self.n = n_qubits
        rng = np.random.default_rng(seed)
        self._circuit = self._build(rng, n_layers)

    def _build(self, rng: np.random.Generator, n_layers: int) -> QuantumCircuit:
        qc = QuantumCircuit(self.n, name="Reservoir")
        for layer in range(n_layers):
            # Random single-qubit unitaries
            for i in range(self.n):
                θ = rng.uniform(0, 2 * np.pi)
                φ = rng.uniform(0, 2 * np.pi)
                λ = rng.uniform(0, 2 * np.pi)
                qc.u(θ, φ, λ, i)
            # Brick-wall CZ (alternate even/odd to avoid repetition)
            start = layer % 2
            for i in range(start, self.n - 1, 2):
                qc.cz(i, i + 1)
            # Long-range entanglement
            if self.n > 2:
                qc.cx(0, self.n - 1)
        return qc

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circuit.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 3. QUANTUM ATTENTION HEAD
#    Three parametrized circuits (Q, K, V), each projecting reservoir states
#    onto a different subspace of the Hilbert space.
#    Features = expectation values ⟨Z_i⟩ ∈ [-1, 1] for each qubit i.
# ══════════════════════════════════════════════════════════════════════════════

class QuantumAttentionHead:
    """
    One multi-head quantum attention head.

    Each projection circuit (Q / K / V) has the structure:
        for each layer:
            RY(θ_i)  RZ(φ_i)  on qubit i
            CNOT ladder

    After applying the projection U_proj to the reservoir state |ψ⟩,
    we read out the feature vector:
        f_i = ⟨ψ| U†_proj  Z_i  U_proj |ψ⟩
    """
    def __init__(self, n_qubits: int, n_layers: int, head_id: int):
        self.n = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * 2 * n_layers   # RY + RZ per qubit per layer

        rng = np.random.default_rng(seed=head_id * 137 + 7)
        self.params_q = rng.uniform(0, 2 * np.pi, self.n_params)
        self.params_k = rng.uniform(0, 2 * np.pi, self.n_params)
        self.params_v = rng.uniform(0, 2 * np.pi, self.n_params)

        # Pre-build Pauli-Z observables for each qubit
        self._obs = [
            SparsePauliOp.from_list(
                [("I" * i + "Z" + "I" * (n_qubits - i - 1), 1.0)]
            )
            for i in range(n_qubits)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_proj_circuit(self, params: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n)
        for layer in range(self.n_layers):
            for i in range(self.n):
                idx = (layer * self.n + i) * 2
                qc.ry(float(params[idx]),     i)
                qc.rz(float(params[idx + 1]), i)
            for i in range(self.n - 1):
                qc.cx(i, i + 1)
        return qc

    def _measure_expectations(self, state: Statevector,
                               params: np.ndarray) -> np.ndarray:
        """Apply projection circuit to state → return ⟨Z_i⟩ vector."""
        proj = self._build_proj_circuit(params)
        projected = state.evolve(proj)
        return np.array([
            float(projected.expectation_value(obs).real)
            for obs in self._obs
        ])

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def project(self, state: Statevector, which: str) -> np.ndarray:
        """Project state with Q / K / V circuit. Returns feature vector."""
        params = {"q": self.params_q, "k": self.params_k, "v": self.params_v}[which]
        return self._measure_expectations(state, params)

    def project_with_params(self, state: Statevector,
                             params: np.ndarray, which: str) -> np.ndarray:
        """Project with explicit param vector (used in parameter-shift)."""
        return self._measure_expectations(state, params)

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter management (flat array for optimizer)
    # ──────────────────────────────────────────────────────────────────────────

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.params_q, self.params_k, self.params_v])

    def set_params(self, flat: np.ndarray):
        n = self.n_params
        self.params_q = flat[0:n].copy()
        self.params_k = flat[n:2*n].copy()
        self.params_v = flat[2*n:3*n].copy()


# ══════════════════════════════════════════════════════════════════════════════
# 4. QUANTUM TRANSFORMER BLOCK
#    Chains: Encoder → Reservoir → Multi-head Quantum Attention
# ══════════════════════════════════════════════════════════════════════════════

class QuantumTransformerBlock:
    """
    Full quantum transformer block.

    For a sequence x = [x_0, ..., x_{T-1}]:
      1. Each x_t → Encoder + Reservoir → quantum state |ψ_t⟩
      2. Multi-head attention over {|ψ_t⟩}:
           Q_t, K_t, V_t  = projection circuits applied to |ψ_t⟩
           a_{ij} = softmax( Q_i·K_j / √d )    (causal masked)
           out_t  = Σ_j a_{tj} · V_j            (weighted sum)
      3. Heads concatenated → feature matrix (T, n_heads·n_qubits)
    """
    def __init__(self, cfg: QTConfig):
        self.cfg = cfg
        self.encoder   = QuantumEncoder(cfg.n_qubits)
        self.reservoir = QuantumReservoir(cfg.n_qubits, cfg.n_res_layers,
                                          cfg.reservoir_seed)
        self.heads = [
            QuantumAttentionHead(cfg.n_qubits, cfg.n_attn_layers, h)
            for h in range(cfg.n_heads)
        ]

    # ──────────────────────────────────────────────────────────────────────────
    def _get_state(self, x: float) -> Statevector:
        """Encode scalar x → reservoir quantum state."""
        enc = self.encoder.encode(x)
        res = self.reservoir.circuit
        full = enc.compose(res)
        return Statevector.from_instruction(full)

    def _attention(self, states: List[Statevector]) -> np.ndarray:
        """Multi-head quantum attention over list of reservoir states."""
        T = len(states)
        head_outs = []

        for head in self.heads:
            Qs = np.array([head.project(s, "q") for s in states])   # (T, d)
            Ks = np.array([head.project(s, "k") for s in states])   # (T, d)
            Vs = np.array([head.project(s, "v") for s in states])   # (T, d)

            d = Qs.shape[-1]
            scores = (Qs @ Ks.T) / np.sqrt(d)                       # (T, T)

            # Causal mask: future positions → -∞
            causal_mask = np.triu(np.full((T, T), -1e9), k=1)
            scores = scores + causal_mask

            # Stable softmax
            scores -= scores.max(axis=-1, keepdims=True)
            exp_s = np.exp(scores)
            attn  = exp_s / exp_s.sum(axis=-1, keepdims=True)       # (T, T)

            out = attn @ Vs                                          # (T, d)
            head_outs.append(out)

        return np.concatenate(head_outs, axis=-1)                    # (T, n_heads·d)

    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """
        Args:
            sequence : (T,) normalized time series window
        Returns:
            features : (T, feature_dim) attention output
        """
        states   = [self._get_state(float(x)) for x in sequence]
        features = self._attention(states)
        return features

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter management (flat vector over all heads)
    # ──────────────────────────────────────────────────────────────────────────

    def get_all_params(self) -> np.ndarray:
        return np.concatenate([h.get_params() for h in self.heads])

    def set_all_params(self, flat: np.ndarray):
        n = len(self.heads[0].get_params())
        for i, head in enumerate(self.heads):
            head.set_params(flat[i * n: (i + 1) * n])


# ══════════════════════════════════════════════════════════════════════════════
# 5. FULL MODEL: QuantumTransformerForecaster
# ══════════════════════════════════════════════════════════════════════════════

class QuantumTransformerForecaster(nn.Module):
    """
    End-to-end quantum transformer for time series forecasting.

    Components:
      · QuantumTransformerBlock  — quantum encoder + reservoir + attention
      · Classical Readout        — Linear → GELU → Linear

    Training:
      · Quantum params → parameter-shift rule (finite diff on quantum loss)
      · Readout params → Adam (standard backprop)
    """
    def __init__(self, cfg: QTConfig):
        super().__init__()
        self.cfg = cfg
        self.qtx = QuantumTransformerBlock(cfg)

        # Classical readout on the LAST token's features
        feat = cfg.feature_dim
        self.readout = nn.Sequential(
            nn.Linear(feat, feat * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat * 2, cfg.horizon),
        )

    def forward_np(self, sequence: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        """Returns (quantum_features, prediction_tensor)."""
        feats      = self.qtx.forward(sequence)               # (T, feat_dim)
        last_feats = torch.tensor(feats[-1], dtype=torch.float32)
        pred       = self.readout(last_feats)                 # (horizon,)
        return feats, pred

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        _, pred = self.forward_np(sequence)
        return pred.numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 6. HYBRID TRAINER
#    · Readout  → Adam
#    · Quantum  → Parameter-shift rule  ∂L/∂θ_j = (L(θ+s) - L(θ-s)) / (2 sin s)
# ══════════════════════════════════════════════════════════════════════════════

def train(
    model:       QuantumTransformerForecaster,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    cfg:         QTConfig,
    verbose:     bool = True,
) -> List[float]:
    """
    Hybrid training loop.

    Parameter-shift rule for quantum params is O(P × batch_quantum × T) per
    epoch, where P = number of quantum params. For large P, a random subset of
    params is shifted each epoch (stochastic parameter-shift).
    """
    optimizer = optim.Adam(model.readout.parameters(), lr=cfg.lr_readout)
    loss_fn   = nn.MSELoss()
    losses    = []
    N         = len(X_train)
    s         = cfg.param_shift
    rng       = np.random.default_rng(0)

    for epoch in range(cfg.n_epochs):
        # ── Step 1: Train readout with Adam ──────────────────────────────────
        epoch_loss = 0.0
        perm = rng.permutation(N)
        for i in perm:
            seq    = X_train[i]
            target = torch.tensor([y_train[i]], dtype=torch.float32)
            optimizer.zero_grad()
            _, pred = model.forward_np(seq)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Step 2: Parameter-shift on quantum params ─────────────────────────
        q_params = model.qtx.get_all_params()
        P        = len(q_params)
        q_grad   = np.zeros(P)

        # Sample a few data points (expensive: 2 circuit runs per param per sample)
        sample_idx = rng.choice(N, cfg.batch_quantum, replace=False)
        # Shift only a random subset of params to keep it tractable
        shift_idx  = rng.choice(P, min(P, 12), replace=False)

        for idx in sample_idx:
            seq = X_train[idx]
            tgt = float(y_train[idx])
            for j in shift_idx:
                # Forward shift  θ_j + s
                p_plus       = q_params.copy(); p_plus[j]  += s
                model.qtx.set_all_params(p_plus)
                _, pr_plus   = model.forward_np(seq)
                loss_plus    = (pr_plus.item() - tgt) ** 2

                # Backward shift  θ_j - s
                p_minus      = q_params.copy(); p_minus[j] -= s
                model.qtx.set_all_params(p_minus)
                _, pr_minus  = model.forward_np(seq)
                loss_minus   = (pr_minus.item() - tgt) ** 2

                q_grad[j] += (loss_plus - loss_minus) / (2.0 * np.sin(s))

        # Restore original params before gradient step
        model.qtx.set_all_params(q_params)
        q_grad  /= len(sample_idx)

        # Gradient descent on quantum params
        q_params -= cfg.lr_quantum * q_grad
        model.qtx.set_all_params(q_params)

        avg_loss = epoch_loss / N
        losses.append(avg_loss)

        if verbose and (epoch % 5 == 0 or epoch == cfg.n_epochs - 1):
            print(f"  Epoch {epoch+1:3d}/{cfg.n_epochs}  |  Loss: {avg_loss:.6f}"
                  f"  |  ‖∇q‖: {np.linalg.norm(q_grad):.4f}")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# 7. DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def make_dataset(
    n_samples: int,
    seq_len:   int,
    horizon:   int = 1,
    noise:     float = 0.04,
    seed:      int = 0,
    mode:      str = "multifreq",     # "multifreq" | "lorenz"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a normalised time series dataset.

    Modes:
      multifreq  — superposition of incommensurate sine waves (default)
      lorenz     — first component of the Lorenz attractor (chaotic)
    """
    rng = np.random.default_rng(seed)

    if mode == "lorenz":
        series = _lorenz_series(n_samples + seq_len + horizon, rng, noise)
    else:
        t = np.linspace(0, 8 * np.pi, n_samples + seq_len + horizon)
        series  = np.sin(t)
        series += 0.5  * np.sin(2.3 * t + 0.5)
        series += 0.25 * np.sin(5.1 * t + 1.2)
        series += noise * rng.standard_normal(len(t))

    # Normalise to [0, 1]
    series = (series - series.min()) / (series.max() - series.min() + 1e-8)

    X, y = [], []
    for i in range(n_samples):
        X.append(series[i: i + seq_len])
        y.append(series[i + seq_len + horizon - 1])

    return np.array(X), np.array(y)


def _lorenz_series(n: int, rng, noise: float,
                   σ=10, ρ=28, β=8/3, dt=0.02) -> np.ndarray:
    """Integrate Lorenz system, return x-component."""
    x, y, z = 1.0, 0.0, 0.0
    xs = []
    for _ in range(n):
        dx = σ * (y - x)
        dy = x * (ρ - z) - y
        dz = x * y - β * z
        x += dx * dt; y += dy * dt; z += dz * dt
        xs.append(x)
    xs = np.array(xs)
    xs += noise * rng.standard_normal(len(xs))
    return xs


# ══════════════════════════════════════════════════════════════════════════════
# 8. VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(y_true: np.ndarray, y_pred: np.ndarray,
                 losses: List[float], cfg: QTConfig,
                 save_path: str = "qt_qrc_results.png"):
    """Two-panel figure: forecast vs true  +  training loss curve."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Panel 1: Forecast
    t = np.arange(len(y_true))
    axes[0].plot(t, y_true, color="#4fc3f7", lw=2,   label="Ground truth")
    axes[0].plot(t, y_pred, color="#ff7043", lw=2, ls="--", label="QT-QRC prediction")
    axes[0].fill_between(t, y_true, y_pred, alpha=0.15, color="#ff7043")
    axes[0].set_title("Forecast vs Ground Truth", color="white", fontsize=13)
    axes[0].set_xlabel("Time step", color="white")
    axes[0].set_ylabel("Normalised value", color="white")
    axes[0].legend(facecolor="#222", labelcolor="white")
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    axes[0].text(0.97, 0.05, f"MSE={mse:.5f}\nMAE={mae:.5f}",
                 transform=axes[0].transAxes, ha="right", va="bottom",
                 color="white", fontsize=9, fontfamily="monospace",
                 bbox=dict(boxstyle="round", fc="#222", ec="#555"))

    # Panel 2: Loss
    axes[1].plot(losses, color="#ce93d8", lw=2)
    axes[1].set_title("Training Loss (MSE)", color="white", fontsize=13)
    axes[1].set_xlabel("Epoch", color="white")
    axes[1].set_ylabel("Loss", color="white")
    axes[1].set_yscale("log")

    # Architecture label
    arch = (f"n_qubits={cfg.n_qubits}  |  "
            f"res_layers={cfg.n_res_layers}  |  "
            f"heads={cfg.n_heads}  |  seq={cfg.seq_len}")
    fig.suptitle(f"Quantum Transformer + QRC\n{arch}",
                 color="white", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure saved → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Quantum Transformer + QRC  ·  Time Series Forecasting ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    cfg = QTConfig(
        n_qubits        = 4,
        n_res_layers    = 3,
        n_heads         = 2,
        n_attn_layers   = 2,
        seq_len         = 6,
        horizon         = 1,
        n_epochs        = 30,
        lr_readout      = 5e-3,
        lr_quantum      = 0.05,
        batch_quantum   = 4,
    )

    # ── Data ────────────────────────────────────────────────────────────────
    print("[1/4] Generating dataset (multifreq sine)…")
    X_train, y_train = make_dataset(50, cfg.seq_len, cfg.horizon, seed=0)
    X_test,  y_test  = make_dataset(20, cfg.seq_len, cfg.horizon, seed=77)
    print(f"      Train {X_train.shape}  |  Test {X_test.shape}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    print("[2/4] Building model…")
    model = QuantumTransformerForecaster(cfg)
    n_q   = len(model.qtx.get_all_params())
    n_r   = sum(p.numel() for p in model.readout.parameters())
    print(f"      Quantum params (trainable) : {n_q}")
    print(f"      Reservoir params (fixed)   : —  (random unitary, seed={cfg.reservoir_seed})")
    print(f"      Readout params             : {n_r}")
    print(f"      Feature dim per token      : {cfg.feature_dim}\n")

    # ── Train ────────────────────────────────────────────────────────────────
    print("[3/4] Training (hybrid: param-shift ⊕ Adam)…")
    losses = train(model, X_train, y_train, cfg, verbose=True)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set…")
    preds = np.array([model.predict(X_test[i])[0] for i in range(len(X_test))])
    mse   = float(np.mean((preds - y_test) ** 2))
    mae   = float(np.mean(np.abs(preds - y_test)))
    r2    = float(1 - np.sum((y_test - preds)**2) / np.sum((y_test - y_test.mean())**2))
    print(f"      MSE : {mse:.6f}")
    print(f"      MAE : {mae:.6f}")
    print(f"      R²  : {r2:.4f}")

    plot_results(y_test, preds, losses, cfg,
                 save_path="quantum_transformer_qrc_results.png")

    print("\n✓ Done.")
    return model, losses, preds, y_test


if __name__ == "__main__":
    model, losses, preds, y_test = main()
