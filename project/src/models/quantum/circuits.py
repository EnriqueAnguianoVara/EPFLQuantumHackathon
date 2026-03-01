"""
Reusable quantum circuit builders for MerLin / Perceval.

Provides factory functions for:
- Fixed reservoir circuits (no trainable parameters)
- Trainable autoencoder bottleneck circuits
- Noisy experiment wrappers for ablation studies
"""

import numpy as np
import torch
import perceval as pcvl

from merlin import QuantumLayer, MeasurementStrategy, ComputationSpace
from merlin.builder import CircuitBuilder


# ---------------------------------------------------------------------------
# Reservoir circuit (FIXED — no trainable parameters)
# ---------------------------------------------------------------------------
def build_reservoir_circuit(
    n_modes: int = 8,
    n_photons: int = 3,
    input_size: int = 6,
    seed: int = 42,
    depth: int = 2,
) -> dict:
    """
    Build a fixed (non-trainable) photonic reservoir circuit.

    The reservoir has random but fixed interferometers (numeric values,
    not symbolic parameters), followed by angle encoding of the input data.
    The output is a probability distribution over Fock states.

    Parameters
    ----------
    n_modes : number of optical modes
    n_photons : number of photons
    input_size : number of input features (PCA components)
    seed : random seed for reproducible reservoir
    depth : number of fixed interferometer layers

    Returns
    -------
    dict with keys:
        'layer': QuantumLayer (nn.Module, no trainable params)
        'output_size': int (dimension of Fock probability vector)
        'n_modes': int
        'n_photons': int
    """
    rng = np.random.RandomState(seed)

    circuit = pcvl.Circuit(n_modes)

    # Build fixed interferometers with NUMERIC values (not symbolic parameters)
    for d in range(depth):
        for i in range(n_modes - 1):
            circuit.add((i, i + 1), pcvl.BS())
            circuit.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))
        for i in range(1, n_modes - 1):
            circuit.add((i, i + 1), pcvl.BS())
            circuit.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))

    # Angle encoding on the first input_size modes (symbolic — these ARE the input)
    encoding_modes = min(input_size, n_modes)
    for mode in range(encoding_modes):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))

    # Post-encoding fixed interferometer
    for i in range(n_modes - 1):
        circuit.add((i, i + 1), pcvl.BS())
        circuit.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))
    for i in range(1, n_modes - 1):
        circuit.add((i, i + 1), pcvl.BS())
        circuit.add(i, pcvl.PS(float(rng.uniform(0, 2 * np.pi))))

    # Input state
    input_state = [1] * n_photons + [0] * (n_modes - n_photons)

    # Create QuantumLayer — only "input" parameters, nothing trainable
    layer = QuantumLayer(
        input_size=input_size,
        circuit=circuit,
        input_state=input_state,
        trainable_parameters=[],
        input_parameters=["input"],
        dtype=torch.float32,
    )

    return {
        "layer": layer,
        "output_size": layer.output_size,
        "n_modes": n_modes,
        "n_photons": n_photons,
        "circuit": circuit,
    }


# ---------------------------------------------------------------------------
# Autoencoder bottleneck circuit (TRAINABLE)
# ---------------------------------------------------------------------------
def build_autoencoder_circuit(
    n_modes: int = 6,
    n_photons: int = 3,
    input_size: int = 6,
    grouping_size: int = 8,
    depth: int = 1,
) -> dict:
    """
    Build a trainable photonic circuit for the autoencoder bottleneck.

    Architecture:
        trainable interferometer -> angle encoding -> trainable rotations -> superpositions

    Parameters
    ----------
    n_modes : number of optical modes (should match input_size)
    n_photons : number of photons
    input_size : features entering the quantum layer
    grouping_size : output dimension after LexGrouping
    depth : number of trainable layers after encoding

    Returns
    -------
    dict with keys:
        'layer': QuantumLayer (nn.Module, trainable)
        'raw_output_size': int (Fock space dimension before grouping)
        'n_modes': int
        'n_photons': int
    """
    builder = CircuitBuilder(n_modes=n_modes)

    # Pre-encoding trainable interferometer
    builder.add_entangling_layer(trainable=True, name="U_pre")

    # Angle encoding of input features
    encoding_modes = list(range(min(input_size, n_modes)))
    builder.add_angle_encoding(modes=encoding_modes, name="input")

    # Post-encoding trainable layers
    for d in range(depth):
        builder.add_rotations(trainable=True, name=f"theta_{d}")
        if d < depth - 1:
            builder.add_superpositions(depth=1)

    # Final mixing
    builder.add_superpositions(depth=1)

    # Create QuantumLayer
    layer = QuantumLayer(
        input_size=input_size,
        builder=builder,
        n_photons=n_photons,
        dtype=torch.float32,
    )

    return {
        "layer": layer,
        "raw_output_size": layer.output_size,
        "n_modes": n_modes,
        "n_photons": n_photons,
    }


# ---------------------------------------------------------------------------
# Variational circuit (TRAINABLE — more general)
# ---------------------------------------------------------------------------
def build_variational_circuit(
    n_modes: int = 6,
    n_photons: int = 3,
    input_size: int = 6,
    n_layers: int = 2,
) -> dict:
    """
    Build a deeper variational circuit for general-purpose QML.

    Architecture:
        (entangling + encoding + rotations) x n_layers + superpositions

    Parameters
    ----------
    n_modes : optical modes
    n_photons : photons
    input_size : classical input dimension
    n_layers : number of variational layers (encoding is repeated)

    Returns
    -------
    dict with 'layer', 'raw_output_size', 'n_modes', 'n_photons'
    """
    builder = CircuitBuilder(n_modes=n_modes)
    encoding_modes = list(range(min(input_size, n_modes)))

    builder.add_entangling_layer(trainable=True, name="U_0")
    builder.add_angle_encoding(modes=encoding_modes, name="input")

    for i in range(n_layers):
        builder.add_rotations(trainable=True, name=f"rot_{i}")
        builder.add_entangling_layer(trainable=True, name=f"U_{i+1}")
        
    builder.add_superpositions(depth=1)

    layer = QuantumLayer(
        input_size=input_size,
        builder=builder,
        n_photons=n_photons,
        dtype=torch.float32,
    )

    return {
        "layer": layer,
        "raw_output_size": layer.output_size,
        "n_modes": n_modes,
        "n_photons": n_photons,
    }


# ---------------------------------------------------------------------------
# Noisy experiment wrapper (for ablation studies)
# ---------------------------------------------------------------------------
def wrap_with_noise(
    circuit: pcvl.Circuit,
    input_size: int,
    input_state: list,
    brightness: float = 0.8,
    transmittance: float = 0.9,
    detector_type: str = "threshold",
    trainable_parameters: list = None,
) -> QuantumLayer:
    """
    Wrap a Perceval circuit with noise model and detectors.

    Parameters
    ----------
    circuit : perceval.Circuit
    input_size : number of classical input features
    input_state : photon input configuration
    brightness : source brightness (0-1)
    transmittance : channel transmittance (0-1)
    detector_type : 'threshold' or 'pnr'
    trainable_parameters : list of parameter prefixes to train

    Returns
    -------
    QuantumLayer with noise model attached
    """
    experiment = pcvl.Experiment(circuit)
    experiment.noise = pcvl.NoiseModel(
        brightness=brightness,
        transmittance=transmittance,
    )

    n_modes = circuit.m
    for mode in range(n_modes):
        if detector_type == "threshold":
            experiment.detectors[mode] = pcvl.Detector.threshold()
        else:
            experiment.detectors[mode] = pcvl.Detector.pnr()

    layer = QuantumLayer(
        input_size=input_size,
        experiment=experiment,
        input_state=input_state,
        input_parameters=["input"],
        trainable_parameters=trainable_parameters or [],
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        computation_space=ComputationSpace.FOCK,
        dtype=torch.float32,
    )

    return layer
