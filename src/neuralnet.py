import io
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import json
import os

from rdkit import Chem
from rdkit.Chem import AllChem

from pycuda import gpuarray
import pycuda.autoinit

from cuda_kernels import (
    get_add_bias_kernel,
    get_apply_weights_biases_kernel,
    get_compute_delta_output_kernel,
    get_sum_delta_rows_kernel,
    get_compute_act_deriv_kernel,
    get_matrix_multiply_kernel,
    get_transpose_matrix_kernel,
)

# TO-DOs
#   Profile existing code on perlmutter, generate roofline models
#   Add additional control to argparse for training params
#   Brainstorm different models
#   Allow a json file to specify the net layers
#   Complete discussion of results

np.random.seed(42)


class InputLayer:
    def __init__(self, size, batch_size, device):
        self.size = size
        self.current_batch = batch_size
        self.device = device
        self.activations = np.zeros((self.size,), dtype=np.float32)
        if self.device == "gpu":
            self.activations_gpu = gpuarray.zeros(
                (self.size, batch_size), dtype=np.float32
            )
            # Same layout as Layer.activations_transposed_gpu; used in bp_gpu on layer 1.
            self.activations_transposed_gpu = gpuarray.empty(
                (batch_size, self.size), dtype=np.float32
            )


class Layer:
    def __init__(self, size, previous_layer, batch_size, device):
        self.size = size
        self.previous_layer = previous_layer
        self.next_layer = None
        self.previous_layer.next_layer = self
        self.batch_size = batch_size
        self.current_batch = batch_size
        self.device = device

        # He initialization
        stdev = np.sqrt(2.0 / self.previous_layer.size)
        # self.biases = np.random.normal(0.0, stdev, (self.size,)).astype(np.float32)
        self.biases = np.random.normal(0.0, stdev, (self.size, 1)).astype(np.float32)
        self.weights = np.random.normal(
            0.0, stdev, (self.size, self.previous_layer.size)
        ).astype(np.float32)

        # The activations, before applying sigmoid
        self.preactivations = np.zeros((self.size,), dtype=np.float32)
        self.activations = np.zeros((self.size,), dtype=np.float32)
        self.weights_grad = np.zeros(
            (self.size, self.previous_layer.size), dtype=np.float32
        )
        self.biases_grad = np.zeros((self.size, 1), dtype=np.float32)
        self.delta = np.zeros((self.size,), dtype=np.float32)

        # send data to gpu device if specified
        if self.device == "gpu":
            from pycuda import gpuarray

            self.tile_size = 32
            self.blocksize = self.tile_size

            self.biases_gpu = gpuarray.to_gpu(self.biases)
            self.weights_gpu = gpuarray.to_gpu(self.weights)

            self.preactivations_gpu = gpuarray.zeros(
                (self.size, batch_size), dtype=np.float32
            )
            self.activations_gpu = gpuarray.zeros(
                (self.size, batch_size), dtype=np.float32
            )
            self.weights_grad_gpu = gpuarray.to_gpu(self.weights_grad)
            self.biases_grad_gpu = gpuarray.to_gpu(self.biases_grad)
            self.delta_gpu = gpuarray.zeros((self.size, batch_size), dtype=np.float32)

            # Temp and tranpose buffers
            self.weights_transposed_gpu = gpuarray.empty(
                (self.previous_layer.size, self.size), dtype=np.float32
            )
            self.activations_transposed_gpu = gpuarray.empty(
                (batch_size, self.size), dtype=np.float32
            )
            self.delta_temp_gpu = gpuarray.empty(
                (self.size, batch_size), dtype=np.float32
            )

            # Get all cuda kernels for gpu implementation
            self.add_bias = get_add_bias_kernel()
            self.apply_weights_biases = get_apply_weights_biases_kernel()
            self.compute_delta_output = get_compute_delta_output_kernel()
            self.sum_delta_rows = get_sum_delta_rows_kernel()
            self.compute_act_deriv = get_compute_act_deriv_kernel()
            self.matrix_multiply = get_matrix_multiply_kernel()
            self.transpose_matrix = get_transpose_matrix_kernel()

    def feedforward(self):
        if self.device == "gpu":
            self.ff_gpu()
        else:
            self.ff_cpu()

    def ff_cpu(self):
        self.preactivations = (
            np.dot(self.weights, self.previous_layer.activations) + self.biases
        )

        if self.next_layer is None:  # Just do linear output
            self.activations = self.preactivations
        else:
            # Apply the softplus activation function
            # self.activations = np.maximum(0, self.preactivations)  # ReLU
            self.activations = np.where(
                self.preactivations > 20.0,
                self.preactivations,
                np.log(1.0 + np.exp(self.preactivations)),
            )

    def ff_gpu(self):
        M = self.size
        N = self.previous_layer.current_batch
        K = self.previous_layer.size

        # grid dimensions
        grid_x = math.ceil(N / self.tile_size)
        grid_y = math.ceil(M / self.tile_size)

        self.matrix_multiply(
            self.weights_gpu,
            self.previous_layer.activations_gpu,
            self.preactivations_gpu,
            np.int32(M),
            np.int32(N),
            np.int32(K),
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )

        # Only if hidden layer apply relu
        if self.next_layer is None:
            relu_flag = np.int32(0)
        else:
            relu_flag = np.int32(1)

        self.add_bias(
            self.preactivations_gpu,
            self.biases_gpu,
            self.activations_gpu,
            np.int32(M),
            np.int32(N),
            relu_flag,
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )

    def backpropagation(self, reference, reference_gpu=None):
        if self.device == "gpu":
            if reference_gpu is None:
                reference_gpu = gpuarray.to_gpu(reference)
            self.bp_gpu(reference_gpu)
        else:
            self.bp_cpu(reference)

    def bp_cpu(self, reference):
        if self.next_layer is None:  # Output layer
            # Get this from differentiating the cost function
            self.delta = self.activations - reference
        else:  # Hidden layer
            # Get this from differentiating the activation function
            # Now differentiating the softplus (sigmoid)
            # activation_grad = np.where(self.preactivations > 0, 1.0, 0.0)  # ReLU
            activation_grad = 1.0 / (1.0 + np.exp(-self.preactivations))
            self.delta = (
                np.dot(self.next_layer.weights.transpose(), self.next_layer.delta)
                * activation_grad
            )

        self.biases_grad += np.sum(self.delta, axis=1, keepdims=True)
        self.weights_grad += np.dot(self.delta, self.previous_layer.activations.T)

    def bp_gpu(self, reference):
        M_prev = self.previous_layer.size
        M_curr = self.size
        N_batch = self.current_batch

        # Output layer
        if self.next_layer is None:
            grid_x = math.ceil(N_batch / self.tile_size)
            grid_y = math.ceil(M_curr / self.tile_size)
            # Just take diff between activations and reference for output layer
            self.compute_delta_output(
                self.activations_gpu,
                reference,
                self.delta_gpu,
                np.int32(M_curr),
                np.int32(N_batch),
                block=(self.blocksize, self.blocksize, 1),
                grid=(grid_x, grid_y, 1),
            )
        # Hidden layer
        else:
            M_next = self.next_layer.size
            # transpose next layer's weights (kernel: x spans N=M_curr cols, y spans M=M_next rows)
            grid_x = math.ceil(M_curr / self.tile_size)
            grid_y = math.ceil(M_next / self.tile_size)
            self.transpose_matrix(
                self.next_layer.weights_gpu,
                self.next_layer.weights_transposed_gpu,
                np.int32(M_next),
                np.int32(M_curr),
                block=(self.blocksize, self.blocksize, 1),
                grid=(grid_x, grid_y, 1),
            )

            # multiply weights by delta to get temp_delta
            grid_x = math.ceil(N_batch / self.tile_size)
            grid_y = math.ceil(M_curr / self.tile_size)
            self.matrix_multiply(
                self.next_layer.weights_transposed_gpu,
                self.next_layer.delta_gpu,
                self.delta_temp_gpu,
                np.int32(M_curr),
                np.int32(N_batch),
                np.int32(M_next),
                block=(self.blocksize, self.blocksize, 1),
                grid=(grid_x, grid_y, 1),
            )

            # take ReLU deriv
            self.compute_act_deriv(
                self.delta_temp_gpu,
                self.preactivations_gpu,
                self.biases_gpu,
                self.delta_gpu,
                np.int32(M_curr),
                np.int32(N_batch),
                block=(self.blocksize, self.blocksize, 1),
                grid=(grid_x, grid_y, 1),
            )

        # For all layers calculate the gradients
        # First transpose the previous layers activations (kernel: x spans N_batch, y spans M_prev)
        grid_x = math.ceil(N_batch / self.tile_size)
        grid_y = math.ceil(M_prev / self.tile_size)
        self.transpose_matrix(
            self.previous_layer.activations_gpu,
            self.previous_layer.activations_transposed_gpu,
            np.int32(M_prev),
            np.int32(N_batch),
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )

        # Multiply the delta and transposed activations to get weights grad
        self.weights_grad_gpu.fill(0)
        grid_x = math.ceil(M_prev / self.tile_size)
        grid_y = math.ceil(M_curr / self.tile_size)
        self.matrix_multiply(
            self.delta_gpu,
            self.previous_layer.activations_transposed_gpu,
            self.weights_grad_gpu,
            np.int32(M_curr),
            np.int32(M_prev),
            np.int32(N_batch),
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )

        self.biases_grad_gpu.fill(0)
        grid_x = math.ceil(M_curr / self.tile_size)
        self.sum_delta_rows(
            self.delta_gpu,
            self.biases_grad_gpu,
            np.int32(M_curr),
            np.int32(N_batch),
            block=(self.blocksize, 1, 1),  # 1D block
            grid=(grid_x, 1, 1),  # 1D grid
        )

    def apply_gradient(self, batch_size, training_rate):
        if self.device == "gpu":
            self.grad_gpu(batch_size, training_rate)
        else:
            self.grad_cpu(batch_size, training_rate)

    def grad_cpu(self, batch_size, training_rate):
        self.weights = self.weights - (training_rate / batch_size) * self.weights_grad
        self.biases = self.biases - (training_rate / batch_size) * self.biases_grad
        self.weights_grad.fill(0.0)
        self.biases_grad.fill(0.0)

    def grad_gpu(self, batch_size, training_rate):
        M = self.size
        K = self.previous_layer.size

        grid_x = math.ceil(K / self.blocksize)
        grid_y = math.ceil(M / self.blocksize)

        self.apply_weights_biases(
            self.weights_gpu,
            self.weights_grad_gpu,
            self.biases_gpu,
            self.biases_grad_gpu,
            np.int32(batch_size),
            np.int32(M),
            np.int32(K),
            np.float32(training_rate),
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )


class Network:
    def __init__(self, sizes, training_inputs, training_references, batch_size, device):
        self.layers = []
        self.training_inputs = training_inputs
        self.training_references = training_references
        self.device = device
        self.batch_size = batch_size

        new_layer = InputLayer(sizes[0], batch_size, device)
        self.layers.append(new_layer)
        for ilayer in range(1, len(sizes)):
            self.layers.append(
                Layer(sizes[ilayer], self.layers[-1], batch_size, device)
            )

    def train(self, nepochs, training_rate):

        # Only pass from CPU to GPU twice per batch
        # current inputs/references to GPU
        # send back loss to cpu for each batch

        ninputs = len(self.training_inputs)
        feedforward_time = 0.0
        backpropagation_time = 0.0
        for iepoch in range(nepochs):
            loss = 0.0
            indices = np.random.permutation(ninputs)
            inputs_shuffled = self.training_inputs[indices]
            refs_shuffled = self.training_references[indices]

            for istart in range(0, ninputs, self.batch_size):
                iend = min(istart + self.batch_size, ninputs)

                current_batch = iend - istart

                self.layers[0].current_batch = current_batch
                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].current_batch = current_batch

                # shape -> (n_features, batch)
                # Morse: 1 feature & AB3: 6 features.
                batch_refs = refs_shuffled[istart:iend]
                references = batch_refs.reshape(1, current_batch)
                batch_inputs = inputs_shuffled[istart:iend]
                if batch_inputs.ndim == 1:
                    self.layers[0].activations = batch_inputs.reshape(1, current_batch)
                else:
                    transposed_inputs = batch_inputs.T.astype(np.float32)
                    self.layers[0].activations = np.ascontiguousarray(transposed_inputs)

                if self.device == "gpu":
                    self.layers[0].activations_gpu.set(self.layers[0].activations)
                    reference_gpu = gpuarray.to_gpu(references)
                else:
                    reference_gpu = None

                # Feedforward through the other layers
                start_time = time.time()
                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].feedforward()

                feedforward_time += time.time() - start_time

                # Do backpropagation
                start_time = time.time()
                for ilayer in range(len(self.layers) - 1):
                    self.layers[-1 - ilayer].backpropagation(references, reference_gpu)
                backpropagation_time += time.time() - start_time

                # Calculate loss (after backprop: output delta = activations - reference)
                if self.device == "gpu":
                    squared_error_gpu = self.layers[-1].delta_gpu ** 2
                    batch_loss_gpu = gpuarray.sum(squared_error_gpu)
                    loss += float(batch_loss_gpu.get())
                else:
                    loss += np.sum((self.layers[-1].activations - references) ** 2)

                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].apply_gradient(current_batch, training_rate)

            standard_deviation = math.sqrt(loss / ninputs)
            print(f"Epoch, deviation: {iepoch}, {standard_deviation}")

        print(f"Feedforward time: {feedforward_time}")
        print(f"Backpropagation time: {backpropagation_time}")

    def save_weights(self, filename, norm=None):
        model_dict = {}
        # Starting after input layer, save weights/biases
        for i, layer in enumerate(self.layers[1:]):
            if layer.device == "gpu":
                layer.weights = layer.weights_gpu.get()
                layer.biases = layer.biases_gpu.get()
            model_dict[f"layer_{i}_weights"] = layer.weights
            model_dict[f"layer_{i}_biases"] = layer.biases

        if norm is not None:
            model_dict["mean_e"] = np.float32(norm["mean_e"])
            model_dict["std_e"] = np.float32(norm["std_e"])

        # np.savez stores multiple arrays into single file
        np.savez_compressed(filename, **model_dict)
        print(f"Model saved to {filename}")

    def load_weights(self, filename):
        with np.load(filename) as model:
            for i, layer in enumerate(self.layers[1:]):
                layer.weights = np.asarray(
                    model[f"layer_{i}_weights"], dtype=np.float32
                )
                layer.biases = np.asarray(model[f"layer_{i}_biases"], dtype=np.float32)
                if layer.device == "gpu":
                    layer.weights_gpu.set(layer.weights)
                    layer.biases_gpu.set(layer.biases)
            norm = None
            if "mean_e" in model and "std_e" in model:
                norm = {
                    "mean_e": float(model["mean_e"]),
                    "std_e": float(model["std_e"]),
                }
        print(f"Model weights loaded from {filename}")
        return norm


def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner


def load_setup(setup_path):
    path = Path(setup_path)
    if not path.is_file():
        fallback = Path("setup") / path.name
        if fallback.is_file():
            path = fallback
            print(
                f"Note: use setup/{path.name} on the command line "
                "(bare filename still accepted)"
            )
        else:
            raise FileNotFoundError(f"Setup file not found: {setup_path}")

    Path("setup").mkdir(parents=True, exist_ok=True)

    with open(path, "r") as file:
        try:
            run_data = json.load(file)
            print(f"Run setup parsed: {path}")
        except json.JSONDecodeError as err:
            raise ValueError(f"Could not parse setup JSON: {path}") from err

    required = {
        "run",
        "problem",
        "ninputs",
        "batch_size",
        "epochs",
        "training_rate",
        "architecture",
    }

    missing = required - run_data.keys()
    if missing:
        raise KeyError(f"Missing required setup keys: {missing}")

    return run_data


class Morse:
    def __init__(self):
        self.De = 1.0
        self.re = 1.0
        self.a = 1.0
        self.min_rvalue = 0.5
        self.max_rvalue = 2.0

    def _normalize_r(self, rvalues):
        return (rvalues - (self.max_rvalue + self.min_rvalue) / 2.0) / (
            self.max_rvalue - self.min_rvalue
        )

    def training_data(self, ninputs):
        rvalues = np.random.uniform(
            self.min_rvalue, self.max_rvalue, (ninputs,)
        ).astype(np.float32)
        erefs = np.array(
            [morse_potential(self.De, self.re, self.a, r) for r in rvalues],
            dtype=np.float32,
        )
        mean_e = np.mean(erefs)
        std_e = np.std(erefs)
        return (
            self._normalize_r(rvalues),
            (erefs - mean_e) / std_e,
            {"mean_e": mean_e, "std_e": std_e},
        )

    def eval_grid(self, n_test=300):
        """Sorted r grid for plotting (not the random training set)."""
        r_test = np.linspace(self.min_rvalue, self.max_rvalue, n_test, dtype=np.float32)
        r_test_norm = self._normalize_r(r_test).reshape(1, n_test)
        true_energy = np.array(
            [morse_potential(self.De, self.re, self.a, r) for r in r_test],
            dtype=np.float32,
        )
        return r_test, r_test_norm, true_energy

    def eval_curves(self, n_test=300):
        x, x_norm, true_e = self.eval_grid(n_test)
        yield "default", x, x_norm, true_e  # x_norm is (1, n_test)


class AB3:
    def __init__(self):
        # Using NH3 as model AB3 mol for training
        self.mol = Chem.MolFromSmiles("N")
        self.mol = Chem.AddHs(self.mol)

        # Setup the Merck molecular force field properties
        self.mp = AllChem.MMFFGetMoleculeProperties(self.mol)

        # 3D coordinates do not exist until after embedding (GetConformer fails before this).
        params = AllChem.ETKDGv3()
        if AllChem.EmbedMolecule(self.mol, params) != 0:
            raise RuntimeError("RDKit failed to embed NH3 3D geometry")

        self.conf = self.mol.GetConformer()
        self.base_pos = np.array(self.conf.GetPositions(), dtype=np.float64)

    def perturb_structure(self):
        # scale = np.random.uniform(0.75, 2.1)
        # A_atom = self.base_pos[0:1, :]
        # B_atoms = self.base_pos[1:, :]
        # base_bond_vecs = B_atoms - A_atom

        # # apply random scale ot the bond vecs
        # scaled_B_atoms = A_atom + (base_bond_vecs * scale)
        # scaled_pos = np.vstack([A_atom, scaled_B_atoms])

        # # perturb from 0 to 0.15 angstroms for all positions
        # perturbation = np.random.normal(0, 0.15, scaled_pos.shape)
        # new_pos = scaled_pos + perturbation
        # self.conf.SetPositions(new_pos)

        # generate some random angles for training
        theta = np.random.uniform(np.radians(85), np.radians(125))
        cos_sq_beta = np.clip((np.cos(theta) + 0.5) / 1.5, 0.0, 1.0)
        beta = np.arccos(np.sqrt(cos_sq_beta))

        # set normal dist around equilibrium length
        # clip slightly larger than evaluation scale of 0.8 to 2.0
        scale = np.random.normal(1.0, 0.3)
        scale = np.clip(scale, 0.75, 2.1)

        # get eq length from the base geometry
        A_atom = self.base_pos[0]
        B_atoms = self.base_pos[1:]
        eq_length = np.mean(np.linalg.norm(B_atoms - A_atom, axis=-1))
        r = eq_length * scale

        # build synthetic base coords to then perturb
        new_pos = np.zeros((4, 3))
        phi = np.radians([0, 120, 240])
        new_pos[1:, 0] = r * np.sin(beta) * np.cos(phi)
        new_pos[1:, 1] = r * np.sin(beta) * np.sin(phi)
        new_pos[1:, 2] = r * np.cos(beta)

        perturbation = np.random.normal(0, 0.05, new_pos.shape)
        self.conf.SetPositions(new_pos + perturbation)

    def get_ff_energy(self):
        ff = AllChem.MMFFGetMoleculeForceField(self.mol, self.mp)
        return ff.CalcEnergy()

    def training_data(self, ninputs):
        energies = np.zeros(ninputs)
        coords = np.zeros((ninputs, self.mol.GetNumAtoms(), 3))

        for i in range(ninputs):
            self.perturb_structure()
            energy = self.get_ff_energy()
            energies[i] = energy
            coords[i] = self.conf.GetPositions()

        # Generate all pairwise bond lengths
        A_atom = coords[:, 0:1, :]
        B_atoms = coords[:, 1:, :]

        bond_vecs = B_atoms - A_atom
        lengths = np.linalg.norm(bond_vecs, axis=-1)

        # Generate all pairwise bond angles
        norm_vecs = bond_vecs / lengths[:, :, np.newaxis]
        dot_B1_A_B2 = np.sum(norm_vecs[:, 0, :] * norm_vecs[:, 1, :], axis=-1)
        dot_B1_A_B3 = np.sum(norm_vecs[:, 0, :] * norm_vecs[:, 2, :], axis=-1)
        dot_B2_A_B3 = np.sum(norm_vecs[:, 1, :] * norm_vecs[:, 2, :], axis=-1)

        dots = np.stack([dot_B1_A_B2, dot_B1_A_B3, dot_B2_A_B3], axis=-1)
        angles = np.arccos(np.clip(dots, -1.0, 1.0))

        # combine lenghts and angles into flat features for model
        features = np.concatenate([lengths, angles], axis=-1)

        mean_e, std_e = energies.mean(), energies.std()
        targets = (energies - mean_e) / std_e
        norm = {"mean_e": mean_e, "std_e": std_e}

        return features.astype(np.float32), targets.astype(np.float32), norm

    def eval_symmetric_stretch(self, n_test=300):
        # Stretching scale (80% to 200%)
        scales = np.linspace(0.8, 2.0, n_test)

        true_energies = np.zeros(n_test, dtype=np.float32)
        features = np.zeros((n_test, 6), dtype=np.float32)

        # Get length and angle components from base geometry
        A_atom = self.base_pos[0:1, :]
        B_atoms = self.base_pos[1:, :]
        base_bond_vecs = B_atoms - A_atom
        base_lengths = np.linalg.norm(base_bond_vecs, axis=-1)
        norm_vecs = base_bond_vecs / base_lengths[:, np.newaxis]

        dot_B1_A_B2 = np.sum(norm_vecs[0] * norm_vecs[1])
        dot_B1_A_B3 = np.sum(norm_vecs[0] * norm_vecs[2])
        dot_B2_A_B3 = np.sum(norm_vecs[1] * norm_vecs[2])
        base_angles = np.arccos(
            np.clip([dot_B1_A_B2, dot_B1_A_B3, dot_B2_A_B3], -1.0, 1.0)
        )

        for i, scale in enumerate(scales):
            new_vecs = base_bond_vecs * scale
            new_pos = np.vstack([A_atom, A_atom + new_vecs])

            # Calc true energy from stretched mol
            self.conf.SetPositions(new_pos)
            true_energies[i] = self.get_ff_energy()

            # Build the features for net
            current_lengths = base_lengths * scale
            features[i] = np.concatenate([current_lengths, base_angles])

        return scales, features, true_energies

    def eval_symmetric_bend(self, n_test=300):
        # Strategy is to keep B-atoms in a ring spaced 120deg apart
        # Change the angle that the ring is relative to the central A atom
        # Should find optimal A-B angle this way

        # Evaluating angles from 90 to 120 degrees
        angles_deg = np.linspace(90, 120, n_test)
        angles_rad = np.radians(angles_deg)

        true_energies = np.zeros(n_test, dtype=np.float32)
        features = np.zeros((n_test, 6), dtype=np.float32)

        # Get equilibrium bond length from base geometry
        A_atom = self.base_pos[0]
        B_atoms = self.base_pos[1:]
        eq_length = np.mean(np.linalg.norm(B_atoms - A_atom, axis=-1))

        # Angles for B-atoms spaced 120 degrees apart
        phi = np.radians([0, 120, 240])

        for i, theta in enumerate(angles_rad):
            # Target B-A-B bond angle is related to beta by:
            # cos^2(Beta) = cos(theta) + 0.5 / 1.5
            cos_sq_beta = (np.cos(theta) + 0.5) / 1.5
            cos_sq_beta = np.clip(cos_sq_beta, 0.0, 1.0)
            # sqrt previous term to reveal beta
            beta = np.arccos(np.sqrt(cos_sq_beta))

            # generate the new B atom coords
            # A atom stays fixed at 0,0,0
            new_pos = np.zeros((4, 3))
            new_pos[1:, 0] = eq_length * np.sin(beta) * np.cos(phi)  # X
            new_pos[1:, 1] = eq_length * np.sin(beta) * np.sin(phi)  # Y
            new_pos[1:, 2] = eq_length * np.cos(beta)  # Z

            # Get true energy using these new coords
            self.conf.SetPositions(new_pos)
            true_energies[i] = self.get_ff_energy()

            lengths = np.array([eq_length, eq_length, eq_length])
            angles = np.array([theta, theta, theta])
            features[i] = np.concatenate([lengths, angles])

        return angles_deg, features, true_energies

    def evaluations(self):
        return {
            "stretch": self.eval_symmetric_stretch,
            "bend": self.eval_symmetric_bend,
        }

    def eval_curves(self, n_test=300):
        for name, func in self.evaluations().items():
            x, features, true_e = func(n_test)
            yield name, x, features, true_e


def predict_curve(net, inputs, n_test, norm):
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    for layer in net.layers:
        layer.current_batch = n_test
    for layer in net.layers[1:]:
        layer.device = "cpu"

    net.layers[0].activations = inputs
    for ilayer in range(1, len(net.layers)):
        net.layers[ilayer].feedforward()

    pred_norm = net.layers[-1].activations[0, :n_test]
    return pred_norm * norm["std_e"] + norm["mean_e"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or evaluate neural net")

    parser.add_argument(
        "setup",
        help="Path to setup JSON (e.g. setup/ab3_basic.json)",
    )

    # Run mode (train or evaluate)
    parser.add_argument(
        "-t", "--train", action="store_true", help="Sets model to train"
    )
    parser.add_argument(
        "-e", "--eval", action="store_true", help="Set model to evaluate"
    )

    # Device mode (CPU or GPU) for training
    parser.add_argument("--cpu", action="store_true", help="Sets device to CPU")
    parser.add_argument("--gpu", action="store_true", help="Sets device to GPU")
    parser.add_argument(
        "--weights",
        default=None,
        help="Checkpoint .npz path (default: models/{run}_{device}.npz)",
    )

    args = parser.parse_args()

    # Extract values and ensure that requirements are met
    data = load_setup(args.setup)
    run = data["run"]
    problem = data["problem"]
    ninputs = data["ninputs"]
    batch_size = data["batch_size"]
    epochs = data["epochs"]
    training_rate = data["training_rate"]
    architecture = data["architecture"]

    # Check to make sure that ninputs and batch size are divisible
    if ninputs % batch_size != 0:
        raise ValueError("ninputs for training must be divisble by batch_size")

    # Set device for training or evaluation
    if args.cpu:
        device = "cpu"
        print("Running on device = CPU")
    elif args.gpu:
        device = "gpu"
        print("Running on device = GPU")
    else:
        device = "cpu"  # cpu by default
        print("Running on default device = CPU")

    # Checkpoint path (device suffix only names the default file for train/save)
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    filename = args.weights or f"models/{run}_{device}.npz"
    checkpoint_label = Path(filename).stem

    # Train the model
    if args.train:
        print(f"training rate: {training_rate}")

        if problem == "Morse":
            problem_obj = Morse()
        elif problem == "AB3":
            problem_obj = AB3()
        else:
            raise ValueError(f"Problem type not recognized: {problem}")

        inputs, refs, norm = problem_obj.training_data(ninputs)
        net = Network(architecture, inputs, refs, batch_size, device)
        start_time = time.time()
        net.train(epochs, training_rate)
        print(f"Training time: {time.time() - start_time}")
        net.save_weights(filename, norm=norm)

    # Evaluate model performance and plot
    elif args.eval:

        n_test = 300
        if problem == "Morse":
            problem_obj = Morse()
        elif problem == "AB3":
            problem_obj = AB3()
        else:
            raise ValueError(f"Problem type not recognized: {problem}")

        # Eval always runs forward on CPU; weights in .npz are device-agnostic.
        print(f"Loading checkpoint: {filename}")
        net_batch = max(batch_size, n_test)
        n_features = architecture[0]
        net = Network(
            architecture,
            np.zeros((n_test, n_features), dtype=np.float32),
            np.zeros(n_test, dtype=np.float32),
            net_batch,
            "cpu",
        )
        norm = net.load_weights(filename)
        if norm is None:
            raise ValueError(
                f"Missing mean_e/std_e in {filename}; retrain with current code."
            )

        for name, x, features, true_e in problem_obj.eval_curves(n_test):
            # Network expects (n_features, n_test)
            if features.ndim == 1:
                inputs = features.reshape(1, n_test)
            elif features.shape[0] == n_test:
                inputs = features.T.astype(np.float32)
            else:
                inputs = np.ascontiguousarray(features, dtype=np.float32)

            # generate prediction curves
            pred_e = predict_curve(net, inputs, n_test, norm)
            rmse = np.sqrt(np.mean((true_e - pred_e) ** 2))

            plt.clf()
            plt.plot(x, true_e, label="true")
            plt.plot(x, pred_e, label="pred")

            if name == "stretch" or name == "bend":
                y_units = "kcal/mol"
            else:
                y_units = "eV"

            if name == "stretch" or name == "default":
                x_units = "Angstroms"
            else:
                x_units = "Degrees"

            plt.title(f"{checkpoint_label} {name}\nRMSE: {rmse}")
            plt.xlabel(x_units)
            plt.ylabel(f"Energy ({y_units})")
            plt.legend()

            plot_path = Path("plots")
            plot_path.mkdir(parents=True, exist_ok=True)
            plot_name = plot_path / f"{checkpoint_label}_{name}.png"
            plt.savefig(plot_name)
            plt.close()

    # User did not specify a run mode, exit safely
    else:
        sys.exit("Run mode not specified (--train or --eval)")
