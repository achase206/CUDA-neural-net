import numpy as np
import time
import math
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys
import json
import os

from pycuda import gpuarray
import pycuda.autoinit

from cuda_kernels import (
    get_add_bias_kernel,
    get_apply_weights_biases_kernel,
    get_compute_delta_output_kernel,
    get_sum_delta_rows_kernel,
    get_compute_ReLU_deriv_kernel,
    get_matrix_multiply_kernel,
    get_transpose_matrix_kernel,
)

# TO-DOs
#   Profile existing code on perlmutter, generate roofline models
#   Add additional control to argparse for training params
#   Brainstorm different models
#   Allow a json file to specify the net layers
#   Complete discussion of results


class InputLayer:
    def __init__(self, size, batch_size, device):
        self.size = size
        self.current_batch = batch_size
        self.device = device
        self.activations = np.zeros((self.size,), dtype=np.float32)
        if self.device == "gpu":
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
            self.compute_ReLU_deriv = get_compute_ReLU_deriv_kernel()
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
            # Apply the activation function
            self.activations = np.maximum(0, self.preactivations)  # ReLU

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

    def backpropagation(self, reference):
        if self.device == "gpu":
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
            activation_grad = np.where(self.preactivations > 0, 1.0, 0.0)  # ReLU
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
            # transpose next layer's weights
            grid_x = math.ceil(M_next / self.tile_size)
            grid_y = math.ceil(M_curr / self.tile_size)
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
            self.compute_ReLU_deriv(
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
        # First transpose the previous layers activations
        grid_x = math.ceil(M_prev / self.tile_size)
        grid_y = math.ceil(N_batch / self.tile_size)
        self.transpose_matrix(
            self.previous_layer.activations_gpu,
            self.previous_layer.activations_transposed_gpu,
            np.int32(M_prev),
            np.int32(N_batch),
            block=(self.blocksize, self.blocksize, 1),
            grid=(grid_x, grid_y, 1),
        )

        # Multiply the delta and transposed activations to get weights grad
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

        # Sum across delta rows to get the biases grad
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

        new_layer = InputLayer(1, batch_size, device)
        self.layers.append(new_layer)
        for ilayer in range(1, len(sizes)):
            self.layers.append(
                Layer(sizes[ilayer], self.layers[-1], batch_size, device)
            )

    def train(self, nepochs, training_rate):

        # Only pass from CPU to GPU twice per batch
        # current inputs/references to GPU
        # send back loss to cpu for each batch

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

                # Grab entire matrix of refs instead of one at a time
                references = refs_shuffled[istart:iend].reshape(1, current_batch)
                self.layers[0].activations = inputs_shuffled[istart:iend].reshape(
                    1, current_batch
                )

                # send activations to gpu if using that device
                if self.device == "gpu":
                    self.layers[0].activations_gpu = gpuarray.to_gpu(
                        self.layers[0].activations
                    )

                # Feedforward through the other layers
                start_time = time.time()
                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].feedforward()

                feedforward_time += time.time() - start_time

                # Do backpropagation
                start_time = time.time()
                for ilayer in range(len(self.layers) - 1):
                    self.layers[-1 - ilayer].backpropagation(references)
                backpropagation_time += time.time() - start_time

                # Calculate loss (after backprop: output delta = activations - reference)
                if self.device == "gpu":
                    squared_error_gpu = self.layers[-1].delta_gpu ** 2
                    batch_loss_gpu = gpuarray.sum(squared_error_gpu)
                    loss += float(batch_loss_gpu.get())
                else:
                    for ilayer in range(1, len(self.layers)):
                        loss += np.sum(
                            (self.layers[ilayer].activations - references) ** 2
                        )

                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].apply_gradient(current_batch, training_rate)

            standard_deviation = math.sqrt(loss / ninputs)
            print(f"Epoch, deviation: {iepoch}, {standard_deviation}")

        print(f"Feedforward time: {feedforward_time}")
        print(f"Backpropagation time: {backpropagation_time}")

    def save_weights(self, filename):
        model_dict = {}
        # Starting after input layer, save weights/biases
        for i, layer in enumerate(self.layers[1:]):
            if layer.device == "gpu":
                layer.weights = layer.weights_gpu.get()
                layer.biases = layer.biases_gpu.get()
            model_dict[f"layer_{i}_weights"] = layer.weights
            model_dict[f"layer_{i}_biases"] = layer.biases

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
        print(f"Model weights loaded from {filename}")


def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner


def load_setup(filename):

    model_path = Path("setup")
    model_path.mkdir(parents=True, exist_ok=True)
    filepath = f"setup/{filename}"

    # Parse setup file
    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            try:
                run_data = json.load(file)
                print("Run setup json parsed successfully")
            except json.JSONDecodeError:
                print("Error: file found but could not be parsed")
    else:
        raise FileNotFoundError(f"File not found: {filepath}")

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or evaluate neural net")

    # # Name of the model to train or evaluate (REQUIRED)
    # parser.add_argument("model", help="Name of model to train or load")
    # parser.add_argument(
    #     "--epochs", type=int, default=500, help="Number of training epochs"
    # )

    # # Learning rate for training
    # parser.add_argument(
    #     "-lr",
    #     "--learning_rate",
    #     type=float,
    #     default=0.05,
    #     help="Number of training epochs",
    # )

    parser.add_argument("run", help="Name of setup run.json to train or evaluate")

    # Run mode (train or evaluate)
    parser.add_argument(
        "-t", "--train", action="store_true", help="Sets model to train"
    )
    parser.add_argument(
        "-e", "--eval", action="store_true", help="Set model to evaluate"
    )

    # Device mode (CPU or GPU)
    parser.add_argument("--cpu", action="store_true", help="Sets device to CPU")
    parser.add_argument("--gpu", action="store_true", help="Sets device to GPU")

    args = parser.parse_args()

    # Extract values and ensure that requirements are met
    data = load_setup(args.run)
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

    # Create full file name from models directory
    model_path = Path("models")
    model_path.mkdir(parents=True, exist_ok=True)
    filename = f"models/{run}_{device}.npz"

    # Set morse params
    De = 1.0
    re = 1.0
    a = 1.0
    min_rvalue = 0.5
    max_rvalue = 2.0

    # Randomly generate a set of distances
    rvalues = np.random.uniform(min_rvalue, max_rvalue, (ninputs,)).astype(np.float32)

    # Generate the reference energies for each of these distances
    erefs = np.empty((ninputs,), dtype=np.float32)
    for idx, r in enumerate(rvalues):
        erefs[idx] = morse_potential(De, re, a, r)
    mean_e = np.mean(erefs)
    std_e = np.std(erefs)
    erefs_normalized = (erefs - mean_e) / std_e

    rvalues_normalized = (rvalues - (max_rvalue + min_rvalue) / 2.0) / (
        max_rvalue - min_rvalue
    )

    # Train the model
    if args.train:

        print(f"training rate: {training_rate}")
        net = Network(
            architecture, rvalues_normalized, erefs_normalized, batch_size, device
        )
        start_time = time.time()
        net.train(epochs, training_rate)
        print(f"Training time: {time.time() - start_time}")
        net.save_weights(filename)

    # Evaluate model performance and plot
    elif args.eval:

        n_test = 300
        net_batch = max(batch_size, n_test)
        net = Network(
            architecture,
            rvalues_normalized,
            erefs_normalized,
            net_batch,
            device,
        )
        net.load_weights(filename)
        r_test = np.linspace(min_rvalue, max_rvalue, n_test).astype(np.float32)

        r_test_norm = (r_test - (max_rvalue + min_rvalue) / 2.0) / (
            max_rvalue - min_rvalue
        )
        r_test_norm = r_test_norm.reshape(1, n_test)

        # perform forward pass through trained net
        for layer in net.layers:
            layer.current_batch = n_test
        net.layers[0].activations = r_test_norm
        if net.device == "gpu":
            net.layers[0].activations_gpu = gpuarray.to_gpu(net.layers[0].activations)
        for ilayer in range(1, len(net.layers)):
            net.layers[ilayer].feedforward()

        # extract energy and un-normalize
        if net.device == "gpu":
            pred_e_norm = np.asarray(net.layers[-1].activations_gpu.get())[0, :n_test]
        else:
            pred_e_norm = net.layers[-1].activations[0, :n_test]
        pred_energy = (pred_e_norm * std_e) + mean_e

        true_energy = np.array([morse_potential(De, re, a, r) for r in r_test])

        plt.plot(r_test, true_energy, label="true")
        plt.plot(r_test, pred_energy, label="pred")
        plt.legend()

        plot_path = Path("plots")
        plot_path.mkdir(parents=True, exist_ok=True)
        plot_name = plot_path / f"{run}_{device}.png"
        plt.savefig(plot_name)

    # User did not specify a run mode, exit safely
    else:
        sys.exit("Run mode not specified (--train or --eval)")
