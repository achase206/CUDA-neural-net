import numpy as np
import time
import math
import matplotlib.pyplot as plt

# GAME PLAN!!!!!
# gpuarray.to_gpu(some data)
# need to do this for all the initilized matrices
# Figure out how to do this with streams?
#   Conceptually where is this appropriate?
#   Which functions touch this data in matrices? I assume basically everything...
#   Look into whether I need to rewrite everything in cuda or can I reuse some of what is here?
#   Probably makes sense to keep this functionality as CPU mode vs GPU mode for debugging
# Lets also clean up how we call these with argparse or something similar this is too crude rn
# Focus on just getting this running asap and profiling then we can work on extending
#   Make sure implementation remains very modular...
# I don't think I can use numpy here for matrix multiplication, need to grab from gemm code?


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.activations = np.zeros((self.size,), dtype=np.float32)


class Layer:
    def __init__(self, size, previous_layer):
        self.size = size
        self.previous_layer = previous_layer
        self.next_layer = None
        self.previous_layer.next_layer = self

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
        # self.biases_grad = np.zeros((self.size,), dtype=np.float32)
        self.biases_grad = np.zeros((self.size, 1), dtype=np.float32)
        self.delta = np.zeros((self.size,), dtype=np.float32)

    def feedforward(self):
        self.preactivations = (
            np.dot(self.weights, self.previous_layer.activations) + self.biases
        )

        if self.next_layer is None:  # Just do linear output
            self.activations = self.preactivations
        else:
            # Apply the activation function
            self.activations = np.maximum(0, self.preactivations)  # ReLU

    def backpropagation(self, reference):
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
        # self.weights_grad += np.outer(self.delta, self.previous_layer.activations)
        self.weights_grad += np.dot(self.delta, self.previous_layer.activations.T)

    def apply_gradient(self, batch_size, training_rate):
        self.weights = self.weights - (training_rate / batch_size) * self.weights_grad
        self.biases = self.biases - (training_rate / batch_size) * self.biases_grad
        self.weights_grad.fill(0.0)
        self.biases_grad.fill(0.0)


class Network:
    def __init__(self, sizes, training_inputs, training_references):
        self.layers = []
        self.training_inputs = training_inputs
        self.training_references = training_references

        new_layer = InputLayer(1)
        self.layers.append(new_layer)
        for ilayer in range(1, len(sizes)):
            self.layers.append(Layer(sizes[ilayer], self.layers[-1]))

    def train(self, nepochs):
        feedforward_time = 0.0
        backpropagation_time = 0.0
        for iepoch in range(nepochs):
            loss = 0.0
            indices = np.random.permutation(ninputs)
            inputs_shuffled = self.training_inputs[indices]
            refs_shuffled = self.training_references[indices]

            # batch_size = ninputs
            batch_size = 32
            for istart in range(0, ninputs, batch_size):
                iend = min(istart + batch_size, ninputs)

                current_batch = iend - istart

                # Now grab matrix of refs instead of one at a time
                references = refs_shuffled[istart:iend].reshape(1, current_batch)
                self.layers[0].activations = inputs_shuffled[istart:iend].reshape(
                    1, current_batch
                )

                # Feedforward through the other layers
                start_time = time.time()
                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].feedforward()

                # Calculate loss across all layers
                for ilayer in range(1, len(self.layers)):
                    loss += np.sum((self.layers[ilayer].activations - references) ** 2)

                feedforward_time += time.time() - start_time

                # Do backpropagation
                start_time = time.time()
                for ilayer in range(len(self.layers) - 1):
                    self.layers[-1 - ilayer].backpropagation(references)
                backpropagation_time += time.time() - start_time

                # for iref in range(istart, iend):
                #     reference = np.array( refs_shuffled[iref], dtype=np.float32 )

                #     # Set the input layer
                #     self.layers[0].activations[0] = inputs_shuffled[iref]

                #     # Feedforward through the other layers
                #     start_time = time.time()
                #     for ilayer in range( 1, len(self.layers) ):
                #         self.layers[ilayer].feedforward()
                #     loss += np.sum( (self.layers[-1].activations - reference)**2 )
                #     feedforward_time += time.time() - start_time

                #     # Do backpropagation
                #     start_time = time.time()
                #     for ilayer in range( len(self.layers)-1 ):
                #         self.layers[-1-ilayer].backpropagation(reference)
                #     backpropagation_time += time.time() - start_time

                for ilayer in range(1, len(self.layers)):
                    self.layers[ilayer].apply_gradient(batch_size, training_rate)

            standard_deviation = math.sqrt(loss / ninputs)
            print(f"Epoch, deviation: {iepoch}, {standard_deviation}")

        print(f"Feedforward time: {feedforward_time}")
        print(f"Backpropagation time: {backpropagation_time}")


def morse_potential(De, re, a, r):
    inner = 1.0 - math.exp(-a * (r - re))
    return De * inner * inner


if __name__ == "__main__":
    ninputs = 300
    De = 1.0
    re = 1.0
    a = 1.0
    min_rvalue = 0.5
    max_rvalue = 2.0
    training_rate = 0.05

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

    net = Network([1, 16, 16, 1], rvalues_normalized, erefs_normalized)
    start_time = time.time()
    net.train(500)
    print(f"Training time: {time.time() - start_time}")

    n_test = 300
    r_test = np.linspace(min_rvalue, max_rvalue, n_test).astype(np.float32)

    r_test_norm = (r_test - (max_rvalue + min_rvalue) / 2.0) / (max_rvalue - min_rvalue)
    r_test_norm = r_test_norm.reshape(1, n_test)

    # perform forward pass through trained net
    net.layers[0].activations = r_test_norm
    for ilayer in range(1, len(net.layers)):
        net.layers[ilayer].feedforward()

    # extract energy and un-normalize
    pred_e_norm = net.layers[-1].activations[0]
    pred_energy = (pred_e_norm * std_e) + mean_e

    true_energy = np.array([morse_potential(De, re, a, r) for r in r_test])

    plt.plot(r_test, true_energy, label="true")
    plt.plot(r_test, pred_energy, label="pred")
    plt.legend()
    plt.savefig("morse.png")
