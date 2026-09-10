# MNIST Classifier from Scratch

A three-layer feedforward neural network written in pure NumPy — no autograd, no
deep learning framework. `torchvision` is used only to download and unpack the
MNIST dataset; every weight, activation, and gradient in this project is computed
by hand.

Reaches roughly **90% test accuracy** on MNIST after 5 epochs.

---

## Architecture

```
784 (pixels)  ──►  16  ──►  16  ──►  10 (digit scores)
                sigmoid   sigmoid    linear
                                        │
                                     softmax
```

| Layer | Shape | Activation |
|---|---|---|
| Hidden 1 | 784 → 16 | sigmoid |
| Hidden 2 | 16 → 16 | sigmoid |
| Output | 16 → 10 | none (raw logits) |

The output layer is deliberately left linear. Softmax is applied afterwards to
turn the logits into a probability distribution over the ten digits — squashing
them through a sigmoid first would compress the range and slow learning.

Each weight matrix stores **one neuron per row**, so `Hidden1Weights[j, :]` is the
784-element weight vector belonging to hidden neuron `j`.

---

## The math

**Forward pass.** For each layer, a weighted sum plus a bias, then an activation:

```
z = w · input + b
a = sigmoid(z) = 1 / (1 + e^-z)
```

**Loss.** Softmax converts the ten raw outputs into probabilities, and
categorical cross-entropy scores them against the one-hot label:

```
p_k = e^(z_k) / Σ e^(z_j)
L   = -Σ y_k · log(p_k)
```

Softmax is computed with the max subtracted from every logit, which is
mathematically identical but avoids overflow in `exp`.

**Backward pass.** The softmax/cross-entropy pairing has a convenient property:
the gradient at the output layer collapses to the difference between prediction
and truth.

```
δ_out     = p - y
δ_hidden2 = (OutputWeightsᵀ · δ_out)     * a₂(1 - a₂)
δ_hidden1 = (Hidden2Weightsᵀ · δ_hidden2) * a₁(1 - a₁)
```

Each delta is the next layer's error pulled back through that layer's weights,
scaled by the local sigmoid derivative. A layer's weight gradient is then the
outer product of its delta with the input it received:

```
∇W = outer(δ, layer_input)
∇b = δ
W -= lr · ∇W
b -= lr · ∇b
```

**Initialisation.** Weights are drawn from a normal distribution scaled by
`sqrt(1 / fan_in)`, biases start at zero. This matters more than it looks: with
784 inputs, uniformly positive weights push the pre-activation sums far into the
flat tail of the sigmoid, where the derivative is near zero and nothing learns.

---

## Training

- **Optimiser:** plain stochastic gradient descent
- **Batch size:** 1 — weights update after every single image
- **Learning rate:** 0.1 (fixed, no schedule)
- **Epochs:** 5
- **Preprocessing:** pixels flattened to 784 values and divided by 255

Mean training loss is printed after each epoch. Useful reference point: random
guessing across ten classes gives a loss of `ln(10) ≈ 2.30`, so meaningful
learning means falling well below that, not just drifting downward.

---

## Files

```
mnist_bare_nn.py    network, training loop, evaluation
README.md           this file
data/               MNIST archives (downloaded automatically on first run)
```

All file I/O is anchored to the approved base directory:

```
C:\Users\MikeBiersteker\Documents\Victoria\Claude
```

MNIST downloads into the `data\` subfolder on the first run and is reused
afterwards.

---

## Requirements

```
numpy
torch
torchvision
pandas          # imported but not currently used
matplotlib      # imported but not currently used
```

```bash
pip install numpy torch torchvision pandas matplotlib
```

## Running it

```bash
python mnist_bare_nn.py
```

The first run downloads about 10 MB of MNIST data. Expect several minutes per
epoch — see below.

---

## Known limitations

**It is slow.** Every image is processed with a Python `for` loop over each
neuron in each layer, which is roughly 42 individual `np.dot` calls per image and
60,000 images per epoch. Replacing the loops with a single matrix multiply per
layer, `sigmoid(X @ W.T + b)`, over a minibatch would be dramatically faster and
give smoother gradients.

**Sigmoid activations cap the ceiling.** Sigmoids saturate and their gradients
shrink as they stack, which is part of why accuracy plateaus around 90%. ReLU in
the hidden layers is the usual fix.

**16 neurons is a tight bottleneck.** All ten digits have to be represented
inside a 16-dimensional space. Widening the hidden layers to 64 or 128 typically
pushes a plain feedforward net past 97% on MNIST.

**No regularisation, no validation split, no learning rate schedule.** The test
set is used directly to check accuracy, so it is not a clean held-out estimate if
hyperparameters get tuned against it.

## Possible next steps

- Vectorise the forward and backward passes over minibatches
- Swap sigmoid for ReLU in the hidden layers
- Widen the hidden layers and add a proper train/validation split
- Add momentum or Adam instead of vanilla SGD
- Plot the loss curve with the already-imported `matplotlib`
- Visualise the first-layer weights as 28×28 images to see what the network learned
