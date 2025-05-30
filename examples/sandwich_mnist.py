import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp                 # JAX NumPy
import matplotlib.pyplot as plt         # Plotting
import numpy as np
import optax                            # Optimisation library
import tensorflow_datasets as tfds      # TFDS to download MNIST.
import tensorflow as tf                 # TensorFlow / `tf.data` operations.

import flax.linen as nn
from robustnn import lbdn

from functools import partial
from pathlib import Path
from utils.plot_utils import startup_plotting

# Set up plot saving
startup_plotting()
dirpath = Path(__file__).resolve().parent
filepath = dirpath / "../results/mnist/"
if not filepath.exists():
    filepath.mkdir(parents=True)

# Set the random seed for reproducibility.
seed = 42
tf.random.set_seed(seed)


#### 1. Data loading

# Load the dataset
train_ds: tf.data.Dataset = tfds.load('mnist', split='train', data_dir="data/")
test_ds: tf.data.Dataset = tfds.load('mnist', split='test', data_dir="data/")

# Data pre-processing:
#   1. Flatten the images (we're just using MLPs here)
#   2. Normalise the data
def flatten_and_normalise(sample):
    image = sample["image"]
    label = sample["label"]
    image = tf.cast(image, tf.float32) / 255
    image = tf.reshape(image, [-1])
    return {"image": image, "label": label}
  
train_ds = train_ds.map(flatten_and_normalise)
test_ds = test_ds.map(flatten_and_normalise)

# Data sizes for MNIST
n_inputs = 28 * 28      # Images are 28 x 28 pixels each
n_out = 10              # Numbers are 0 to 9, so 10 options

# Hyperparameters/data sizes
train_steps = 2000      # Number of training steps to take
eval_every = 100        # How often to evaluate during training
batch_size = 128        # Training batch size
test_batch_size = 256   # Test batch size

# Shuffle the dataset and group into batches. Skip any incomplete batches
train_ds = train_ds.repeat().shuffle(1024, seed=seed)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(test_batch_size, drop_remainder=True).prefetch(1)


#### 2. Define Flax model

class UnconstrainedMLP(nn.Module):
    """A simple MLP model."""

    def setup(self):
      self.linear1 = nn.Dense(64)       # Layer 1 has 64 hidden neurons
      self.linear2 = nn.Dense(64)       # Layer 2 has 64 hidden neurons
      self.linear3 = nn.Dense(n_out)    # Layer 3 has n_out=10 outputs (one for each number)

    def __call__(self, x):
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = self.linear3(x)
        return x
      
class LipschitzMLP(nn.Module):
    """A simple Lipschitz-bounded MLP built with Sandwich layers."""
    gamma: jnp.float32 = 1.0 # type: ignore

    def setup(self):
      self.sandwich1 = lbdn.SandwichLayer(n_inputs, 64, activation=nn.relu)
      self.sandwich2 = lbdn.SandwichLayer(64, 64, activation=nn.relu)
      self.sandwich3 = lbdn.SandwichLayer(64, n_out, is_output=True)
      self.scale = jnp.sqrt(self.gamma)

    def __call__(self, x):
        x = self.scale * x
        x = self.sandwich1(x)
        x = self.sandwich2(x)
        x = self.sandwich3(x)
        x = self.scale * x
        return x

# Instantiate the models
model_mlp = UnconstrainedMLP()
model_lip = LipschitzMLP(gamma=2.0)


#### 3. Define loss metrics and utils

# Loss function: standard cross-entropy loss for image classification
def get_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()

# Compute classification accuracy
def compute_accuracy(logits, labels):
    return 100 * jnp.mean(jnp.argmax(logits, axis=-1) == labels)

# Helper function to make predictions on a batch of data given a model and its learnable parameters
@partial(jax.jit, static_argnums=0)
def predict(model, params, batch):
    logits = model.apply(params, batch['image'])
    return logits.argmax(axis=1)


#### 4. Define training function

def train_mnist_classifier(model, seed=42, verbose=True):

    # Initialise the model parameters
    rng = jax.random.key(seed)
    inputs = jnp.ones((1, n_inputs), jnp.float32)
    params = model.init(rng, inputs)

    # Set up the optimiser
    optimizer = optax.adam(learning_rate=0.005)
    opt_state = optimizer.init(params)

    # Loss function
    @jax.jit
    def loss_fn(params, batch):
        logits = model.apply(params, batch['image'])
        loss = get_loss(logits, batch['label'])
        return loss, logits

    # A single training step
    @jax.jit
    def train_step(params, opt_state, batch):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, _ = grad_fn(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    # Train over many batches and log test accuracy
    metrics = {"step": [], "test_accuracy": []}
    for step, batch in enumerate(train_ds.as_numpy_iterator()):

        # Run the optimiser for one step
        params, opt_state = train_step(params, opt_state, batch)

        # Log metrics intermittently
        if step == 0 or (step % eval_every == 0 or step == train_steps - 1):
            batch_accuracy = []
            for test_batch in test_ds.as_numpy_iterator():
                _, test_logits = loss_fn(params, test_batch)
                acc = compute_accuracy(test_logits, test_batch["label"])
                batch_accuracy.append(acc)
            metrics["step"].append(step)
            metrics["test_accuracy"].append(np.mean(batch_accuracy))

            # Print the results to inform the user
            if verbose and (step % (5*eval_every) == 0 or step == train_steps - 1):
                print(f"Training step: {step}\tTest accuracy (%): {metrics['test_accuracy'][-1]:.2f}")

    return params, metrics


#### 5. Train models
params_mlp, metrics_mlp = train_mnist_classifier(model_mlp, seed, verbose=True)
params_lip, metrics_lip = train_mnist_classifier(model_lip, seed, verbose=True)

# Plot loss and accuracy in subplots
color_mlp = "#009E73"
color_lip = "#D55E00"

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(metrics_mlp["step"], metrics_mlp["test_accuracy"], color=color_mlp, label="Unconstrained")
ax.plot(metrics_lip["step"], metrics_lip["test_accuracy"], "--", color=color_lip, label="Lipschitz")

ax.set_xlabel("Training epochs")
ax.set_ylabel("Test accuracy (\%)")
ax.set_xlim(0, train_steps)
ax.legend(loc="lower right")
plt.tight_layout()

plt.savefig(filepath / "train.pdf")
plt.close()


#### 6. Perform inference

def plot_mnist_results(test_batch, pred, name):

    fig, axs = plt.subplots(1, 3, figsize=(9, 5))
    for i, ax in enumerate(axs.flatten()):

        # Reshape image again for plotting
        i = i + 1   # Choose nice examples
        label = test_batch['label'][i]
        image = test_batch['image'][i]
        image = jnp.reshape(image, (28, 28))

        # Plot the number
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {label}, Pred: {pred[i]}")
        ax.axis('off')
        fig.suptitle(name)
    plt.savefig(filepath / f"test_{name}.pdf")
    plt.close()

# Run the predictions
test_batch = test_ds.as_numpy_iterator().next()
pred_mlp = predict(model_mlp, params_mlp, test_batch)
pred_lip = predict(model_lip, params_lip, test_batch)

# Plot the predictions
plot_mnist_results(test_batch, pred_mlp, "Unconstrained MLP")
plot_mnist_results(test_batch, pred_lip, "Lipschitz-bounded MLP")


#### 7. Add adversarial attacks with PGD

# Compute l2-optimal adversarial attacks with projected gradient descent.
def pgd_attack(
    model,
    params,
    test_batch,
    attack_size=1,
    max_iter=500,
    learning_rate=0.01,
    seed=42
):

    # Edge case
    if attack_size == 0:
        return jnp.zeros(test_batch["image"].shape), test_batch

    # Define how to constrain attack size (l2 norm)
    def project_attack(attack, attack_size):
        attack = attack / jnp.linalg.norm(attack, axis=-1, keepdims=True)
        return attack_size * attack

    # Initialise an attack
    rng = jax.random.key(seed)
    rng, key1 = jax.random.split(rng)
    attack = jax.random.uniform(key1, test_batch["image"].shape)
    attack = (project_attack(attack, attack_size),)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(attack)

    # Loss function
    @jax.jit
    def loss_fn(attack, batch):
        attack = project_attack(attack[0], attack_size)
        attacked_image = batch['image'] + attack
        logits = model.apply(params, attacked_image)
        return -get_loss(logits, batch['label'])

    # A single attack step with projected gradient descent
    @jax.jit
    def attack_step(attack, opt_state, batch):
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(attack, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        attack = optax.apply_updates(attack, updates)
        return attack, opt_state

    # Use gradient descent to estimate the Lipschitz bound
    for _ in range(max_iter):
        attack, opt_state = attack_step(attack, opt_state, test_batch)

    # Return the attack and the perturbed image
    attack = project_attack(attack[0], attack_size)
    attack_batch = {"image": test_batch["image"] + attack,
                      "label": test_batch["label"]}
    return attack, attack_batch
  
  
# Compute accuracy as a function of attack size
def attacked_test_error(model, params, test_batch, attack_size):
    _, attack_batch = pgd_attack(model, params, test_batch, attack_size)
    logits = model.apply(params, attack_batch['image'])
    labels = test_batch["label"]
    return 100 * jnp.mean(jnp.argmax(logits, axis=-1) == labels)

# Run the attacks
attack_resolution = 0.2
attack_sizes = jnp.arange(0, 3 + attack_resolution, attack_resolution)
acc_mlp, acc_lip = [], []
print("Attack size:", end=" ")
for a in attack_sizes:
    print(f"{a:.1f}", end=", ")
    acc_mlp.append(attacked_test_error(model_mlp, params_mlp, test_batch, a))
    acc_lip.append(attacked_test_error(model_lip, params_lip, test_batch, a))
print()


# Plot the results
plt.plot(attack_sizes, acc_mlp, color=color_mlp, label="Unconstrained")
plt.plot(attack_sizes, acc_lip, "--", color=color_lip, label="Lipschtiz")
plt.xlabel("Attack size (normalised)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig(filepath / "attacks.pdf")

# Examples when MLP is at about 20% accuracy
attack_size = 0.9
_, attack_batch_mlp = pgd_attack(model_mlp, params_mlp, test_batch, attack_size)
pred_mlp = predict(model_mlp, params_mlp, attack_batch_mlp)
plot_mnist_results(attack_batch_mlp, pred_mlp, f"Unconstrained MLP with attack size {attack_size}")

# Examples when Lipschitz is at about 20 % accuracy
attack_size = 2.0
_, attack_batch_lip = pgd_attack(model_lip, params_lip, test_batch, attack_size)
pred_lip = predict(model_lip, params_lip, attack_batch_lip)
plot_mnist_results(attack_batch_lip, pred_lip, f"Lipschitz-bounded MLP with attack size {attack_size}")
