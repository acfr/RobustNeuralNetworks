# Additional requirements to add:
#   - tensorflow: 2.19.0
#   - tensorflow-datasets: 4.9.8

import jax
import jax.numpy as jnp                 # JAX NumPy
import matplotlib.pyplot as plt         # Plotting
import numpy as np
import optax                            # Optimisation library
import tensorflow_datasets as tfds      # TFDS to download MNIST.
import tensorflow as tf                 # TensorFlow / `tf.data` operations.

import flax.linen as nn
from robustnn import lbdn

from pathlib import Path
from utils.plot_utils import startup_plotting

# Set up plot saving
startup_plotting()
dirpath = Path(__file__).resolve().parent
filepath = dirpath / "results/mnist/"
if not filepath.exists():
    filepath.mkdir(parents=True)

# Set the random seed for reproducibility.
seed = 42
tf.random.set_seed(seed)

# Define some learning hyperparameters
# We'll use these to re-size the data
train_steps = 2400      # Do fewer to speed up
eval_every = 100        # Do fewer to speed up
batch_size = 64         # Do fewer to speed up
test_batch_size = 256


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

# Shuffle the dataset and group into batches. Skip any incomplete batches
train_ds = train_ds.repeat().shuffle(1024, seed=seed)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(test_batch_size, drop_remainder=True).prefetch(1)


#### 2. Define Flax model

# Data sizes for MNIST
n_inputs = 28 * 28    # Images are 28 x 28 pixels each
n_out = 10        # Numbers are 0 to 9, so 10 options

class MLP(nn.Module):
    """A simple MLP model."""
    
    def setup(self):
      self.linear1 = nn.Dense(64)
      self.linear2 = nn.Dense(64)
      self.linear3 = nn.Dense(n_out)

    def __call__(self, x):
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = self.linear3(x)
        return x
      
class LBDN(nn.Module):
    """A simple LBDN model built with Sandwich layers."""
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
model_mlp = MLP()
model_lbdn = LBDN(gamma=2.0)


#### 3. Define loss metrics

# Define function for tracking loss and accuracy
def get_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
  
def compute_metrics(logits, labels):
    loss = get_loss(logits, labels)
    accuracy = 100 * jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {"loss": loss, "accuracy": accuracy}


#### 4. Define training function

def train_mnist_classifier(model, seed=0):
  
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
      
    # Train over many batches and log test error metrics
    metrics = {"test_loss": [], "test_accuracy": [], "step": []}
    for step, batch in enumerate(train_ds.as_numpy_iterator()):
        
        # Run the optimiser for one step
        params, opt_state = train_step(params, opt_state, batch)
        
        # Log metrics intermittently
        if step == 0 or (step % eval_every == 0 or step == train_steps - 1):
            batch_metrics = {"loss": [], "accuracy": []}
            for test_batch in test_ds.as_numpy_iterator():
                _, test_logits = loss_fn(params, test_batch)
                results = compute_metrics(test_logits, test_batch["label"])
                batch_metrics["loss"].append(results["loss"])
                batch_metrics["accuracy"].append(results["accuracy"])
            metrics["test_loss"].append(np.mean(batch_metrics["loss"]))
            metrics["test_accuracy"].append(np.mean(batch_metrics["accuracy"]))
            metrics["step"].append(step)
            
    return params, metrics


#### 5. Train models
params_mlp, metrics_mlp = train_mnist_classifier(model_mlp, seed)
params_lbdn, metrics_lbdn = train_mnist_classifier(model_lbdn, seed)

# Plot loss and accuracy in subplots
color_mlp = "#009E73"
color_lbdn = "#D55E00"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

ax1.plot(metrics_mlp["step"], metrics_mlp["test_loss"], color=color_mlp, label="MLP")
ax2.plot(metrics_mlp["step"], metrics_mlp["test_accuracy"], color=color_mlp, label="MLP")

ax1.plot(metrics_lbdn["step"], metrics_lbdn["test_loss"], "--", color=color_lbdn, label="Lipschitz")
ax2.plot(metrics_lbdn["step"], metrics_lbdn["test_accuracy"], "--", color=color_lbdn, label="Lipschitz")

ax1.set_xlabel("Training epochs")
ax2.set_xlabel("Training epochs")
ax1.set_ylabel("Test loss")
ax2.set_ylabel("Test accuracy (\%)")

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.savefig(filepath / "train.pdf")
plt.close()


#### 6. Perform inference
def eval_mnist_classifier(model, params, test_batch):
  
    @jax.jit
    def pred_step(params, batch):
      logits = model.apply(params, batch['image'])
      return logits.argmax(axis=1)
    
    return pred_step(params, test_batch)
  
def plot_mnist_results(test_batch, pred, name):

    fig, axs = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axs.flatten()):
        
        # Reshape image again for plotting
        label = test_batch['label'][i]
        image = test_batch['image'][i]
        image = jnp.reshape(image, (28, 28))
        
        # Plot the number
        ax.imshow(image, cmap='gray')
        ax.set_title(f"Label: {label}, Pred: {pred[i]}")
        ax.axis('off')
    plt.savefig(filepath / f"test_{name}.pdf")
    plt.close()


# Run the predictions
test_batch = test_ds.as_numpy_iterator().next()
pred_mlp = eval_mnist_classifier(model_mlp, params_mlp, test_batch)
pred_lbdn = eval_mnist_classifier(model_lbdn, params_lbdn, test_batch)

# Plot the predictions
plot_mnist_results(test_batch, pred_mlp, "mlp")
plot_mnist_results(test_batch, pred_lbdn, "lbdn")


#### 7. Add adversarial attacks with PGD

def pgd_attack(    
    model,
    params,
    test_batch,
    attack_size=1,
    max_iter=500,
    learning_rate=0.01,
    verbose=False,
    seed=0
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
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(attack, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        attack = optax.apply_updates(attack, updates)
        return attack, opt_state, loss

    # Use gradient descent to estimate the Lipschitz bound
    for iter in range(max_iter):
        attack, opt_state, loss = attack_step(attack, opt_state, test_batch)
        if verbose and iter % 20 == 0:
            print("Iter: ", iter, "\t L: ", loss)
    
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
  
attack_sizes = jnp.arange(0, 3.1, 0.1)
acc_mlp = []
acc_lbdn = []
for a in attack_sizes:
    acc_mlp.append(attacked_test_error(model_mlp, params_mlp, test_batch, a))
    acc_lbdn.append(attacked_test_error(model_lbdn, params_lbdn, test_batch, a))

# Plot the results
plt.plot(attack_sizes, acc_mlp, color=color_mlp, label="MLP")
plt.plot(attack_sizes, acc_lbdn, "--", color=color_lbdn, label="Lipschtiz")
plt.xlabel("Attack size (normalised)")
plt.ylabel("Accuracy (\%)")
plt.legend()
plt.tight_layout()
plt.savefig(filepath / "attacks.pdf")

# Examples when MLP is at about 20% accuracy
attack_size = 1.0
_, attack_batch_mlp = pgd_attack(model_mlp, params_mlp, test_batch, attack_size)
pred_mlp = eval_mnist_classifier(model_mlp, params_mlp, attack_batch_mlp)
plot_mnist_results(attack_batch_mlp, pred_mlp, "mlp_attacked_10")

_, attack_batch_lbdn = pgd_attack(model_lbdn, params_lbdn, test_batch, attack_size)
pred_lbdn = eval_mnist_classifier(model_lbdn, params_lbdn, attack_batch_lbdn)
plot_mnist_results(attack_batch_lbdn, pred_lbdn, "lipschitz_attacked_10")

# Examples when Lipschitz is at about 20 % accuracy
attack_size = 2.1
_, attack_batch_mlp = pgd_attack(model_mlp, params_mlp, test_batch, attack_size)
pred_mlp = eval_mnist_classifier(model_mlp, params_mlp, attack_batch_mlp)
plot_mnist_results(attack_batch_mlp, pred_mlp, "mlp_attacked_21")

_, attack_batch_lbdn = pgd_attack(model_lbdn, params_lbdn, test_batch, attack_size)
pred_lbdn = eval_mnist_classifier(model_lbdn, params_lbdn, attack_batch_lbdn)
plot_mnist_results(attack_batch_lbdn, pred_lbdn, "lipschitz_attacked_21")
