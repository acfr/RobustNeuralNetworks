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
train_steps = 1200      # Do fewer to speed up
eval_every = 100        # Do fewer to speed up
batch_size = 64         # Do fewer to speed up


#### 1. Data loading

# Load the dataset
train_ds: tf.data.Dataset = tfds.load('mnist', split='train', data_dir="data/")
test_ds: tf.data.Dataset = tfds.load('mnist', split='test', data_dir="data/")

# Data pre-processing:
#   1. Flatten the images (we're just using MLPs here)
#   2. Normalise the data (TODO should we bother?)
def flatten_and_normalise(sample):
    image = sample["image"]
    label = sample["label"]
    image = tf.cast(image, tf.float32) / 255
    image = tf.reshape(image, [-1])
    return {"image": image, "label": label}
  
train_ds = train_ds.map(flatten_and_normalise)
test_ds = test_ds.map(flatten_and_normalise)

# Shuffle the dataset and group into batches. Skip any incomplete batches
train_ds = train_ds.repeat().shuffle(1024)
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)


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
# model = MLP()
model = LBDN(gamma=10.0)

rng = jax.random.key(seed)
inputs = jnp.ones((1, n_inputs), jnp.float32)
params = model.init(rng, inputs)


#### 3. Create the optimiser and define loss metrics

# Hyperparameters
learning_rate = 0.005
momentum = 0.9

# Optimiser
optimizer = optax.adamw(learning_rate, momentum)
opt_state = optimizer.init(params)

# Define function for tracking loss and accuracy
def get_loss(logits, labels):
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return loss.mean()
  
def compute_metrics(logits, labels):
    loss = get_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {"loss": loss, "accuracy": accuracy}


#### 4. Define training step function

# Loss function
@jax.jit
def loss_fn(params, batch):
    logits = model.apply(params, batch['image'])
    loss = get_loss(logits, batch['label'])
    return loss, logits

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    grad_fn = jax.grad(loss_fn, has_aux=True)
    grads, _ = grad_fn(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state


#### 5. Run the training

# # from IPython.display import clear_output
metrics = {
  "test_loss": [],
  "test_accuracy": [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    
    # Run the optimiser for one step
    params, opt_state = train_step(params, opt_state, batch)
    
    # Start logging metrics after a step has passed
    if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
        
        # Do test evaluation on all the test batches
        batch_metrics = {"loss": [], "accuracy": []}
        for t_step, test_batch in enumerate(test_ds.as_numpy_iterator()):
            _, test_logits = loss_fn(params, test_batch)
            results = compute_metrics(test_logits, test_batch["label"])
            batch_metrics["loss"].append(results["loss"])
            batch_metrics["accuracy"].append(results["accuracy"])
        metrics["test_loss"].append(np.mean(batch_metrics["loss"]))
        metrics["test_accuracy"].append(np.mean(batch_metrics["accuracy"]))

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
ax1.plot(metrics[f'test_loss'])
ax2.plot(metrics[f'test_accuracy'])

ax1.set_xlabel("Training epochs")
ax1.set_ylabel("Test loss")

ax2.set_ylabel("Test accuracy")
ax2.set_xlabel("Training epochs")

plt.tight_layout()
plt.savefig(filepath / "train.pdf")
plt.close()


#### 6. Perform inference

@jax.jit
def pred_step(params, batch):
  logits = model.apply(params, batch['image'])
  return logits.argmax(axis=1)

est_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(params, test_batch)

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
plt.savefig(filepath / "test.pdf")


#### 7. TODO: ADVERSARIAL ATTACKS