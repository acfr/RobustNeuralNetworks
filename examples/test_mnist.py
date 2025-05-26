# Additional requirements to add:
#   - tensorflow: 2.19.0
#   - tensorflow-datasets: 4.9.8

import jax.numpy as jnp                 # JAX NumPy
import matplotlib.pyplot as plt         # Plotting
import optax                            # Optimisation library
import tensorflow_datasets as tfds      # TFDS to download MNIST.
import tensorflow as tf                 # TensorFlow / `tf.data` operations.

from flax import nnx                    # Flax NNX API

from pathlib import Path
from utils.plot_utils import startup_plotting

# Set up plot saving
startup_plotting()
dirpath = Path(__file__).resolve().parent
filepath = dirpath / "results/mnist/"
if not filepath.exists():
    filepath.mkdir(parents=True)

# Set the random seed for reproducibility.
tf.random.set_seed(0)

# Define some learning hyperparameters
# We'll use these to re-size the data
train_steps = 1200       # Do fewer to speed up
eval_every = 200         # Do fewer to speed up
batch_size = 32


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


#### 2. Define Flax model (TODO: Need to change from NNX to linen)
class MLP(nnx.Module):
    """A simple MLP model."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(28*28, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 64, rngs=rngs)
        self.linear3 = nnx.Linear(64, 10, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Instantiate the model
model = MLP(rngs=nnx.Rngs(0))


#### 3. Create the optimiser and define loss metrics

# Hyperparameters
learning_rate = 0.005
momentum = 0.9

# Optimiser (TODO: Need to change from NNX to linen)
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)


#### 4. Define training step function

# Loss function
def loss_fn(model: MLP, batch):
    logits = model(batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
    logits=logits, labels=batch['label']
    ).mean()
    return loss, logits

# Training step
@nnx.jit
def train_step(model: MLP, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])
    optimizer.update(grads)

# Evaluation step
@nnx.jit
def eval_step(model: MLP, metrics: nnx.MultiMetric, batch):
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, logits=logits, labels=batch['label'])


#### 5. Run the training

# from IPython.display import clear_output
metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}

for step, batch in enumerate(train_ds.as_numpy_iterator()):
    
  # Run the optimiser for one step
  train_step(model, optimizer, metrics, batch)

  # Log metrics after one step has passed
  if step > 0 and (step % eval_every == 0 or step == train_steps - 1):
      
    # Training metric
    for metric, value in metrics.compute().items():
      metrics_history[f'train_{metric}'].append(value)
    metrics.reset()

    # Test metric
    for test_batch in test_ds.as_numpy_iterator():
      eval_step(model, metrics, test_batch)
      
    for metric, value in metrics.compute().items():
      metrics_history[f'test_{metric}'].append(value)
    metrics.reset()

# Plot loss and accuracy in subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
# for dataset in ('train', 'test'):
for dataset in ['test']:
    ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
    ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dataset}_accuracy')

ax1.set_xlabel("Training epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.set_ylabel("Accuracy")
ax2.set_xlabel("Training epochs")
ax2.legend()

plt.tight_layout()
plt.savefig(filepath / "train_nnx.pdf")
plt.close()


#### 6. Perform inference

model.eval() # Switch to evaluation mode.

@nnx.jit
def pred_step(model: MLP, batch):
  logits = model(batch['image'])
  return logits.argmax(axis=1)

est_batch = test_ds.as_numpy_iterator().next()
pred = pred_step(model, test_batch)

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
plt.savefig(filepath / "test_nnx.pdf")
