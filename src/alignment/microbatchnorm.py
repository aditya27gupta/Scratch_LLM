"""
How normalizing activations within each mini-batch stabilizes training and enables deeper networks.
"""

# === TRADEOFFS ===
# + Enables higher learning rates by stabilizing activation distributions
# + Acts as implicit regularization (batch noise prevents overfitting)
# + Reduces sensitivity to weight initialization
# - Behavior differs between training and inference (running stats vs. batch stats)
# - Breaks down with small batch sizes (noisy statistics)
# - Introduces cross-sample dependency within a batch (problematic for some tasks)
# WHEN TO USE: Deep CNNs and MLPs where training instability or vanishing
#   gradients are limiting depth. Standard for computer vision architectures.
# WHEN NOT TO: Sequence models (use LayerNorm), batch size < 8, or online
#   learning where single-sample updates are required.

import time

import numpy as np

from utility import logger

data_rng = np.random.default_rng(42)

# === SYNTHETIC DATA GENERATION ===
# Concentric rings: each class is a circle at radius r = class_idx + 1, with Gaussian noise added to the radius. This creates a non-linearly-separable problem that requires multiple layers to solve - a single hyperplane can't separate nested rings.


def generate_rings(n_classes: int, n_per_class: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D concentric ring dataset.

    Each ring has radius proportional to its class index. Points are placed
    uniformly around the circle with Gaussian radial noise. The result is
    n_classes nested annuli that require nonlinear decision boundaries.
    """
    ys: np.ndarray = np.random.randint(0, n_classes, size=n_classes * n_per_class)  # pre-generate labels
    radius = 1.0 + ys * 0.8 + np.random.normal(0, noise, size=ys.shape)  # pre-generate noisy radii
    angle = np.random.uniform(0, 2 * np.pi, size=ys.shape)  # pre-generate angles
    xs = np.stack((radius * np.cos(angle), radius * np.sin(angle)), axis=1)
    return xs, ys


# === SCALAR AUTOGRAD ENGINE ===
class BatchNormLayer:
    """Batch normalization for a single feature dimension across a mini-batch.

    Forward (training):
        mu_B = (1/m) * sum(x_i)
        var_B = (1/m) * sum((x_i - mu_B)^2)
        x_hat_i = (x_i - mu_B) / sqrt(var_B + epsilon)
        y_i = gamma * x_hat_i + beta

    Forward (eval):
        Uses running_mean and running_var instead of batch statistics.

    Running stats update:
        running_mean = (1 - momentum) * running_mean + momentum * mu_B
        running_var  = (1 - momentum) * running_var  + momentum * var_B
    """

    def __init__(self, n_features: int, bn_momentum: float = 0.1, bn_epsilon: float = 1e-5) -> None:
        # Learnable parameters: gamma (scale) and beta (shift), initialized to gamma=1, beta=0 so BN starts as an identity-like
        self.gamma: np.ndarray = np.ones(n_features)
        self.beta: np.ndarray = np.zeros(n_features)
        self.running_mean: np.ndarray = np.zeros(n_features)
        self.running_var: np.ndarray = np.ones(n_features)
        self.n_features = n_features
        self.bn_momentum = bn_momentum  # exponential moving average decay for running stats
        self.bn_epsilon = bn_epsilon  # numerical stability in variance normalization
        self.dgamma: float = 0.0
        self.dbeta: float = 0.0
        self._x_hat: np.ndarray | None = None
        self._inv_std: np.ndarray | None = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply batch normalization across a mini-batch.

        Args:
            batch: list of m samples, each a list of n_features
            training: if True, use batch statistics; if False, use running stats

        Returns:
            Normalized batch with same shape as input
        """
        # Normalize each feature independently across the batch. This per-feature normalization is what makes BN different from normalizing the entire activation vector (that's LayerNorm).

        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0, ddof=0)
            self.running_mean = (1 - self.bn_momentum) * self.running_mean + self.bn_momentum * batch_mean
            self.running_var = (1 - self.bn_momentum) * self.running_var + self.bn_momentum * batch_var
            inv_std = 1.0 / np.sqrt(batch_var + self.bn_epsilon)
            x_hat = (x - batch_mean) * inv_std
        else:
            inv_std = 1.0 / np.sqrt(self.running_var + self.bn_epsilon)
            x_hat = (x - self.running_mean) * inv_std
        self._x_hat = x_hat
        self._inv_std = inv_std
        return self.gamma * x_hat + self.beta

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """BN backward pass

        Efficient form that avoids recomputing mean/var
         dx = (inv_std / m) * (m * dx_hat - sum(dx_hat) - x_hat * sum(dx_hat * x_hat))
        """
        m = dy.shape[0]
        self.dgamma = np.sum(dy * self._x_hat, axis=0)
        self.dbeta = np.sum(dy, axis=0)
        dx_hat = dy * self.gamma
        dx = (self._inv_std / m) * (
            m * dx_hat - np.sum(dx_hat, axis=0) - self._x_hat * np.sum(dx_hat * self._x_hat, axis=0)
        )
        return dx

    def update(self, lr: float) -> None:
        self.gamma -= lr * self.dgamma
        self.beta -= lr * self.dbeta

    def param_count(self) -> int:
        return 2 * self.n_features  # gamma and beta


# === MODEL DEFINITION ===
# Architecture: Input(2) → [Linear → BN → ReLU] × 5 → Linear → Output(N_CLASSES)

# Placement: BN goes between Linear and ReLU (Linear → BN → ReLU). This is the original paper's recommendation.
# The linear layer's output is pre-activation — BN normalizes it before the nonlinearity. This matters because ReLU kills negative values: if the pre-activation distribution drifts negative, most neurons die. BN centers activations around zero, ensuring roughly half survive ReLU.


class Linear:
    """Fully connected layer: y = x * W.T + b"""

    def __init__(self, n_in: int, n_out: int) -> None:
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        self.W = data_rng.normal(0, std_dev, (n_out, n_in))
        self.b = np.zeros(n_out)
        self._x: np.ndarray  # cache for backward
        self.dW: np.ndarray
        self.db: np.ndarray

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x  # cache for backward
        return x @ self.W.T + self.b

    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dW = dy.T @ self._x
        self.db = np.sum(dy, axis=0)
        return dy @ self.W

    def update(self, lr: float) -> None:
        self.W -= lr * self.dW
        self.b -= lr * self.db

    def param_count(self) -> int:
        return self.W.size * self.b.size


class Relu:
    def __init__(self) -> None:
        self._mask: np.ndarray  # cache for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0  # cache for backward
        return x * self._mask

    def backward(self, dy: np.ndarray) -> np.ndarray:
        return dy * self._mask


class MLP:
    """5-layer MLP with optional batch normalization."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int, use_bn: bool) -> None:
        self.use_bn = use_bn
        self.layers: list = []
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]

        for i in range(len(dims) - 1):
            self.layers.append(Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # hidden layers only
                if self.use_bn:
                    self.layers.append(BatchNormLayer(dims[i + 1]))
                self.layers.append(Relu())

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for a batch of inputs. training controls BN behavior."""
        for layer in self.layers:
            x = layer.forward(x, training) if isinstance(layer, BatchNormLayer) else layer.forward(x)
        return x

    def backward(self, dy: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dy = layer.backward(dy)

    def update(self, lr: float) -> None:
        for layer in self.layers:
            if isinstance(layer, (Linear, BatchNormLayer)):
                layer.update(lr)

    def param_count(self) -> int:
        return sum(layer.param_count() for layer in self.layers if isinstance(layer, (Linear, BatchNormLayer)))


# === LOSS AND ACCURACY ===


def softmax_cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    """Numerically stable softmax cross-entropy.
    Subtract max before exp to prevent overflow.
    Average cross-entropy loss over the batch.

    Loss = -(1/m) * sum_i log(softmax(logits_i)[target_i])
    """
    m = logits.shape[0]
    shifted = logits - np.max(logits, axis=1, keepdims=True)  # for numerical stability
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    loss = -np.log(probs[np.arange(m), targets] + 1e-10).mean()
    dlogits = probs.copy()
    dlogits[np.arange(m), targets] -= 1  # gradient of loss w
    dlogits /= m  # average over batch
    return loss, dlogits


def accuracy(logits_batch: np.ndarray, targets: np.ndarray) -> float:
    """Classification accuracy: fraction of correct predictions."""
    predictions = np.argmax(logits_batch, axis=1)
    return np.mean(predictions == targets)


# === TRAINING ===
def train_model(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int,
    batch_size: int,
    lr: float,
    label: str,
) -> list[tuple[float, float]]:
    """Train a model with SGD + LR decay, return per-epoch (loss, accuracy) history."""
    history: list[tuple[float, float]] = []
    n = x_train.shape[0]

    for epoch in range(num_epochs):
        idx = data_rng.permutation(n)  # shuffle indices for each epoch
        epoch_loss = 0.0
        epoch_correct = 0

        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            m = batch_idx.shape[0]

            # Skip batches that are too small for meaningful BN statistics.
            if m < 4:
                continue

            # Convert raw floats to Value nodes for autograd tracking
            x_b = x_train[batch_idx]
            y_b = y_train[batch_idx]

            # Forward pass (training mode — uses batch statistics for BN)
            logits = model.forward(x_b, training=True)
            loss, dlogits = softmax_cross_entropy_loss(logits, y_b)

            # Backward pass
            model.backward(dlogits)

            # SGD update with linear learning rate decay
            model.update(lr * (1.0 - epoch / num_epochs))

            # Track statistics
            epoch_loss += loss * m  # sum loss over batch for averaging later
            epoch_correct += np.sum(np.argmax(logits, axis=1) == y_b)

        avg_loss = epoch_loss / n
        avg_acc = epoch_correct / n
        history.append((avg_loss, avg_acc))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"{label} epoch {epoch + 1:>3}/{num_epochs} | loss: {avg_loss:.4f} | acc:{avg_acc:.3f}")

    return history


# === EVAL MODE DEMONSTRATION ===
def eval_model(model: MLP, x_data: np.ndarray, y_data: np.ndarray) -> float:
    """Evaluate model accuracy in inference mode (running stats for BN)."""
    logits = model.forward(x_data, training=False)  # eval mode uses running stats for BN
    return accuracy(logits, y_data)


# === LAYER NORMALIZATION (COMPARISON) ===

# BN normalizes across the batch for each feature:  stats over samples, per feature
# LN normalizes across features for each sample:    stats over features, per sample
# Transformers use LN because token positions have different semantic roles, making batch statistics across positions meaningless. CNNs use BN because spatial features (e.g. edge detectors) have consistent meaning across samples.


def layer_norm_forward(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, bn_epsilon: float = 1e-5) -> np.ndarray:
    """LayerNorm: x_hat_j = (x_j - mu) / sqrt(var + eps), y_j = gamma_j * x_hat_j + beta_j"""
    mean = np.mean(x)
    var = np.var(x)
    x_hat = (x - mean) / np.sqrt(var + bn_epsilon)
    return gamma * x_hat + beta


# === MAIN ===

if __name__ == "__main__":
    # Model parameters
    n_classes = 5
    n_samples_per_class = 600
    noise_std = 0.15
    n_hidden_layers = 20  # depth where vanishing/exploding gradients become real
    hidden_dim = 16  # neurons per hidden layer
    num_epochs = 15  # training epochs
    batch_size = 64  # mini-batch size (BN needs ≥8 for stable statistics)
    learning_rate = 0.05  # SGD learning rate (BN allows higher LR — that's the point)

    # --- Generate dataset ---
    logger.info(f"\nGenerating dataset: {n_classes} classes, {n_samples_per_class} samples/class")
    x_all, y_all = generate_rings(n_classes, n_samples_per_class, noise_std)
    n_total = x_all.shape[0]

    # Train/test split (80/20)
    split = int(0.8 * n_total)
    perm = data_rng.permutation(n_total)
    train_idx, test_idx = perm[:split], perm[split:]
    x_train, x_test = x_all[train_idx], x_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    logger.info(f"Train: {x_train.shape[0]}, Test: {x_test.shape[0]}")

    # --- Train WITHOUT batch normalization ---
    logger.info(f"\n{'─' * 65}")
    logger.info("Training 5-layer MLP WITHOUT batch normalization")
    logger.info(f"{'─' * 65}")
    model_no_bn = MLP(2, hidden_dim, n_classes, n_hidden_layers, use_bn=False)
    t0 = time.time()
    history_no_bn = train_model(model_no_bn, x_train, y_train, num_epochs, batch_size, learning_rate, "no BN")
    time_no_bn = time.time() - t0

    # --- Train WITH batch normalization ---
    logger.info(f"\n{'─' * 65}")
    logger.info("Training 5-layer MLP WITH batch normalization")
    logger.info(f"{'─' * 65}")
    model_bn = MLP(2, hidden_dim, n_classes, n_hidden_layers, use_bn=True)
    t0 = time.time()
    history_bn = train_model(model_bn, x_train, y_train, num_epochs, batch_size, learning_rate, "BN")
    time_bn = time.time() - t0

    # --- Evaluate on test set ---
    logger.info(f"\n{'─' * 65}")
    logger.info("Evaluation (test set, inference mode)")
    logger.info(f"{'─' * 65}")
    test_acc_no_bn = eval_model(model_no_bn, x_test, y_test)
    test_acc_bn = eval_model(model_bn, x_test, y_test)

    # --- Layer Normalization comparison ---
    # Brief demonstration: apply LayerNorm to a single hidden layer's output
    # to show the API difference and when you'd prefer LN over BN.
    logger.info(f"\n{'─' * 65}")
    logger.info("Layer Normalization (single-sample comparison)")
    logger.info(f"{'─' * 65}")
    ln_gamma = np.ones(hidden_dim)  # scale parameter (learnable, but start as identity)
    ln_beta = np.zeros(hidden_dim)  # shift parameter (learnable, but start
    sample_input = np.random.normal(0, 2.0, hidden_dim)
    ln_output = layer_norm_forward(sample_input, ln_gamma, ln_beta)

    # Verify normalization: output should have ~zero mean and ~unit variance
    ln_mean = np.mean(ln_output)
    ln_var = np.var(ln_output)
    logger.info(f"Input mean: {np.mean(sample_input):.4f}, var: {np.var(sample_input):.4f}")
    logger.info(f"Output mean: {ln_mean:+.4f}, var: {ln_var:.4f}")

    # === RESULTS AND COMPARISON TABLE ===
    logger.info(f"{'=' * 65}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'=' * 65}")
    logger.info(f"Architecture: {n_hidden_layers}-layer MLP, hidden_dim={hidden_dim}, {n_classes}-class classification")
    logger.info(f"Dataset: concentric rings, {n_total} samples (train={len(x_train)}, test={len(x_test)})")
    logger.info(f"Training: {num_epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    logger.info(f"{'Metric':<30} {'Without BN':>12} {'With BN':>12}")
    logger.info(f"{'─' * 54}")
    logger.info(f"{'Final train loss':<30} {history_no_bn[-1][0]:>12.4f} {history_bn[-1][0]:>12.4f}")
    logger.info(f"{'Final train accuracy':<30} {history_no_bn[-1][1]:>11.1%} {history_bn[-1][1]:>11.1%}")
    logger.info(f"{'Test accuracy':<30} {test_acc_no_bn:>11.1%} {test_acc_bn:>11.1%}")
    logger.info(f"{'Training time (s)':<30} {time_no_bn:>12.1f} {time_bn:>12.1f}")
    logger.info(f"{'Parameters':<30} {model_no_bn.param_count():>12,} {model_bn.param_count():>12,}")

    # Show convergence trajectory
    logger.info("Convergence trajectory (accuracy at epoch):")
    logger.info(f"{'Epoch':<8} {'Without BN':>12} {'With BN':>12}")
    logger.info(f"{'─' * 32}")
    for ep in [0, 2, 4, 7, 9, 12, min(14, num_epochs - 1)]:
        if ep < len(history_no_bn):
            logger.info(f"  {ep + 1:<8} {history_no_bn[ep][1]:>11.1%} {history_bn[ep][1]:>11.1%}")
