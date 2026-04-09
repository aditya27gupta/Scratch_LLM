"""
Why randomly destroying neurons during training prevents overfitting — dropout, weight decay,
and early stopping as complementary strategies.
"""

# === TRADEOFFS ===
# + Prevents co-adaptation: forces distributed representations across neurons
# + Approximates ensemble of exponentially many sub-networks
# + Zero additional parameters: regularization via training-time noise
# - Increases training time (effectively training on partial network each step)
# - Requires scaling at inference time (or inverted dropout during training)
# - Interacts unpredictably with batch normalization (both modify activations)
# WHEN TO USE: When your model has excess capacity relative to data size and is
#   overfitting on training data. Standard for fully connected and attention layers.
# WHEN NOT TO: When the model is underfitting (dropout will make it worse), or
#   in convolutional layers where spatial dropout is more appropriate.

import time

import numpy as np

from utility import load_names_data, logger

data_rng = np.random.default_rng(42)  # for reproducibility


def build_dataset(
    names: list[str],
    stoi: dict[str, int],
    context_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (context, target) pairs from names using a sliding window.

    Each training example is a fixed-size context window of character indices
    followed by the target character index. Names are padded with '.' (the
    boundary token) so the model learns to start and end names.

    Example with context_size=3: "emma" -> [(..., ..., .), e], [(..., ., e), m], ...
    """
    xs: list[list[int]] = []
    ys: list[int] = []
    for name in names:
        padded = ["."] * context_size + list(name) + ["."]
        for i in range(len(padded) - context_size):
            context = [stoi[ch] for ch in padded[i : i + context_size]]
            target = stoi[padded[i + context_size]]
            xs.append(context)
            ys.append(target)
    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


# === MODEL DEFINITION ===

# Architecture: character-level MLP with one hidden layer.
# Input: CONTEXT_SIZE character indices -> concatenated embeddings -> hidden layer -> output logits.
# This is essentially Bengio et al. (2003) neural language model, simple enough to overfit
# quickly but complex enough that regularization makes a measurable difference.


def init_model(vocab_size: int) -> dict[str, np.ndarray]:
    """Initialize model parameters: embeddings, hidden layer, output layer."""
    params: dict[str, np.ndarray] = {
        "emb": data_rng.normal(0, 0.1, size=(vocab_size, N_EMBD)),
        "w1": data_rng.normal(0, 0.1, size=(N_HIDDEN, CONTEXT_SIZE * N_EMBD)),
        "b1": data_rng.normal(0, 0.1, size=(1, N_HIDDEN)),
        "w2": data_rng.normal(0, 0.1, size=(vocab_size, N_HIDDEN)),
        "b2": data_rng.normal(0, 0.1, size=(1, vocab_size)),
    }
    return params


# === REGULARIZATION IMPLEMENTATIONS ===


def apply_dropout(
    activations: np.ndarray,
    training: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply inverted dropout to a layer's activations.

    During training: randomly zero each activation with probability drop_prob, then scale surviving activations by 1/(1-p) to maintain expected value.

    During inference: use all activations unchanged. The 1/(1-p) scaling during training ("inverted dropout") means no correction is needed at test time.

    Math: h_drop[i] = h[i] * mask[i] / (1 - p) where mask[i] ~ Bernoulli(1 - p)

    Intuition: dropout forces redundancy. If any neuron might be absent, the network can't rely on a single neuron to encode a pattern. Instead, information is spread across many neurons — a kind of learned ensemble.
    Signpost: production transformers typically use dropout=0.1. Larger models have better implicit regularization from sheer parameter count.
    """
    if not training:
        return activations, None

    scale = 1.0 / (1.0 - DROPOUT_P)
    mask = (data_rng.random(activations.shape) >= DROPOUT_P).astype(np.float16)
    result: np.ndarray = activations * mask * scale
    return result, mask


class EarlyStopper:
    """Monitor validation loss and signal when to stop training.

    Tracks the best validation loss seen so far. If validation loss fails to improve for `patience` consecutive checks, signals that training should stop.

    This prevents the model from continuing to memorize training data after it has already learned the generalizable patterns. The validation loss curve typically has a U-shape: it decreases as the model learns real patterns, reaches a minimum, then increases as the model starts memorizing noise.
    Early stopping halts training near the minimum.
    """

    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def check(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# === FORWARD PASS AND LOSS ===


def forward(
    context_batch: np.ndarray,
    params: dict[str, np.ndarray],
    training: bool = True,
    use_dropout: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Forward pass through the character-level MLP.

    1. Look up embeddings for each context character
    2. Concatenate into a single input vector
    3. Hidden layer: linear transform -> tanh activation -> optional dropout
    4. Output layer: linear transform -> logits

    The architecture is simple enough that each regularization technique's effect is clearly visible in the train/val loss gap.
    """
    # Step 1: Embedding lookup and concatenation
    # Each context character becomes an N_EMBD-dimensional vector; we concatenate them into a single (CONTEXT_SIZE * N_EMBD)-dimensional input.
    emb_concat = params["emb"][context_batch].reshape(len(context_batch), -1)

    # Step 2: Hidden layer — h = tanh(W1 @ x + b1)
    h_pre = emb_concat @ params["w1"].T + params["b1"]  # shape: (N_HIDDEN,)
    h = np.tanh(h_pre)  # shape: (N_HIDDEN,)

    # Step 3: Apply dropout to hidden activations (only during training if enabled)
    mask = None
    h_drop = h
    if use_dropout:
        h_drop, mask = apply_dropout(h, training)

    # Step 4: Output layer — logits = W2 @ h + b2
    logits = h_drop @ params["w2"].T + params["b2"]  # shape: (vocab_size,)
    cache = {
        "context_batch": context_batch,
        "emb_concat": emb_concat,
        "h": h,
        "h_drop": h_drop,
        "mask": mask,
        "use_dropout": use_dropout and training,
    }
    return logits, cache


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)


def backward(
    probs: np.ndarray,
    target_idx: np.ndarray,
    cache: dict[str, np.ndarray],
    params: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Backward pass to compute gradients for a single training example.

    1. Compute gradient of loss w.r.t. logits (softmax + cross-entropy)
    2. Backprop through output layer to get gradients for W2, b2, and hidden activations
    3. Backprop through dropout mask if applied
    4. Backprop through tanh activation to get gradients for W1, b1, and input embeddings

    This manual backward pass is necessary because we're using scalar autograd and need to explicitly compute gradients for each parameter.
    """
    batch_size = target_idx.shape[0]
    # Step 1: Gradient of loss w.r.t. logits
    d_logits = probs.copy()
    d_logits[np.arange(batch_size), target_idx] -= 1  # dL/d(logits) = probs - one_hot(target)
    d_logits /= batch_size  # average over batch

    # Step 2: Output layer gradients
    db2 = np.sum(d_logits, axis=0)  # shape: (vocab_size,)
    dw2 = d_logits.T @ cache["h_drop"]  # shape: (vocab_size, N_HIDDEN)
    dh_drop = d_logits @ params["w2"]  # shape: (N_HIDDEN,)

    # Step 3: Dropout backprop
    dh = dh_drop * cache["mask"] / (1.0 - DROPOUT_P) if cache["use_dropout"] else dh_drop

    # Step 4: Hidden layer backprop
    dh_pre = dh * (1 - cache["h"] ** 2)  # dL/d(h_pre) using tanh derivative
    db1 = np.sum(dh_pre, axis=0)  # shape: (N_HIDDEN,)
    dw1 = dh_pre.T @ cache["emb_concat"]  # shape: (N_HIDDEN, CONTEXT_SIZE * N_EMBD)
    demb_concat = dh_pre @ params["w1"]  # shape: (CONTEXT_SIZE * N_EMBD,)

    # Embedding gradients: reshape back to (CONTEXT_SIZE, N_EMBD) and accumulate into the embedding matrix
    demb = np.zeros_like(params["emb"])  # shape: (CONTEXT_SIZE * N_EMBD,)
    d_emb_3d = demb_concat.reshape(batch_size, CONTEXT_SIZE, N_EMBD)
    for i in range(CONTEXT_SIZE):
        np.add.at(
            demb, cache["context_batch"][:, i], d_emb_3d[:, i, :]
        )  # accumulate gradients for each context character

    return {"emb": demb, "w1": dw1, "b1": db1, "w2": dw2, "b2": db2}


def eval_loss(
    xs: np.ndarray,
    ys: np.ndarray,
    params: dict[str, np.ndarray],
) -> float:
    """Evaluate loss without dropout (inference mode) over all examples.

    No dropout applied — at inference time, all neurons are active.
    """
    logits, _ = forward(xs, params, training=False)
    probs = softmax(logits)
    return -np.log(np.maximum(probs[np.arange(ys.shape[0]), ys], 1e-10)).mean()


# === TRAINING LOOP ===


def train_model(
    train_dataset: tuple[np.ndarray, np.ndarray],
    val_dataset: tuple[np.ndarray, np.ndarray],
    params: dict[str, np.ndarray],
    train_params: dict[str, int],
    config_name: str,
    use_dropout: bool = False,
    use_weight_decay: bool = False,
    use_early_stopping: bool = False,
) -> tuple[float, float, int, float]:
    """Train one model configuration and return (train_loss, val_loss, steps_run).

    Each configuration starts from a fresh random initialization (re-seeded for fair comparison) and trains for up to NUM_STEPS steps. The only difference between runs is the regularization strategy applied.
    """
    start_time = time.perf_counter()
    train_xs, train_ys = train_dataset
    val_xs, val_ys = val_dataset

    stopper = EarlyStopper(EARLY_STOP_PATIENCE) if use_early_stopping else None
    steps_run = 0
    n = train_xs.shape[0]

    logger.info(f"--- {config_name} ---")
    num_steps = train_params["num_steps"]

    for step in range(num_steps):
        # Sample a random mini-batch
        idx = data_rng.integers(0, n, size=train_params["batch_size"])

        # Forward pass
        logits, cache = forward(train_xs[idx], params, training=True, use_dropout=use_dropout)
        probs = softmax(logits)

        # Backward pass
        grads = backward(probs, train_ys[idx], cache, params)

        lr_t = train_params["learning_rate"] * (1.0 - step / num_steps)
        for key in params:
            if use_weight_decay:
                params[key] -= lr_t * (grads[key] + WEIGHT_DECAY * params[key])
            else:
                params[key] -= lr_t * grads[key]

        steps_run = step + 1

        # Periodic evaluation
        if (step + 1) % EVAL_INTERVAL == 0:
            t_loss = eval_loss(train_xs, train_ys, params)
            v_loss = eval_loss(val_xs, val_ys, params)
            logger.info(f"  step {step + 1:>4}/{num_steps} | train: {t_loss:.4f} | val: {v_loss:.4f}")

            # Early stopping check
            if stopper is not None and stopper.check(v_loss):
                logger.info(f"** early stopping at step {step + 1} (patience={EARLY_STOP_PATIENCE})")
                break

    # Final evaluation
    final_train = eval_loss(train_xs, train_ys, params)
    final_val = eval_loss(val_xs, val_ys, params)
    logger.info(f"final    | train: {final_train:.4f} | val: {final_val:.4f}")

    return final_train, final_val, steps_run, round(time.perf_counter() - start_time, 2)


# === COMPARISON AND RESULTS ===

if __name__ == "__main__":
    # -- Load and prepare data --
    logger.info("Loading data...")
    r_names = load_names_data()
    names = [name.lower() for name in r_names.decode("utf-8").splitlines() if name.strip()]

    # Model Parameters
    N_EMBD = 16  # embedding dimension
    N_HIDDEN = 64  # hidden layer size (4x embedding = substantial excess capacity)
    CONTEXT_SIZE = 5  # number of preceding characters used as input

    # Training parameters
    train_params = {"learning_rate": 0.05, "num_steps": 4000, "batch_size": 16}

    # Regularization hyperparameters
    DROPOUT_P = 0.1  # probability of zeroing each hidden activation
    WEIGHT_DECAY = 0.001  # L2 penalty coefficient (lambda)

    # Early stopping
    EARLY_STOP_PATIENCE = 5  # consecutive checks with rising val loss before stopping
    EVAL_INTERVAL = 200  # steps between validation loss checks

    # Build vocabulary: 26 lowercase letters + '.' boundary token
    chars = sorted(set("".join(names)))
    chars = ["."] + chars  # '.' at index 0 as the boundary/padding token
    stoi = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    # Use a subset of names for tractable training with scalar autograd.
    # Smaller training set makes overfitting more pronounced, which is exactly
    # what we want — it amplifies the regularization signal.
    data_rng.shuffle(names)
    logger.info(f"Using {len(names)} names (subset), vocabulary size: {vocab_size}")

    # Split data 80/20 before building examples.
    # Splitting names (not examples) prevents data leakage: no name appears in both train and validation, so the model can't memorize specific names during training and get credit for them during validation.
    split_idx = int(0.8 * len(names))
    train_names = names[:split_idx]
    val_names = names[split_idx:]

    train_dataset = build_dataset(train_names, stoi, CONTEXT_SIZE)
    val_dataset = build_dataset(val_names, stoi, CONTEXT_SIZE)
    params = init_model(vocab_size)

    logger.info(f"Training size: {train_dataset[0].shape[0]}, Validation size: {val_dataset[0].shape[0]}")
    logger.info(f"Parameters per model: {np.sum([params[key].size for key in params])}")

    # -- Run five training configurations --
    # Each configuration uses identical architecture and initialization (via re-seeding).
    # The only variable is the regularization strategy. This controlled experiment isolates the effect of each technique.

    results: list[tuple[str, float, float, int, float]] = []

    # Config 1: No regularization (baseline — expected to overfit)
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name="No regularization (baseline)",
    )
    results.append(("No regularization", t, v, s, d))

    # Config 2: Dropout only
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name=f"Dropout only (p={DROPOUT_P})",
        use_dropout=True,
    )
    results.append(("Dropout", t, v, s, d))

    # Config 3: Weight decay only
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name=f"Weight decay only (lambda={WEIGHT_DECAY})",
        use_weight_decay=True,
    )
    results.append(("Weight decay", t, v, s, d))

    # Config 4: Dropout + weight decay (combined)
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name=f"Dropout + weight decay (p={DROPOUT_P}, lambda={WEIGHT_DECAY})",
        use_dropout=True,
        use_weight_decay=True,
    )
    results.append(("Dropout + weight decay", t, v, s, d))

    # Config 5: Early stopping (monitors validation loss, stops when it rises)
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name=f"Early stopping (patience={EARLY_STOP_PATIENCE})",
        use_early_stopping=True,
    )
    results.append(("Early stopping", t, v, s, d))

    # Config 6: Combined Early stopping + Dropout + Weight Decay
    param_copy = params.copy()
    t, v, s, d = train_model(
        train_dataset,
        val_dataset,
        param_copy,
        train_params,
        config_name="Early stopping + Dropout + weight decay",
        use_early_stopping=True,
        use_dropout=True,
        use_weight_decay=True,
    )
    results.append(("Early stopping + Dropout + weight decay", t, v, s, d))

    # -- logger.info comparison table --
    # The "gap" column is the key metric: it measures how much worse the model
    # performs on unseen data compared to training data. A large gap = overfitting.
    # Effective regularization reduces this gap while keeping validation loss low.
    logger.info("\n" + "=" * 78)
    logger.info("REGULARIZATION COMPARISON")
    logger.info("=" * 78)
    logger.info(f"{'Strategy':<40} {'Train':>8} {'Val':>8} {'Gap':>8} {'Steps':>7} {'Duration':>7}")
    logger.info("-" * 78)
    for name, train_loss, val_loss, steps, duration in results:
        gap = val_loss - train_loss
        logger.info(f"{name:<40} {train_loss:>8.4f} {val_loss:>8.4f} {gap:>+8.4f} {steps:>7} {duration:.2f}")
    logger.info("-" * 78)

    # Identify best configuration by validation loss
    best = min(results, key=lambda r: r[2])
    logger.info(f"Best generalization: {best[0]} (val loss: {best[2]:.4f})")
