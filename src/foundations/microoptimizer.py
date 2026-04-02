"""
Why Adam converges when SGD stalls — momentum, adaptive learning rates, and the geometry of
loss landscapes, demonstrated head-to-head.
"""

# === TRADEOFFS ===
# + Adam converges faster than SGD on most tasks via adaptive per-parameter learning rates
# + Momentum-based methods escape shallow local minima that trap vanilla SGD
# + Learning rate schedules (warmup + decay) improve final convergence quality
# - Adam uses 3x memory of SGD (stores m and v per parameter)
# - Adaptive methods can generalize worse than well-tuned SGD on some tasks
# - More hyperparameters to tune (beta1, beta2, epsilon, schedule shape)
# WHEN TO USE: Default to Adam/AdamW for most deep learning tasks, especially
#   transformers. Switch to SGD+momentum only if Adam overfits or memory is tight.
# WHEN NOT TO: Extremely memory-constrained training, or convex optimization
#   problems where SGD with a proper schedule converges optimally.

import time
from typing import Callable

import numpy as np

from utility import load_names_data, logger

np.random.default_rng(42)  # for reproducibility in any numpy operations


# === CONSTANTS AND HYPERPARAMETERS ===

# Model architecture — deliberately simple so optimizer differences are visible
N_EMBD = 16  # embedding dimension (small to keep scalar autograd tractable)
VOCAB_SIZE = 0  # set after loading data (number of unique characters + BOS)

# Training parameters — shared across all optimizer runs
NUM_STEPS = 600  # training iterations per optimizer
BATCH_SIZE = 64  # names sampled per step — small because scalar autograd builds a
# full computation graph per Value operation; batch_size=4 with
# ~5 bigrams per name ≈ 20 forward passes per step, each creating
# thousands of Value nodes. Larger batches are cost-prohibitive.

# Optimizer-specific hyperparameters
SGD_LR = 0.05  # SGD needs a larger learning rate to make progress
MOMENTUM_LR = 0.05  # momentum accumulates velocity, so smaller lr is stable
MOMENTUM_BETA = 0.9  # exponential decay rate for velocity (standard choice)
RMSPROP_LR = 0.01  # adaptive lr means we can start smaller
RMSPROP_BETA = 0.99  # decay rate for squared gradient average
RMSPROP_EPS = 1e-8  # prevents division by zero in denominator
ADAM_LR = 0.01  # Adam's bias correction allows moderate lr
ADAM_BETA1 = 0.9  # first moment decay (momentum component)
ADAM_BETA2 = 0.999  # second moment decay (RMSProp component)
ADAM_EPS = 1e-8  # numerical stability in denominator

# Signpost: these hyperparameters are tuned for this specific problem scale. In production,
# Adam(lr=3e-4, beta1=0.9, beta2=0.999) is the standard starting point for most deep learning.

# Extension parameters — warmup + cosine decay applied to Adam
WARMUP_STEPS = 20  # linearly ramp lr from 0 to peak over this many steps
COSINE_LR = 0.01  # peak learning rate after warmup


# === BIGRAM MODEL ===
# A character bigram predicts the next character given only the current character.
# Architecture: embed(char) → linear → softmax → next char distribution.
# This is deliberately simple — the optimizer comparison is the focus, not model sophistication. All four optimizers train this exact same architecture.


def make_params(vocab_size: int, embd_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Initialize model parameters: embedding matrix [vocab_size, embd_dim] and output projection [vocab_size, embd_dim].

    Weight initialization uses Gaussian noise scaled by 1/sqrt(embd_dim) — the Xavier/Glorot heuristic that keeps activation variance roughly constant across layers. Critical for gradient flow in deeper models, helpful here for consistent starting conditions across optimizer runs.
    """
    std = 1.0 / np.sqrt(embd_dim)
    embedding = np.random.normal(0, std, (vocab_size, embd_dim))
    projection = np.random.normal(0, std, (vocab_size, embd_dim))
    return embedding, projection


def clone_params(params: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Deep copy model parameters so each optimizer starts from identical weights.

    Without cloning, all optimizers would share the same Value objects, meaning one optimizer's gradient updates would corrupt another's starting point.
    """
    embedding, projection = params
    return embedding.copy(), projection.copy()


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax: converts logits to probabilities.

    Subtract max(logits) before exp() to prevent overflow. Without this, large logits cause exp() to return inf, breaking the computation.
    Math: softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
    """
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_val = np.exp(shifted)
    return exp_val / np.sum(exp_val, axis=-1, keepdims=True)


def bigram_loss_and_grads(
    params: tuple[np.ndarray, np.ndarray], contexts: np.ndarray, targets: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute cross-entropy loss and parameter gradients for a batch of bigrams"""
    embedding, projection = params
    N = len(contexts)

    # Forward Pass
    emb = embedding[contexts]  # [N, embd_dim]
    logits = emb @ projection.T  # [N, vocab_size]
    probs = softmax(logits)  # [N, vocab_size]
    loss = -np.log(np.clip(probs[np.arange(N), targets], 1e-10, None)).mean()  # cross-entropy loss

    # Backward Pass (compute gradients)
    grad_logits = probs.copy()
    grad_logits[np.arange(N), targets] -= 1  # dL/dlogits
    grad_logits /= N  # average over batch

    grad_projection = grad_logits.T @ emb  # [vocab_size, embd_dim]
    grad_emb = grad_logits @ projection  # [N, embd_dim]
    grad_embedding = np.zeros_like(embedding)
    np.add.at(grad_embedding, contexts, grad_emb)  # accumulate gradients for embedding

    return loss, grad_embedding, grad_projection


# === OPTIMIZER IMPLEMENTATIONS ===
# Each optimizer takes the same interface: a list of Value parameters and their gradients.
# The key insight is that they all compute parameter updates using the same gradient
# information, but accumulate and scale it differently.


def step_sgd(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    learning_rate: float,
    state: dict,
) -> None:
    """Vanilla stochastic gradient descent.

    Update rule: θ = θ - lr * ∇L

    The simplest possible optimizer: move each parameter in the direction opposite to its gradient, scaled by the learning rate. No memory of past gradients.
    """
    for param, grad in zip(params, grads):
        param -= learning_rate * grad


def step_momentum(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    learning_rate: float,
    state: dict,
) -> None:
    """SGD with momentum — adds a velocity term that accumulates past gradients.

    Update rule:
        v = β*v + ∇L           (accumulate velocity)
        θ = θ - lr * v          (update parameters using velocity)

    Momentum acts like a ball rolling downhill: it accelerates through consistent gradient directions and dampens oscillation in directions where gradients alternate sign. β controls how much past gradients influence the current step.
    At β=0.9, the effective window is ~10 past gradients (1/(1-β)).

    This helps escape saddle points and shallow local minima where vanilla SGD stalls.
    """
    if "velocity" not in state:
        state["velocity"] = [np.zeros_like(p) for p in params]  # initialize velocity for each parameter

    for i, (param, grad) in enumerate(zip(params, grads)):
        state["velocity"][i] = MOMENTUM_BETA * state["velocity"][i] + grad  # v = β*v + ∇L
        param -= learning_rate * state["velocity"][i]  # θ = θ - lr


def step_rmsprop(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    learning_rate: float,
    state: dict,
) -> None:
    """RMSProp — adapts the learning rate per-parameter using squared gradient history.

    Update rule:
        s = β*s + (1-β)*∇L²     (running average of squared gradients)
        θ = θ - lr * ∇L/√(s+ε)  (scale update by inverse RMS of gradient history)

    The key insight: dividing by √s normalizes each parameter's update by the typical magnitude of its gradient. Parameters with historically large gradients get smaller effective learning rates (preventing overshooting), while parameters with small gradients get larger effective rates (accelerating learning).

    Signpost: RMSProp was proposed by Hinton in an unpublished lecture. It fixes AdaGrad's problem of monotonically decreasing learning rates by using an exponential moving average instead of a cumulative sum.
    """
    if "sq_avg" not in state:
        state["sq_avg"] = [np.zeros_like(p) for p in params]

    for i, (param, grad) in enumerate(zip(params, grads)):
        state["sq_avg"][i] = RMSPROP_BETA * state["sq_avg"][i] + (1 - RMSPROP_BETA) * grad**2  # s = β*s + (1-β)*∇L²
        param -= learning_rate * grad / (np.sqrt(state["sq_avg"][i]) + RMSPROP_EPS)  # θ = θ - lr * ∇L/√(s+ε)


def step_adam(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    learning_rate: float,
    state: dict,
) -> None:
    """Adam — combines momentum (first moment) with RMSProp (second moment) plus bias correction.

    Update rule:
        m = β1*m + (1-β1)*∇L        (first moment: momentum/mean of gradients)
        v = β2*v + (1-β2)*∇L²       (second moment: uncentered variance of gradients)
        m̂ = m / (1-β1^t)            (bias correction for first moment)
        v̂ = v / (1-β2^t)            (bias correction for second moment)
        θ = θ - lr * m̂ / √(v̂ + ε)  (parameter update)

    Why bias correction matters: m and v are initialized to 0. In early steps, they're biased toward zero because the exponential moving average hasn't had time to "warm up". Without correction, early updates would be much too small (m ≈ 0) or misgauged (v ≈ 0). The correction factor 1/(1-β^t) compensates — it's large when t is small and approaches 1 as t grows.

    Adam's dominance in practice comes from combining the best of both worlds: momentum provides noise-averaged gradient direction, while adaptive scaling provides per-parameter step sizes.
    """
    if "step_count" not in state:
        state["step_count"] = 0
        state["m"] = [np.zeros_like(p) for p in params]
        state["v"] = [np.zeros_like(p) for p in params]

    state["step_count"] += 1
    t = state["step_count"]
    bc1 = 1 - ADAM_BETA1**t
    bc2 = 1 - ADAM_BETA2**t

    for i, (param, grad) in enumerate(zip(params, grads)):
        state["m"][i] = ADAM_BETA1 * state["m"][i] + (1 - ADAM_BETA1) * grad  # m = β1*m + (1-β1)*∇L
        state["v"][i] = ADAM_BETA2 * state["v"][i] + (1 - ADAM_BETA2) * grad**2  # v = β2*v + (1-β2)*∇L²

        m_hat = state["m"][i] / bc1  # m̂ = m / (1-β1^t)
        v_hat = state["v"][i] / bc2  # v̂ = v / (1-β2^t)

        param -= learning_rate * m_hat / (np.sqrt(v_hat) + ADAM_EPS)  # θ = θ - lr * m̂ / √(v̂ + ε)


# === TRAINING LOOP ===
# Train the same bigram model architecture with each optimizer independently.
# Each run starts from identical initial weights (via clone_params) so differences
# in convergence are purely due to the optimizer, not initialization luck.


def train_optimizer(
    optimizer_name: str,
    step_fn: Callable,
    learning_rate: float,
    params: tuple[np.ndarray, np.ndarray],
    bigrams: tuple[np.ndarray, np.ndarray],
    num_steps: int,
    lr_schedule_fn: Callable | None = None,
) -> tuple[list[float], float]:
    """Train a bigram model using a specific optimizer and return loss history."""
    context_data, target_data = bigrams
    param_list = list(params)  # convert tuple to list for mutability
    state: dict = {}  # optimizer state (e.g., momentum velocity, Adam moments)
    loss_history: list[float] = []

    start_time = time.perf_counter()
    for step in range(num_steps):
        idx = np.random.randint(0, len(context_data) - 1, BATCH_SIZE)
        loss, grad_emb, grad_proj = bigram_loss_and_grads(params, context_data[idx], target_data[idx])
        effective_lr = lr_schedule_fn(step, num_steps) if lr_schedule_fn else learning_rate
        step_fn(param_list, [grad_emb, grad_proj], effective_lr, state)
        loss_history.append(loss)

        if step % 100 == 0:
            logger.info(
                f"{optimizer_name:>20s} -> Step {step + 1:>3d}/{num_steps} | Loss: {loss:.4f} | LR: {effective_lr:.5f}"
            )

    return loss_history, time.perf_counter() - start_time


def cosine_schedule(step: int, num_steps: int) -> float:
    """Compute learning rate with linear warmup followed by cosine decay.
    Returns the actual learning rate (not a multiplier), matching Adam's expected lr range.
    """
    if step < WARMUP_STEPS:
        # Linear warmup: lr grows from 0 to COSINE_LR over WARMUP_STEPS
        return COSINE_LR * (step + 1) / WARMUP_STEPS

    # Cosine decay: lr decreases from COSINE_LR to 0 following cos curve
    progress = (step - WARMUP_STEPS) / max(1, num_steps - WARMUP_STEPS)
    return COSINE_LR * 0.5 * (1 + np.cos(np.pi * progress))


# === Create Dataset ===


def create_training_dataset(names_list: list[str]) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, int]]:
    # Build vocabulary from unique characters
    global VOCAB_SIZE
    unique_chars = sorted(set("".join(names_list)))
    char_set = {ch: idx for idx, ch in enumerate(unique_chars)}
    special_tokens = {"BOS": len(unique_chars), "EOS": len(unique_chars) + 1}
    char_set.update(special_tokens)
    VOCAB_SIZE = len(char_set)

    logger.info(f"Vocabulary: {VOCAB_SIZE} tokens ({len(unique_chars)} chars)")

    # Tokenize all names: [BOS, char_0, char_1, ..., char_n, BOS]
    all_context: list[int] = []
    all_targets: list[int] = []
    for name in names_list:
        token_seq = [char_set["BOS"]] + [char_set[ch] for ch in name] + [char_set["EOS"]]
        for i in range(len(token_seq) - 1):
            all_context.append(token_seq[i])
            all_targets.append(token_seq[i + 1])

    context_data = np.array(all_context, dtype=np.int32)
    target_data = np.array(all_targets, dtype=np.int32)
    return (context_data, target_data), char_set


# === COMPARISON AND RESULTS ===


def run_optimizer_comparison(
    sgd_lr: float,
    momentum_lr: float,
    rmsprop_lr: float,
    adam_lr: float,
    cosine_lr: float,
    num_steps: int,
) -> None:
    """Run all optimizers with the given learning rates and step count."""
    global VOCAB_SIZE, NUM_STEPS, COSINE_LR

    NUM_STEPS = num_steps
    COSINE_LR = cosine_lr

    # -- Load and prepare data --
    logger.info("Loading data...")
    docs = load_names_data()
    docs = [doc.strip().lower() for doc in docs.decode("utf-8").splitlines()]
    np.random.shuffle(docs)

    training_data, char_set = create_training_dataset(docs)
    logger.info(f"Total data in dataset: {len(training_data):,}")

    # Initialize base model parameters (shared starting point via cloning)
    base_params = make_params(VOCAB_SIZE, N_EMBD)

    param_count = sum(p.size for p in base_params)
    logger.info(f"Model parameters: {param_count:,}, Training: {num_steps} steps, batch size {BATCH_SIZE}")

    # -- Train with each optimizer --
    optimizers = [
        ("SGD", step_sgd, sgd_lr, None),
        ("SGD + Momentum", step_momentum, momentum_lr, None),
        ("RMSProp", step_rmsprop, rmsprop_lr, None),
        ("Adam", step_adam, adam_lr, None),
        ("Adam + Schedule", step_adam, cosine_lr, cosine_schedule),
    ]

    results: list[tuple[str, list[float], float]] = []

    for name, step_fn, lr, schedule_fn in optimizers:
        logger.info(f"--- {name} -> (lr={lr}) ---")
        params_copy = clone_params(base_params)
        loss_history, elapsed = train_optimizer(
            name,
            step_fn,
            lr,
            params_copy,
            training_data,
            num_steps,
            lr_schedule_fn=schedule_fn,
        )
        results.append((name, loss_history, elapsed))
        logger.info(f"Final loss: {loss_history[-1]:.4f} | Time: {elapsed:.1f}s\n")

    # -- Comparison table --
    # Find the step where each optimizer first drops below a loss threshold.
    # This measures convergence speed: fewer steps = faster convergence.
    loss_threshold = 3.0

    logger.info("=" * 76)
    logger.info(
        f"{'Optimizer':<20s} {'Final Loss':>12s} {'Steps to <' + str(loss_threshold):>16s} "
        f"{'Time (s)':>10s} {'Best Loss':>12s}"
    )
    logger.info("-" * 76)

    for name, loss_history, elapsed in results:
        final_loss = loss_history[-1]
        best_loss = min(loss_history)

        # Find first step below threshold
        steps_to_threshold = "never"
        for step_idx, loss_val in enumerate(loss_history):
            if loss_val < loss_threshold:
                steps_to_threshold = str(step_idx + 1)
                break

        logger.info(f"{name:<20s} {final_loss:>12.4f} {steps_to_threshold:>16s} {elapsed:>10.1f} {best_loss:>12.4f}")

    logger.info("=" * 76)

    # -- Inference: generate names with the best model (Adam + Schedule) --
    # Use the last trained model (Adam + Schedule) for generation
    logger.info("\n--- Generating names with Adam + Schedule model ---\n")
    best_params = clone_params(base_params)
    best_param_list = list(best_params)  # convert to list for mutability
    adam_state: dict = {}

    for step in range(num_steps):
        idx = np.random.randint(0, len(training_data[0]) - 1, BATCH_SIZE)
        _, grad_emb, grad_proj = bigram_loss_and_grads(best_params, training_data[0][idx], training_data[1][idx])
        step_adam(best_param_list, [grad_emb, grad_proj], cosine_schedule(step, num_steps), adam_state)

    # Generate 10 names via autoregressive sampling
    embedding, projection = best_params
    temperature = 0.8

    for sample_idx in range(10):
        token_id = char_set["BOS"]  # start with BOS token
        generated: list[str] = []

        for _ in range(20):  # max name length
            # Forward pass: embed → project → softmax
            emb = embedding[token_id]
            logits = projection @ emb
            logits = (logits - logits.max()) / temperature  # apply temperature scaling
            exp_vals = np.exp(logits)
            probs = exp_vals / np.sum(exp_vals)

            token_id = int(np.random.choice(VOCAB_SIZE, p=probs))  # sample next token
            if token_id == char_set["EOS"] or token_id == char_set["BOS"]:
                break
            generated.append(next(ch for ch, idx in char_set.items() if idx == token_id))

        logger.info(f"  {sample_idx + 1:>2}. {''.join(generated)}")


if __name__ == "__main__":
    # === DEFAULT BEHAVIOR (unchanged) ===
    run_optimizer_comparison(SGD_LR, MOMENTUM_LR, RMSPROP_LR, ADAM_LR, COSINE_LR, NUM_STEPS)
