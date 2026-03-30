"""
How meaning becomes geometry

Training vectors where distance equals similarity, using only character n-grams and contrastive loss.
"""

import time
from collections import Counter

import numpy as np

from utility import load_names_data, logger


class EmbeddingModel:
    def __init__(self, vocab_size: int = 1_000, embedding_dim: int = 32) -> None:
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_params = embedding_dim * vocab_size
        # Initialize projection matrix W: [embedding_dim × vocab_size]
        self.W = np.random.normal(0, 0.01, (embedding_dim, vocab_size))
        self.vocab = {}  # n-gram to index mapping

    def extract_ngrams(self, name: str) -> list[str]:
        n_grams = []
        name = f"^{name}$"  # Add boundary markers
        for n in range(2, 4):  # Use bigrams and trigrams
            n_grams.extend(name[i : i + n] for i in range(len(name) - n + 1))
        return n_grams

    def build_n_gram_vocab(self, names: list[str]) -> None:
        counts: Counter[str] = Counter()
        for name in names:
            n_grams = self.extract_ngrams(name)
            counts.update(n_grams)
        most_common = counts.most_common(self.vocab_size)
        self.vocab = {ngram: idx for idx, (ngram, _) in enumerate(most_common)}

    def encode_ngrams_sparse(self, text: str) -> np.ndarray:
        """Convert text to sparse n-gram count dict (index → count).

        Returns only non-zero entries. This is critical for performance: names have ~10-15 n-grams out of a vocab of 500, so sparse representation skips 97% of the computation in gradient and encoder loops.
        """
        sparse: np.ndarray = np.zeros(self.vocab_size)
        for ngram in self.extract_ngrams(text):
            if ngram in self.vocab:
                idx = self.vocab[ngram]
                sparse[idx] += 1.0
        return sparse

    def encode_sparse_raw(self, sparse_ngrams: np.ndarray) -> np.ndarray:
        """Project sparse n-gram features to embedding space WITHOUT normalization.

        Math: z = W @ x (raw, unnormalized embedding)
        Used in training where we need to backpropagate through normalization.
        """
        embedding = np.dot(self.W, sparse_ngrams)
        return embedding

    def l2_normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length.

        Why L2 normalization: constrains embeddings to the unit hypersphere. After
        normalization, cosine similarity = dot product, which simplifies the math
        and makes the embedding space isotropic (all directions have equal variance).
        This is standard practice in contrastive learning (SimCLR, CLIP).
        """
        norm = np.sqrt(np.sum(vec**2))
        if norm < 1e-10:
            return vec
        return vec / norm

    def augment(self, name: str) -> str:
        """Create positive pair by random character deletion or swap.

        Why augmentation: forces the encoder to learn invariances to small changes.
        If "anna" and "ana" map to similar embeddings, the model has learned that character deletion preserves identity — this is the contrastive learning principle that similar inputs should have similar representations.
        """
        if len(name) <= 2:
            return name  # too short to augment safely

        if np.random.random() < 0.5:
            # Delete one random character
            idx = np.random.randint(0, len(name) - 1)
            return name[:idx] + name[idx + 1 :]
        else:
            # Swap two adjacent characters
            idx = np.random.randint(0, len(name) - 2)
            chars = list(name)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors.

        Since vectors are L2-normalized, cosine similarity = dot product.
        Range: [-1, 1] where 1 = identical direction, -1 = opposite, 0 = orthogonal.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def infonce_loss_and_grads(
        self,
        anchor_embs: np.ndarray,
        positive_embs: np.ndarray,
        temperature: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute InfoNCE (NT-Xent) loss and embedding-space gradients.

        For each (anchor, positive) pair in the batch, the loss encourages high similarity to the positive and low similarity to all negatives (other samples in the batch).

        Math (for anchor i):
            sim_pos = cos(anchor_i, positive_i) / tau
            sim_neg_j = cos(anchor_i, anchor_j) / tau   for j != i
            loss_i = -log(exp(sim_pos) / (exp(sim_pos) + sum_j exp(sim_neg_j)))

        Why temperature: controls sharpness of the similarity distribution. Low tau (e.g. 0.1) makes the loss focus on hard negatives. tau=0.1 is standard in SimCLR.

        Returns: (avg_loss, anchor_grads, positive_grads)
        """
        bs = len(anchor_embs)
        total_loss = 0.0

        anchor_grads = np.zeros((bs, self.embedding_dim))
        positive_grads = np.zeros((bs, self.embedding_dim))

        for i in range(bs):
            # Similarity to positive pair
            sim_pos = self.cosine_similarity(anchor_embs[i], positive_embs[i]) / temperature

            # Similarities to all negatives (other anchors in batch)
            sim_negs: np.ndarray = np.dot(anchor_embs[i], anchor_embs.T) / temperature  # shape: [batch_size]
            sim_negs = np.delete(sim_negs, i)  # Remove similarity to itself

            # Log-sum-exp trick for numerical stability (subtract max before exp)
            max_sim = max([sim_pos] + sim_negs)
            exp_pos = np.exp(sim_pos - max_sim)
            exp_negs = np.exp(sim_negs - max_sim)
            denom = exp_pos + sum(exp_negs)

            # Loss: -log(softmax probability of positive pair)
            total_loss += -np.log(max(exp_pos / denom, 1e-10))

            # Gradient of loss w.r.t. anchor embedding:
            # d(loss)/d(anchor_i) = (1/tau) * (sum_j p_j * anchor_j - positive_i)
            # where p_j = exp(sim_neg_j) / denom is the softmax probability

            # Positive contribution: pushes anchor toward positive
            p_pos = exp_pos / denom
            anchor_grads[i] += (p_pos - 1.0) / temperature * positive_embs[i]
            positive_grads[i] += (p_pos - 1.0) / temperature * anchor_embs[i]

            # Negative contributions: pushes anchor away from negatives
            p_neg = exp_negs / denom  # shape: [num_negatives]
            anchor_grads[i] += (
                np.sum(p_neg[:, np.newaxis] * anchor_embs[[j for j in range(bs) if j != i]], axis=0) / temperature
            )

        return total_loss / bs, anchor_grads, positive_grads

    def grad_through_norm(self, raw_emb: np.ndarray, grad_normalized: np.ndarray) -> np.ndarray:
        """Backpropagate gradient through L2 normalization.

        If z = raw_emb and e = z/||z|| (the normalized embedding), then:
            d(L)/d(z_i) = (g_i - e_i * dot(g, e)) / ||z||

        The normalization Jacobian projects out the radial component of the gradient, leaving only the tangential direction on the unit sphere. Without this projection, gradients can push all embeddings in the same radial direction, causing "representation collapse" — the most common failure mode in contrastive learning.
        """
        norm = np.sqrt(np.sum(raw_emb**2))
        if norm < 1e-10:
            return grad_normalized
        e = raw_emb / norm
        g_dot_e = np.dot(grad_normalized, e)
        return (grad_normalized - e * g_dot_e) / norm

    def train(
        self,
        names: list[str],
        num_epochs: int = 30,
        batch_size: int = 1_000,
        learning_rate: float = 0.05,
        temperature: float = 0.1,
    ) -> None:
        sparse_matrix = np.zeros((len(names), self.vocab_size))
        for i, name in enumerate(names):
            sparse_matrix[i] = self.encode_ngrams_sparse(name)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_start in range(0, len(names), batch_size):
                batch = names[batch_start : batch_start + batch_size]
                if len(batch) < 2:
                    continue

                # Encode anchors and positives (sparse n-grams → dense embeddings). Store both raw (pre-normalization) and normalized embeddings and raw embeddings are needed for the normalization Jacobian in backprop
                anchor_sparse = np.zeros((len(batch), self.vocab_size))
                positive_sparse = np.zeros((len(batch), self.vocab_size))
                anchor_raw = np.zeros((len(batch), self.embedding_dim))
                positive_raw = np.zeros((len(batch), self.embedding_dim))
                anchor_embs = np.zeros((len(batch), self.embedding_dim))
                positive_embs = np.zeros((len(batch), self.embedding_dim))

                for i in range(len(batch)):
                    a_sp = sparse_matrix[batch_start + i]
                    anchor_sparse[i] = a_sp
                    p_sp = self.encode_ngrams_sparse(self.augment(batch[i]))
                    positive_sparse[i] = p_sp

                anchor_raw = self.encode_sparse_raw(anchor_sparse.T).T  # [batch_size × embedding_dim]
                anchor_embs = self.l2_normalize(anchor_raw)
                positive_raw = self.encode_sparse_raw(positive_sparse.T).T  # [batch_size × embedding_dim]
                positive_embs = self.l2_normalize(positive_raw)

                # Compute loss and gradients w.r.t. NORMALIZED embeddings
                loss, a_grads, p_grads = self.infonce_loss_and_grads(anchor_embs, positive_embs, temperature)
                epoch_loss += loss
                num_batches += 1

                # Backpropagate gradients to W using SPARSE computation.
                # Chain rule: d(L)/d(W) = d(L)/d(emb_norm) * d(emb_norm)/d(emb_raw) * d(emb_raw)/d(W)
                # The normalization Jacobian (middle term) projects out the radial
                # gradient component, preventing representation collapse.
                grad_W = np.zeros((self.embedding_dim, self.vocab_size))  # [embedding_dim × vocab_size]

                for b_idx in range(len(batch)):
                    # Transform gradients through normalization Jacobian
                    a_grad_raw = self.grad_through_norm(anchor_raw[b_idx], a_grads[b_idx])
                    p_grad_raw = self.grad_through_norm(positive_raw[b_idx], p_grads[b_idx])

                    grad_W += a_grad_raw[:, np.newaxis] * anchor_sparse[b_idx][np.newaxis, :]
                    grad_W += p_grad_raw[:, np.newaxis] * positive_sparse[b_idx][np.newaxis, :]

                # SGD update (only for entries with non-zero gradients)
                scale = learning_rate / len(batch)
                np.subtract(self.W, scale * grad_W, out=self.W)  # In-place update for efficiency

            avg_loss = epoch_loss / max(num_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch + 1:>3}/{num_epochs}  loss={avg_loss:.4f}")

    def encode_sparse(self, sparse_ngrams: np.ndarray) -> np.ndarray:
        """Project sparse n-gram features to embedding space and normalize.

        Math: emb = normalize(W @ x)
        Sparse version: only sums over non-zero entries in x, which is 10-15
        n-grams instead of the full 500-entry vocabulary.
        """
        return self.l2_normalize(self.encode_sparse_raw(sparse_ngrams))

    def find_nearest_neighbors(
        self,
        query: str,
        candidates: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """Find k nearest neighbors by cosine similarity in embedding space."""
        q_emb = self.encode_sparse(self.encode_ngrams_sparse(query))

        similarities = []
        for candidate in candidates:
            if candidate == query:
                continue
            c_emb = self.encode_sparse(self.encode_ngrams_sparse(candidate))
            sim = self.cosine_similarity(q_emb, c_emb)
            similarities.append((candidate, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


if __name__ == "__main__":
    # Load the names data
    names_data = load_names_data()
    names_list = [name.lower().strip() for name in names_data.decode("utf-8").splitlines()]
    logger.info(f"Loaded {len(names_list)} names from the dataset.")

    embedding_model = EmbeddingModel()
    # Build n-gram vocabulary from training set (capped at MAX_VOCAB)
    embedding_model.build_n_gram_vocab(names_list)
    logger.info(f"Built n-gram vocabulary with {len(embedding_model.vocab)} entries.")

    # Train
    logger.info("Starting training...")
    start = time.perf_counter()
    embedding_model.train(names_list)
    time_duration = time.perf_counter() - start
    logger.info(f"Training completed in {time_duration:.2f} seconds.")

    # === EVALUATION ===
    logger.info("Evaluating embedding similarities on test pairs:")
    # Positive pairs: similar-sounding names (should have high similarity)
    positive_pairs = [
        ("anna", "anne"),
        ("john", "jon"),
        ("elizabeth", "elisabeth"),
        ("michael", "michelle"),
        ("alexander", "alexandra"),
    ]

    # Random pairs: dissimilar names (should have low similarity)
    random_pairs = [
        ("anna", "zachary"),
        ("john", "penelope"),
        ("elizabeth", "bob"),
        ("michael", "quinn"),
        ("alexander", "ivy"),
    ]

    pos_sims = []
    rand_sims = []
    for name1, name2 in positive_pairs + random_pairs:
        e1 = embedding_model.encode_sparse(embedding_model.encode_ngrams_sparse(name1))
        e2 = embedding_model.encode_sparse(embedding_model.encode_ngrams_sparse(name2))
        sim = embedding_model.cosine_similarity(e1, e2)
        if (name1, name2) in positive_pairs:
            pos_sims.append(sim)
        else:
            rand_sims.append(sim)
        logger.info(f"{name1:<12} <-> {name2:<12}  sim={sim:>6.3f}")

    logger.info(f"Average positive pair similarity: {sum(pos_sims) / len(pos_sims):.3f}")
    logger.info(f"Average random pair similarity:   {sum(rand_sims) / len(rand_sims):.3f}")

    query_names = ["anna", "john", "elizabeth", "michael"]
    logger.info("\nNearest neighbor retrieval:")
    for query in query_names:
        neighbors = embedding_model.find_nearest_neighbors(query, names_list, k=5)
        neighbor_str = ", ".join(f"{n} ({s:.2f})" for n, s in neighbors)
        logger.info(f"{query:<12} -> {neighbor_str}")
