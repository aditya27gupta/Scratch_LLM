"""
How text becomes numbers.

The compression algorithm hiding inside every LLM. Byte-Pair Encoding learns a vocabulary by iteratively merging the most frequent adjacent token pairs, then encodes new text by replaying those merges in priority order.
"""

import time

import numpy as np

from utility import load_names_data, logger

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


class Tokenizer:
    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = np.int32(vocab_size)
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

    def apply_merge(self, token_ids: np.ndarray, new_pair: tuple[int, int], new_id: int) -> np.ndarray:
        results = token_ids.copy()
        all_pos = np.where((token_ids[:-1] == new_pair[0]) & (token_ids[1:] == new_pair[1]))[0]
        if len(all_pos) > 1:
            all_pos = all_pos[np.insert(np.diff(all_pos) > 0, 0, True)]
        skip = np.zeros(len(token_ids), dtype=bool)
        skip[all_pos + 1] = True
        results[all_pos] = new_id
        return results[~skip]

    def _best_pair(self, token_ids: np.ndarray) -> tuple[tuple[np.int32, np.int32], np.int32]:
        temp = np.empty(len(token_ids) - 1, dtype=np.int32)
        np.multiply(token_ids[:-1], self.vocab_size, out=temp)
        np.add(temp, token_ids[1:], out=temp)
        counts = np.bincount(temp)
        best = int(counts.argmax())
        return (best // self.vocab_size, best % self.vocab_size), counts[best]

    def train_bpe(self, token_ids: np.ndarray) -> None:
        ids = token_ids.copy()
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            if len(ids) < 2:
                break
            pair, pair_count = self._best_pair(ids)
            new_id = 256 + i
            ids = self.apply_merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if (i + 1) % 32 == 0:
                logger.info(f"Merge {i + 1}: ({pair[0]}, {pair[1]}) -> {new_id} freq={pair_count}")

        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def encode(self, text: str) -> np.ndarray:
        token_ids = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(dtype=np.uint32)
        for pair, new_id in self.merges.items():
            if len(token_ids) < 2:
                break
            token_ids = self.apply_merge(token_ids, pair, new_id)
        return token_ids

    def decode(self, token_ids: np.ndarray) -> str:
        raw_bytes = b"".join(self.vocab[tid] for tid in token_ids)
        return raw_bytes.decode("utf-8")


if __name__ == "__main__":
    raw_data = load_names_data()
    corpus_ids = np.frombuffer(raw_data, dtype=np.uint8).astype(dtype=np.uint32)
    logger.info(f"Corpus: {len(raw_data):} bytes, base vocab: 256 token")
    tokenizer = Tokenizer()
    start_time = time.perf_counter()
    logger.info("Training BPE...")
    tokenizer.train_bpe(corpus_ids)
    duration = time.perf_counter() - start_time
    logger.info(f"Training took {duration:.2f} seconds. {len(tokenizer.merges)} merges learned")

    test_strings = ["Emma", "Xiomara", "Mary-Jane", "O'Brien", "", "Z"]
    for s in test_strings:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        status = "PASS" if decoded == s else "FAIL"
        color = RED if status == "FAIL" else GREEN
        logger.info(f"[{color}{status}{RESET}] -> {len(encoded)} tokens -> {decoded!r}")

    corpus_text = raw_data.decode("utf-8")
    corpus_encoded = tokenizer.encode(corpus_text)
    ratio = len(raw_data) / len(corpus_encoded)
    logger.info(f"Compression: {len(raw_data)} bytes -> {len(corpus_encoded)} tokens -> Ratio: {ratio:.2f}")
