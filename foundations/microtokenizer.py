"""
How text becomes numbers.

The compression algorithm hiding inside every LLM.
Byte-Pair Encoding learns a vocabulary by iteratively merging the most frequent
adjacent token pairs, then encodes new text by replaying those merges in priority order.
"""

import itertools
from collections import Counter
from pathlib import Path

import requests

from log import logger

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

VOCAB_SIZE = 512  # Final vocab = 256 byte tokens + 256 merges = 512 tokens.
DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
DATA_FILE = "data/names.txt"


def load_data(url: str, filename: str) -> bytes:
    """Download dataset if not cached, return raw bytes."""
    file_path = Path(filename)
    if not file_path.exists():
        logger.info(f"Downloading {filename}...")
        response = requests.get(url=url, allow_redirects=True, timeout=5)
        response.raise_for_status()
        file_path.write_bytes(response.content)
    return file_path.read_bytes()


class Tokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: dict[tuple[int, int], int] = {}

    def apply_merge(self, token_ids: list[int], new_pair: tuple[int, int], new_id: int) -> list[int]:
        i = 0
        new_token_ids = []
        while i < len(token_ids):
            if i < len(token_ids) - 1 and token_ids[i] == new_pair[0] and token_ids[i + 1] == new_pair[1]:
                new_token_ids.append(new_id)
                i += 2
                continue
            new_token_ids.append(token_ids[i])
            i += 1
        return new_token_ids

    def train_bpe(self, token_ids: list[int]) -> None:
        ids = list(token_ids)
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            counts = Counter(itertools.pairwise(ids))
            if not counts:
                break
            pair, pair_count = counts.most_common(1)[0]
            new_id = 256 + i
            ids = self.apply_merge(ids, pair, new_id)
            self.merges[pair] = new_id
            if (i + 1) % 32 == 0:
                logger.info(f"Merge {i + 1}: ({pair[0]}, {pair[1]}) -> {new_id} freq={pair_count}")

    def build_vocab(self) -> None:
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    def encode(self, text: str) -> list[int]:
        token_ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges.items():
            token_ids = self.apply_merge(token_ids, pair, new_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[tid] for tid in token_ids)
        return raw_bytes.decode("utf-8")


if __name__ == "__main__":
    raw_data = load_data(DATA_URL, DATA_FILE)
    corpus_ids = list(raw_data)
    logger.info(f"Corpus: {len(raw_data):} bytes, base vocab: 256 token, final_vocab: {VOCAB_SIZE} token")
    tokenizer = Tokenizer(VOCAB_SIZE)
    logger.info("Training BPE...")
    tokenizer.train_bpe(corpus_ids)
    tokenizer.build_vocab()
    logger.info(f"Training complete. {len(tokenizer.merges)} merges learned")

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
