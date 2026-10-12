from functools import lru_cache

import regex
from collections.abc import Iterable, Iterator

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_SPLIT_PATTERN_RE = regex.compile(GPT2_SPLIT_PATTERN)


def _bpe_merge(
    raw_bytes: bytes, reverse_vocab: dict[bytes, int], merge_ranks: dict[tuple[bytes, bytes], int]
) -> tuple[int, ...]:
    byte_list = [bytes([b]) for b in raw_bytes]
    while True:
        if len(byte_list) <= 1:
            break
        pairs = set(zip(byte_list, byte_list[1:]))
        if not pairs:
            break

        # find pair in sequence with lowest rank in merge dict
        min_rank = float("inf")
        choose = None
        for pair in pairs:
            rank = merge_ranks.get(pair, float("inf"))
            if rank < min_rank:
                min_rank = rank
                choose = pair
        if choose is None:
            break

        # replace chosen pair with new bytes
        merged_bytes = choose[0] + choose[1]
        new_byte_list = []
        idx = 0
        while idx < len(byte_list):
            if idx < len(byte_list) - 1 and byte_list[idx] == choose[0] and byte_list[idx + 1] == choose[1]:
                new_byte_list.append(merged_bytes)
                idx += 2
            else:
                new_byte_list.append(byte_list[idx])
                idx += 1
        byte_list = new_byte_list

    return tuple(reverse_vocab[b] for b in byte_list)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        bpe_merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.reverse_vocab = {token: idx for idx, token in vocab.items()}
        self.bpe_merges = bpe_merges
        self.merge_ranks = {pair: i for i, pair in enumerate(bpe_merges)}
        self.special_tokens = special_tokens or []

        # special token regex pattern
        if len(self.special_tokens) == 0:
            self.special_pattern = None
        else:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            special_rule = "|".join(map(regex.escape, sorted_special))
            self.special_pattern = regex.compile(f"({special_rule})")

        # LRU cache for bpe merge proces
        merge_ranks = self.merge_ranks
        reverse_vocab = self.reverse_vocab

        @lru_cache(maxsize=4096)
        def bpe_merge_with_cache(raw_bytes: bytes) -> tuple[int, ...]:
            return _bpe_merge(raw_bytes, reverse_vocab, merge_ranks)

        self.merge_cached = bpe_merge_with_cache

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs.

        Pre-tokenizes the input with the GPT-2 regex, applies BPE merges in
        the order specified by ``bpe_merges``, and looks up each resulting
        token in the vocabulary. Special tokens are preserved as single tokens.
        """
        if not text:
            return []

        # split by special tokens
        if self.special_pattern is not None:
            chunks = self.special_pattern.split(text)
        else:
            chunks = [text]

        encoded_ids = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                encoded_ids.append(self.reverse_vocab[chunk.encode("utf-8")])
            else:
                for word in GPT2_SPLIT_PATTERN_RE.findall(chunk):
                    if not word:
                        continue
                    ids = self._apply_bpe_merge(word.encode("utf-8"))
                    encoded_ids.extend(ids)

        return encoded_ids

    def _apply_bpe_merge(self, raw_bytes: bytes) -> tuple[int, ...]:
        return self.merge_cached(raw_bytes)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token IDs back into a string.

        Maps each ID to its byte representation via the vocabulary,
        concatenates all bytes, and decodes the result as UTF-8.
        """

        out = []
        for tid in ids:
            if tid not in self.vocab:
                raise ValueError(f"Token ID {tid} not found in vocab dict")
            out.append(self.vocab[tid])
        return b"".join(out).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings, yielding token IDs one at a time.

        Processes the input lazily so that large inputs (such as file objects)
        are not loaded entirely into memory.
        """
        for text in iterable:
            yield from self.encode(text)
