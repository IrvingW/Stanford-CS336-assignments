import os
from collections import Counter


Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]
PretokenCounts = Counter[Pretoken]
PairCounts = Counter[Pair]


def _initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Create the initial byte-level vocabulary.

    The vocabulary must contain all 256 single-byte tokens and each unique
    special token. Token IDs should be deterministic so later merges can be
    appended consistently.
    """
    raise NotImplementedError


def _count_pretokens(
    input_path: str | os.PathLike,
    special_tokens: list[str],
) -> PretokenCounts:
    """Read the corpus and count its pre-tokenized byte sequences.

    Special tokens must act as hard boundaries and must not contribute to the
    pre-token counts. Ordinary text should be split with the GPT-2-style regex
    from the assignment, encoded as UTF-8, and represented as tuples of
    single-byte tokens.
    """
    raise NotImplementedError


def _count_pairs(pretoken_counts: PretokenCounts) -> PairCounts:
    """Count adjacent token pairs across all pre-tokens.

    Each pair's contribution must be weighted by the frequency of the
    pre-token that contains it. Pairs must never cross pre-token boundaries.
    """
    raise NotImplementedError


def _select_best_pair(pair_counts: PairCounts) -> Pair | None:
    """Select the pair to merge next.

    Choose the pair with the highest frequency. When frequencies are tied,
    choose the lexicographically greatest pair, as required by the assignment.
    Return ``None`` when no mergeable pair remains.
    """
    raise NotImplementedError


def _merge_pair_in_sequence(tokens: Pretoken, pair: Pair) -> Pretoken:
    """Merge every non-overlapping occurrence of one pair in one pre-token.

    This helper operates on a single pre-token only. It should preserve the
    order of all unaffected tokens and return a new token tuple.
    """
    raise NotImplementedError


def _apply_merge(pretoken_counts: PretokenCounts, pair: Pair) -> PretokenCounts:
    """Apply one selected merge to the complete pre-token frequency table.

    Preserve each pre-token's corpus frequency. If multiple old pre-tokens
    become the same token tuple after merging, combine their frequencies.
    """
    raise NotImplementedError


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer and return its vocabulary and merges.

    Orchestrate vocabulary initialization, pre-token counting, pair counting,
    deterministic pair selection, and repeated merging until ``vocab_size`` is
    reached or no mergeable pair remains.

    Args:
        input_path: Path to the text corpus used for BPE training.
        vocab_size: Maximum final vocabulary size, including byte tokens,
            special tokens, and tokens created by merges.
        special_tokens: Strings that are added to the vocabulary and treated
            as hard boundaries during training.

    Returns:
        A pair containing the token-ID-to-bytes vocabulary and the merge pairs
        in creation order.
    """
    raise NotImplementedError
