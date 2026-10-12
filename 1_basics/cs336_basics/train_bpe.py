import os
import regex as re
from collections import Counter


Token = bytes
Pretoken = tuple[Token, ...]
Pair = tuple[Token, Token]
PretokenCounts = Counter[Pretoken]
PairCounts = Counter[tuple[Token, Token]]

# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """Create the initial byte-level vocabulary.

    The vocabulary must contain all 256 single-byte tokens and each unique
    special token. Token IDs should be deterministic so later merges can be
    appended consistently.
    """
    # 0~255 char
    vocab_dict = {i: bytes([i]) for i in range(256)}
    # special tokens start from the index of 256
    new_idx = 256
    existing = set(vocab_dict.values())
    for special in special_tokens:
        sp_bytes = special.encode('utf-8')
        if sp_bytes not in existing:
            vocab_dict[new_idx] = sp_bytes
            existing.add(sp_bytes)
            new_idx += 1
            
    return vocab_dict


def _pre_tokenize(
    input_text: str,
    special_tokens: list[str],
) -> PretokenCounts:
    """Read the corpus and count its pre-tokenized byte sequences.

    Special tokens must act as hard boundaries and must not contribute to the
    pre-token counts. Ordinary text should be split with the GPT-2-style regex
    from the assignment, encoded as UTF-8, and represented as tuples of
    single-byte tokens.
    """
    # split using special tokens like <EOF>
    if len(special_tokens) > 0:
        chunks = re.split('|'.join(map(re.escape,special_tokens)), input_text)
    else:
        chunks = [input_text]
    
    result :PretokenCounts = Counter()
    for chunk in chunks:
        for word in re.findall(GPT2_SPLIT_PATTERN, chunk):
            pre_token = tuple(bytes([b]) for b in word.encode("utf-8"))
            result[pre_token] += 1
            
    return result


def _count_pairs(pretoken_counts: PretokenCounts) -> PairCounts:
    """Count adjacent token pairs across all pre-tokens.

    Each pair's contribution must be weighted by the frequency of the
    pre-token that contains it. Pairs must never cross pre-token boundaries.
    """
    pair_counts = Counter()
    for pre_token, freq in pretoken_counts.items():
        for i in range(len(pre_token) - 1):
            pair_counts[pre_token[i], pre_token[i + 1 ]] += freq
            
    return pair_counts
        


def _select_best_pair(pair_counts: PairCounts) -> Pair | None:
    """Select the pair to merge next.

    Choose the pair with the highest frequency. When frequencies are tied,
    choose the lexicographically greatest pair, as required by the assignment.
    Return ``None`` when no mergeable pair remains.
    """
    # all pre-tokens are already single token
    if not pair_counts:
        return None
   
    # comment: Easy way to write 
    # max_freq = max(pair_counts.values())
    # candidates = [k for k, v in pair_counts.items() if v == max_freq]
    # select = max(candidates)
    # return select
    return max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _merge_pair(seq: Pretoken, pair: Pair) -> Pretoken:
    """Merge every non-overlapping occurrence of one pair in one pre-token.

    This helper operates on a single pre-token only. It should preserve the
    order of all unaffected tokens and return a new token tuple.
    """
    result = []
    new_token = pair[0] + pair[1]
    idx = 0
    while idx < len(seq):
        if idx < len(seq) - 1 and (seq[idx], seq[idx + 1]) == pair:
            result.append(new_token)
            idx += 2
        else:
            result.append(seq[idx])
            idx += 1
    return tuple(result)


def _apply_merge(pretoken_counts: PretokenCounts, merge_pair: Pair, pair_counts: PairCounts) -> tuple[PretokenCounts, PairCounts]:
    """Apply one selected merge to the complete pre-token frequency table.

    Preserve each pre-token's corpus frequency. If multiple old pre-tokens
    become the same token tuple after merging, combine their frequencies.
    """
    affected_pretoken = Counter()
    for pre_token, freq in pretoken_counts.items():
        if any((pre_token[i], pre_token[i+1]) == merge_pair for i in range(len(pre_token) - 1)):
            affected_pretoken[pre_token] = freq

    merged_pretoken = Counter()
    for seq, freq in affected_pretoken.items():
        del pretoken_counts[seq]
        for pair in [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]:
            pair_counts[pair] -= freq
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
        new_seq = _merge_pair(seq, merge_pair)
        pretoken_counts[new_seq] += freq
        merged_pretoken[new_seq] += freq

    new_pair_counts = _count_pairs(merged_pretoken)            
    pair_counts.update(new_pair_counts)
    
    return pretoken_counts, pair_counts


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
    assert vocab_size >= 256 + len(special_tokens)
   
    # initialize vocab dict 
    vocab_dict :dict[int, bytes] = _initialize_vocab(special_tokens=special_tokens)
    num_merges = vocab_size - len(vocab_dict)
    new_token_id = len(vocab_dict)
   
    with open(input_path, encoding="utf-8") as f:
        input_text = f.read()
    
    pre_token_counts: PretokenCounts = _pre_tokenize(input_text, special_tokens)
    pair_counts: PairCounts = _count_pairs(pre_token_counts)
    merges = []
    while len(merges) < num_merges:
       select_pair: Pair|None = _select_best_pair(pair_counts)
       if not select_pair:
           # could not do merge any more
           break
       vocab_dict[new_token_id] = select_pair[0] + select_pair[1]
       merges.append(select_pair)
       new_token_id += 1
       
       # update counts map
       pre_token_counts, pair_counts = _apply_merge(pretoken_counts=pre_token_counts, merge_pair=select_pair, pair_counts=pair_counts)
       
    return vocab_dict, merges
