"""Tokenizer-filtered disjoint gendered name pools."""
import random


def single_token(tokenizer, word: str) -> int | None:
    """Return the unique token for a space-prefixed word, or reject it."""
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 and ids[0] != tokenizer.unk_token_id else None


def build_name_pools(tokenizer, candidates: dict, seed: int) -> dict:
    """Filter names and split each gender independently into held-out pools."""
    rng = random.Random(seed)
    pools = {split: {} for split in ("discovery", "validation")}
    all_names = set()
    for gender in ("male", "female"):
        accepted = []
        for name in candidates[gender]:
            token_id = single_token(tokenizer, name)
            if token_id is not None and name not in all_names:
                accepted.append({"name": name, "token_id": token_id})
                all_names.add(name)
        rng.shuffle(accepted)
        middle = len(accepted) // 2
        if middle < 3:
            raise ValueError(f"Need at least six single-token {gender} names")
        pools["discovery"][gender] = accepted[:middle]
        pools["validation"][gender] = accepted[middle:]
    ids = [{n["token_id"] for gender in pool.values() for n in gender} for pool in pools.values()]
    if ids[0] & ids[1]:
        raise ValueError("Discovery/validation names share token IDs")
    return pools
