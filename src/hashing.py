from typing import Callable, List, Set
import xxhash


def hash_family(i: int) -> Callable[[str], str]:
    def hashMember(x: str) -> str:
        return xxhash.xxh64(x, seed=37 * (2 * i + 1)).hexdigest()

    return hashMember


def min_hashing(hash_set: Set[int], num_hash_funcs: int) -> List[str]:
    hash_funcs = [hash_family(i) for i in range(num_hash_funcs)]

    tc_signature: List[str] = ["ffffffffffffffff" for _ in range(num_hash_funcs)]
    for _hash in hash_set:
        for i in range(num_hash_funcs):
            tc_hash = hash_funcs[i](str(_hash))
            tc_signature[i] = min(tc_hash, tc_signature[i])

    return tc_signature
