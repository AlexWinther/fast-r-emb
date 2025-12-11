import hashlib
import os
import torch
import pickle
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from tqdm import tqdm

from typing import Iterable, List, Optional, Sequence, Dict, Protocol, runtime_checkable

# MODEL_ID = "Salesforce/codet5p-220m"  # CodeT5+ 220M
MODEL_ID = "Salesforce/codet5p-110m-embedding"


@runtime_checkable
class Embedder(Protocol):
    def embed_test_cases(self, test_cases: Sequence[str]) -> torch.Tensor: ...


def resolve_device(arg_device: str | None) -> str:
    if arg_device is None or arg_device == "auto":
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
    if arg_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return arg_device


def batched(iterable: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


class CodeT5PlusEmbedder(Embedder):
    """Transformers-backed embedder that satisfies the Embedder protocol."""

    def __init__(
        self,
        model_dir: str = MODEL_ID,
        dataset: str = "grep_v3",
        cache_file: str = "codet5_embeddings.pkl",
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.model_dir = model_dir
        self.dataset = dataset
        _, version = dataset.split("_")
        self.java_flag = version == "v0"
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.device = resolve_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(
            self.device
        )
        self.model.eval()

        self.cache = self._load_cache()

    def _get_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, torch.Tensor]:
        cache_path = "data/embedding/{}/{}".format(self.dataset, self.cache_file)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        cache_dir = "data/embedding/{}/".format(self.dataset)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        cache_path = cache_dir + self.cache_file
        with open(cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def _embed_batch(self, texts: List[str]) -> List[torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512 if not self.java_flag else 2048,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model(**inputs)

        return embeddings.cpu()

    def embed_test_cases(self, test_cases: Sequence[str]) -> torch.Tensor:
        new_cache = {}
        texts_to_embed = []
        text_hashes = []

        for text in test_cases:
            h = self._get_hash(text)
            text_hashes.append(h)
            if h not in self.cache:
                texts_to_embed.append(text)

        if texts_to_embed:
            for batch in tqdm(
                batched(texts_to_embed, self.batch_size),
                total=len(texts_to_embed) // self.batch_size + 1,
            ):
                embeddings = self._embed_batch(batch)
                for text, emb in zip(batch, embeddings):
                    h = self._get_hash(text)
                    self.cache[h] = emb

        # Clean cache to only keep relevant hashes
        for h in text_hashes:
            new_cache[h] = self.cache[h]

        self.cache = new_cache
        self._save_cache()

        # Return embeddings in input order
        return torch.stack([self.cache[h] for h in text_hashes])
