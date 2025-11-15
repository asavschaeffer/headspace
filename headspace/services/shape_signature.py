"""Generate deterministic procedural shape signatures for chunks.

The front-end reconstructs planetary meshes from these signatures, ensuring
identical shapes across devices without recomputing heavy embedding transforms
client-side.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class ShapeSignatureConfig:
    detail: int = 32
    deformation_scale: float = 0.32
    smoothing_iterations: int = 2
    min_harmonics: int = 6
    max_harmonics: int = 16


class ShapeSignatureBuilder:
    """Builds deterministic shape signatures from embedding vectors."""

    def __init__(self, config: ShapeSignatureConfig | None = None):
        self.config = config or ShapeSignatureConfig()

    def build(self, embedding: Sequence[float], chunk_id: str | None = None, tags: Iterable[str] | None = None) -> Dict[str, Any]:
        if not embedding:
            return self._fallback_signature(tags)

        normalised = self._normalise_embedding(embedding)
        harmonic_count = self._determine_harmonic_count(len(normalised))
        segments = self._segment_values(normalised, harmonic_count)

        # Seed phases from a stable hash of the embedding + chunk id
        hash_source = f"{chunk_id or 'chunk'}:{len(embedding)}:{sum(embedding):.6f}".encode()
        hash_digest = hashlib.sha256(hash_source).digest()

        harmonics: List[Dict[str, float]] = []
        for index, values in enumerate(segments):
            if not values:
                values = [0.0]

            amplitude = float(statistics.fmean(values))
            variability = float(statistics.pstdev(values)) if len(values) > 1 else 0.0

            phase_seed = int.from_bytes(hash_digest[index:index + 4], "big", signed=False)
            phase = (phase_seed % (2_000_000)) / 1_000_000 * math.pi  # Range [0, 2π)

            harmonics.append(
                {
                    "frequency": index + 1,
                    "amplitude": amplitude,
                    "variability": variability,
                    "phase": phase,
                }
            )

        noise_profile = self._build_noise_profile(normalised)

        signature: Dict[str, Any] = {
            "version": 1,
            "type": "procedural",
            "detail": self.config.detail,
            "deformation_scale": self.config.deformation_scale,
            "smoothing": self.config.smoothing_iterations,
            "harmonics": harmonics,
            "noise": noise_profile,
            "hash": hash_digest.hex(),
            "texture": "crystalline",
            "scale": 3.4,
        }

        if tags:
            signature["tags"] = sorted(tags)

        return signature

    def _fallback_signature(self, tags: Iterable[str] | None) -> Dict[str, Any]:
        signature: Dict[str, Any] = {
            "version": 1,
            "type": "sphere",
            "detail": self.config.detail,
            "deformation_scale": 0.0,
            "harmonics": [],
            "noise": {},
            "texture": "smooth",
            "scale": 3.4,
        }
        if tags:
            signature["tags"] = sorted(tags)
        return signature

    def _normalise_embedding(self, embedding: Sequence[float]) -> List[float]:
        min_val = min(embedding)
        max_val = max(embedding)
        if math.isclose(max_val, min_val):
            return [0.0 for _ in embedding]
        return [2 * ((value - min_val) / (max_val - min_val)) - 1 for value in embedding]

    def _determine_harmonic_count(self, dimension: int) -> int:
        estimate = int(math.sqrt(max(dimension, 1)))
        return max(self.config.min_harmonics, min(self.config.max_harmonics, estimate))

    def _segment_values(self, values: Sequence[float], segments: int) -> List[List[float]]:
        step = max(len(values) / segments, 1)
        buckets: List[List[float]] = []
        for i in range(segments):
            start = int(round(i * step))
            end = int(round((i + 1) * step))
            buckets.append(list(values[start:end]))
        return buckets

    def _build_noise_profile(self, values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"levels": []}
        abs_values = [abs(v) for v in values]
        quarter = max(len(abs_values) // 4, 1)
        levels = [
            float(statistics.fmean(abs_values[:quarter])),
            float(statistics.fmean(abs_values[quarter: quarter * 2])),
            float(statistics.fmean(abs_values[quarter * 2: quarter * 3])),
            float(statistics.fmean(abs_values[quarter * 3:] or abs_values[-quarter:])),
        ]
        return {"levels": levels}
