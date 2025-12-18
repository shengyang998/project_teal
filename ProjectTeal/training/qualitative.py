"""Qualitative evaluation manifest helpers.

The evaluation plan calls for a curated set of scenarios (backlit, mixed
light, bright windows, city lights, foliage, fine patterns). This module
loads a manifest describing those captures and enforces coverage so
regressions can be tracked alongside quantitative metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Sequence

DEFAULT_REQUIRED_TAGS: tuple[str, ...] = (
    "backlit",
    "mixed_light",
    "bright_windows",
    "city_lights",
    "foliage",
    "fine_patterns",
)


@dataclass(frozen=True)
class QualitativeSample:
    """Metadata for a single qualitative capture."""

    capture_id: str
    path: Path
    tags: List[str]
    notes: str | None = None

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


@dataclass
class QualitativeSet:
    """A collection of qualitative captures with coverage checks."""

    samples: List[QualitativeSample]

    def ensure_coverage(self, required_tags: Sequence[str] = DEFAULT_REQUIRED_TAGS) -> None:
        missing = [tag for tag in required_tags if not any(s.has_tag(tag) for s in self.samples)]
        if missing:
            raise ValueError(f"missing required tags: {', '.join(missing)}")

    def filter_by_tag(self, tag: str) -> "QualitativeSet":
        return QualitativeSet([sample for sample in self.samples if sample.has_tag(tag)])

    def tag_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sample in self.samples:
            for tag in sample.tags:
                counts[tag] = counts.get(tag, 0) + 1
        return counts


def _load_manifest_dict(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_samples(manifest: dict, manifest_dir: Path) -> List[QualitativeSample]:
    raw_samples = manifest.get("captures")
    if raw_samples is None:
        raise ValueError("manifest missing 'captures' list")
    samples: List[QualitativeSample] = []
    for entry in raw_samples:
        capture_id = entry.get("id")
        relative_path = entry.get("path")
        tags = entry.get("tags")
        notes = entry.get("notes")
        if not capture_id or not isinstance(capture_id, str):
            raise ValueError("each capture requires a non-empty string 'id'")
        if not relative_path or not isinstance(relative_path, str):
            raise ValueError("each capture requires a non-empty string 'path'")
        if not tags or not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ValueError("each capture requires a non-empty list of string 'tags'")
        samples.append(
            QualitativeSample(
                capture_id=capture_id,
                path=(manifest_dir / relative_path).resolve(),
                tags=tags,
                notes=notes,
            )
        )
    return samples


def load_qualitative_manifest(
    manifest_path: str | Path, *, required_tags: Sequence[str] = DEFAULT_REQUIRED_TAGS
) -> QualitativeSet:
    """Load a qualitative manifest and enforce required coverage."""

    path = Path(manifest_path)
    manifest = _load_manifest_dict(path)
    samples = _parse_samples(manifest, path.parent)
    qualitative_set = QualitativeSet(samples)
    qualitative_set.ensure_coverage(required_tags)
    return qualitative_set
