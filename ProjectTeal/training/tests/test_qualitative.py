import json
from pathlib import Path

import pytest

from training.qualitative import (
    DEFAULT_REQUIRED_TAGS,
    QualitativeSet,
    load_qualitative_manifest,
)


FIXTURE_MANIFEST = Path(__file__).parent / "fixtures" / "qualitative_manifest.json"


def test_manifest_loads_and_enforces_coverage(tmp_path):
    manifest = load_qualitative_manifest(FIXTURE_MANIFEST)
    assert isinstance(manifest, QualitativeSet)
    assert len(manifest.samples) == 6

    tag_counts = manifest.tag_counts()
    for tag in DEFAULT_REQUIRED_TAGS:
        assert tag in tag_counts
        assert tag_counts[tag] >= 1

    backlit = manifest.filter_by_tag("backlit")
    assert all("backlit" in sample.tags for sample in backlit.samples)


def test_missing_required_tags_raise(tmp_path):
    # Drop the foliage tag to force coverage failure
    manifest_dict = json.loads(FIXTURE_MANIFEST.read_text())
    manifest_dict["captures"] = [
        {**entry, "tags": [t for t in entry["tags"] if t != "foliage"]}
        for entry in manifest_dict["captures"]
    ]
    broken_manifest = tmp_path / "broken_manifest.json"
    broken_manifest.write_text(json.dumps(manifest_dict))

    with pytest.raises(ValueError) as excinfo:
        load_qualitative_manifest(broken_manifest)

    assert "foliage" in str(excinfo.value)
