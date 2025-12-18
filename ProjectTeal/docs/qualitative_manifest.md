# Qualitative Evaluation Manifest

The evaluation harness includes a curated qualitative sweep to stress the 48MP linear RGB pipeline in scenarios that commonly
trigger local tone-mapping artifacts. Manifests are JSON files consumed by `training.qualitative.load_qualitative_manifest`,
which enforces coverage across the required tags:

- `backlit`
- `mixed_light`
- `bright_windows`
- `city_lights`
- `foliage`
- `fine_patterns`

## Format

```json
{
  "captures": [
    {
      "id": "backlit_sunroom",
      "path": "qualitative/backlit_sunroom.dng",
      "tags": ["backlit", "bright_windows", "interior"],
      "notes": "Sunroom with strong window backlight and shadowed furniture."
    }
  ]
}
```

- `id`: Stable identifier for the capture.
- `path`: Relative path to the DNG/JPEG asset from the manifest location.
- `tags`: One or more tags describing the scenario. At least one entry must exist for each required tag above.
- `notes` (optional): Human-readable context to guide regression reviews.

## Sample coverage

A starter manifest lives in `training/tests/fixtures/qualitative_manifest.json` to document the expected coverage set and to
support automated validation.
