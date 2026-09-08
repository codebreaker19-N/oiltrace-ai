# M2 — Satellite & Geospatial Handoff

## Input

Refined Deep-SAR oil spill dataset.

Image format:
- PNG
- 256 × 256 pixels

Mask format:
- RGB PNG
- Background = 0
- Oil spill = 255

## Processing

The satellite module performs:

1. PNG image loading
2. Image normalization to 0–1
3. Mask conversion to binary 0/1
4. Spill contour extraction
5. Spill geometry extraction
6. Shape feature extraction
7. Standardized JSON output

## Output

The main output is:

`data/processed/spill_output.json`

Structure:

```json
{
  "image_id": "palsar_0",
  "geometry": {},
  "geometry_summary": {},
  "features": {}
}