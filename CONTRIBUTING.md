# Contribution & Review Standard

Changes to this repository should be small, reviewable and reproducible.

## Change expectations

1. Keep source inputs separate from generated outputs.
2. Document material assumptions, transformations and model/control changes.
3. Do not commit credentials, secrets, private data or licensed source data that cannot be redistributed.
4. Re-run the relevant analysis or application checks before merging.
5. Update documentation when behaviour, assumptions, data contracts or outputs change.
6. Prefer descriptive commits such as `feat:`, `fix:`, `docs:`, `test:` and `refactor:`.

## Review checklist

- Source provenance and licensing considered
- Data/model assumptions documented
- Reproducibility preserved
- No sensitive information committed
- Outputs and conclusions trace back to code or documented analysis
- Limitations remain explicit

For analytical repositories, published findings should not be silently replaced by results from a different sample period or methodology. Material methodology changes should be documented as a new revision.
