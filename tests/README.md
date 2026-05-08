# Test Suite

The repository uses `pytest` with deterministic seeds, structured markers, and coverage enforcement.

Current validated categories:

- unit tests
- integration tests
- training pipeline tests
- dataset tests
- hook validation tests
- visualization tests
- dashboard tests
- stress tests
- reproducibility tests

Run the full suite:

```bash
pytest -q
```

Coverage is enforced through `pyproject.toml`.
