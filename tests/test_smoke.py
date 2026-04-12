"""Smoke test to verify package is importable."""


def test_import():
    import pydantic_mas

    assert pydantic_mas is not None
