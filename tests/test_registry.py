from src.registry import Registry


def test_registry_registers_and_loads_objects():
    registry = Registry("test")

    @registry.register("sample")
    class Sample:
        pass

    assert "sample" in registry
    assert registry.get("sample") is Sample


def test_registry_rejects_duplicates():
    registry = Registry("test")
    registry.register("sample")(object)

    try:
        registry.register("sample")(object)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate registration to fail.")
