def __getattr__(name: str):
    if name == "run_multizones_training":
        from b2b.training.multizones import run_multizones_training

        return run_multizones_training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_multizones_training"]
