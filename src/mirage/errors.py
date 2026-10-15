class MirageError(Exception):
    """Base class for errors the CLI reports cleanly instead of as a traceback."""


class ConfigError(MirageError):
    pass


class DataError(MirageError):
    pass


class MissingDependency(MirageError):
    def __init__(self, package: str, extra: str = "train"):
        super().__init__(
            f"'{package}' is not installed. This step needs the GPU-side extras: "
            f"pip install -e \".[{extra}]\""
        )
