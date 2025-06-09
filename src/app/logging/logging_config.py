import logging
import os


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL")
    if not level:
        level = logging.INFO

    if level not in logging.getLevelNamesMapping():
        raise ValueError(f"Provider LOG_LEVEL '{level}' from env vars is not recognised")

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)

    main_logger = logging.getLogger("__main__")
    main_logger.setLevel(level)
    main_logger.addHandler(handler)

    module_logger = logging.getLogger("src")
    module_logger.setLevel(level)
    module_logger.addHandler(handler)
