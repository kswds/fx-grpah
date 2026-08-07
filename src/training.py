try:
    from .training_core import *  # type: ignore  # noqa: F401,F403
except ImportError:
    from training_core import *  # type: ignore  # noqa: F401,F403
