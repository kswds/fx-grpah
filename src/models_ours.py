try:
    from .models.arc_fx import *  # type: ignore  # noqa: F401,F403
except ImportError:
    from models.arc_fx import *  # type: ignore  # noqa: F401,F403
