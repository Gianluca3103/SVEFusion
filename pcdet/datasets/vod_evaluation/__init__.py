"""VoD evaluation helpers.

The standalone Evaluation helper needs the optional vod-tudelft devkit.
Keep it lazy so the bundled KITTI evaluator works without that package.
"""

__all__ = ["Evaluation"]


def __getattr__(name):
    if name == "Evaluation":
        from .evaluate import Evaluation

        return Evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
