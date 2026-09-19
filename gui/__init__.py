"""
GUI модуль для приложения сегментации тканей.

Импорт ленивый: `import gui.database` не тянет за собой PyQt/torch.
"""

__all__ = ['SegmentationApp', 'ModelManager']


def __getattr__(name):
    if name == 'SegmentationApp':
        from .mainwindow import SegmentationApp
        return SegmentationApp
    if name == 'ModelManager':
        from .model_manager import ModelManager
        return ModelManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
