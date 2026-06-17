"""Entry point — run with: python run.py"""
import sys


def _acquire_single_instance() -> object:
    """
    Returns a Win32 mutex handle that must be kept alive for the process lifetime.
    Exits immediately (silently) if another Iris instance already holds the mutex.
    """
    import ctypes
    _CreateMutex = ctypes.windll.kernel32.CreateMutexW
    _GetLastError = ctypes.windll.kernel32.GetLastError
    ERROR_ALREADY_EXISTS = 183

    handle = _CreateMutex(None, True, "Global\\IrisSingleInstance")
    if _GetLastError() == ERROR_ALREADY_EXISTS:
        sys.exit(0)
    return handle  # keep reference alive — GC must not collect this


if sys.platform == "win32":
    _mutex = _acquire_single_instance()

from app.src.UIX.main import main

if __name__ == "__main__":
    main()
