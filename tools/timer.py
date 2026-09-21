"""
    tools / timer.py
    ----------------
    Lightweight high-resolution timer for measuring execution time.
    The timer uses `time.perf_counter()`, which is designed for measuring
    elapsed time and is suitable for performance comparisons.
"""
from __future__ import annotations

from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, TypeVar

import time



F = TypeVar("F", bound=Callable[..., Any])

class Timer(ContextDecorator):
    """
    Measure the elapsed time of a block of code or function. The timer 
    can be used as a context manager or as a decorator.
    """
    def __init__(self,
        name: str | None = None, *, show: bool = False, auto_format: bool = True,
    ):
        """
            Initializing Clock Instance
        """
        self.clock, self.show, self.format = time.perf_counter, show, auto_format
        
        self.src: float | None = None
        self.dst: float | None = None

    
    def __enter__(self) -> Timer:
        """
        Start the timer and return then timer instance
        """
        self.src = self.clock()
        return self
    

    def __exit__(self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: Any
    ) -> None:
        """
        Stop the timer and optionally prints the result.
        """
        if self.src is None:
            return
        
        self._elapsed = self.clock() - self.src
        if self.show:
            print(self.result())
    

    def __str__(self) -> str:
        """ Return the formatted timing result """
        return self.result()
    

    def __repr__(self) -> str:
        """ Return a friendly representation of the timer. """
        name = f"name = {self.name!r}" if self.name else None
        elapsed = f"elapsed = {self.elapsed:.6f}s" if self._elapsed is not None \
            else ""
        
        values = ", ".join(value for value in (name, elapsed) if value)
        return f"Timer({values})"
    

    @property
    def elapsed(self) -> float:
        """ Return the elapsed time in seconds """
        return 0.0 if self._elapsed is None else self._elapsed 
    

    @property 
    def elapsed_ms(self) -> float:
        """Return the elapsed time in milliseconds.""" 
        return self.elapsed * 1_000.0 
    
    @property 
    def elapsed_us(self) -> float:
        """Return the elapsed time in microseconds."""
        return self.elapsed * 1_000_000.0
    
    @property
    def elapsed_ns(self) -> int:
        """Return the elapsed time in nanoseconds."""
        return int(self.elapsed * 1_000_000_000.0)
    

    def result(self) -> str:
        """
        Return a readable representation of the elapsed time. When `auto_format`
        is enabled, the unit is selected according to the measured duration.
        """
        if self._elapsed is None:
            return "Timer not started or not stopped"
        
        elapsed = self._elapsed
        if not self.format:
            time_str = f"{elapsed:.6f} s"
        
        elif elapsed >= 1.0:
            time_str = f"{elapsed:.3f} s"
        
        elif elapsed >= 1e-3:
            time_str = f"{elapsed * 1_000:.3f} ms"
        
        else:
            time_str = f"{elapsed * 1_000_000:.3f} µs" 
        
        prefix = f"{self.name}: " if self.name else "" 
        return f"{prefix}{time_str}"
    

    def reset(self) -> Timer:
        """
        Reset the timer without starting it.
        """
        self.src, self._elapsed = None, None
        return self
    

    def restart(self) -> Timer:
        """
        Restart the timer and immediately run it again
        """
        self.reset()
        return self.__enter__()



def timer(name: str | None = None, *, show: bool = True) -> Timer:
    """
    Create a d return a Timer instance. This is a convenient wrapper 
    around :class:`Timer`.
    """
    return Timer(name=name, show=show)


def time_it(f: F) -> F:
    """
    Decorator that measures and prints a function's execution time. The
    wrapped function's return value and arguments are unchanged.
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Timer(name=f.__name__, show=True):
            return f(*args, **kwargs)
    
    return wrapper
