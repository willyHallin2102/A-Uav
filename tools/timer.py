"""
    tools / timer.py
    ----------------
    Lightweight high-resolution timer for measuring execution time. It uses 
    ``time.perf_counter()`` for elapsed-time measurements and supports

        - context-manager usage
        - decorator usage
        - manual start / stop
        - restart / reset
        - seconds, milliseconds, microseconds, nanoseconds
        - optionally: ``automatic formatting``
        - optionally: output through ``show=True``

"""
from __future__ import annotations

from contextlib import ContextDecorator
from functools import wraps
from typing import Any, Callable, TypeVar

import time



F = TypeVar("F", bound = Callable[..., Any])

class Timer(ContextDecorator):
    """
    Measure the time elapsed of a block of code or a function from call.
    Example:
    --------
    Context Manager::

        with Timer("timer-name", show=True) as t:
            f()
        
        print(t.elapsed_ms)
        print(t.elapsed_us)
        print(t.elapsed_ns)
    
    Decorator::

        @Timer("Expensive Operation", show=True)
        def expensive_operation():
            ···
    
    Manual Usage::

        timer = Timer("Operation").start()
        ···
        timer.stop()
        print(timer.elapsed)
    """
    __slots__ = (
        "name", "show", "auto_format", "_clock", "_start", "_elapsed"
    )

    def __init__(self,
        name: str | None = None, *, show: bool = True, auto_format: bool = True, 
    ):
        """
            Initialize Timer Instance
        """
        self.name, self.show, self.auto_format = name, show, auto_format
        
        self._clock = time.perf_counter
        self._start: float | None = None
        self._elapsed: float | None = None
    
    
    # ============================================================
    #       Context Manager
    # ============================================================

    def __enter__(self) -> Timer:
        return self.start()
    

    def __exit__(self,
        exc_type: Type[BaseException] | None, exc_value: BaseException | None,
        traceback: Any
    ) -> bool:
        """
        """
        self.stop()

        if self.show:
            print(self)
        
        return False
    

    # ============================================================
    #       Manual Control
    # ============================================================
    
    def start(self) -> Timer:
        """
        Start or restart the timer
        """
        self._start = self._clock()
        self._elapsed = None

        return self
    
    
    def stop(self) -> Timer:
        """
        Stop the timer and store the elapsed duration.
        """
        if self._start is None:
            return self
        
        self._elapsed = self._clock() - self._start
        self._start = None

        return self
    

    def reset(self) -> Timer:
        """
        Reset teh timer without starting it.
        """
        self._start, self._elapsed = None, None
        return self
    

    def restart(self) -> Timer:
        """
        Reset and immediately start the timer.
        """
        return self.reset().start()
    

    # ============================================================
    #       Measurements
    # ============================================================

    @property
    def running(self) -> bool:
        """ Whether the timer is currently running. """
        return self._start is not None
    

    @property
    def elapsed(self) -> float:
        """
        Elapsed time in seconds. Specifically, returns 0.0 whenever the
        timer has not completed its measurements.
        """
        if self._elapsed is not None: 
            return self._elapsed 
        
        if self._start is not None:
            return self._clock() - self._start
        
        return 0.0
    

    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1_000.0
    
    
    @property
    def elapsed_us(self) -> float:
        """Elapsed time in microseconds."""
        return self.elapsed * 1_000_000.0
    
    
    @property
    def elapsed_ns(self) -> int:
        """Elapsed time in nanoseconds."""
        return int(self.elapsed * 1_000_000_000)
    

    # ============================================================
    #       Formatting
    # ============================================================

    def result(self) -> str:
        """
        Return a readable representation of the duration
        """
        if self._elapsed is None:
            if self._start is None:
                return "Timer is not started."
        
        elapsed = self._elapsed
        if not self.auto_format:
            value = f"{elapsed:.6f} s"
        
        elif elapsed >= 1.0:
            value = f"{elapsed:.3f} s"
        
        elif elapsed >= 1e-3:
            value = f"{elapsed * 1_000:.3f} ms"
        
        elif elapsed >= 1e-6:
            value = f"{elapsed * 1_000_000:.3f} µs"
        
        else:
            value = f"{elapsed * 1_000_000_000:.3f} ns"
        
        return f"{self.name}: {value}" if self.name else value
    

    def __str__(self) -> str:
        return self.result()
    

    def __repr__(self) -> str:
        values = []

        if self.name:
            values.append(f"name={self.name!r}")
        
        if self._elapsed is not None:
            values.append(f"elapsed={self._elapsed:.6f}s")
        
        elif self.running:
            values.append("running=True")
        
        return f"Timer({', '.join(values)})"



def timer(
    name: str | None = None, *, show: bool = True, auto_format: bool = True
) -> Timer:
    """Convenience factory for :class:`Timer`."""
    return Timer(name=name, show=show, auto_format=auto_format)



def time_it(
    f: F | None = None, *, name: str | None = None, show: bool = True
) -> Callable[[F], F] | F:
    """
    Decorator that measures function execution time. Supports
    both:: @time_it and:: @time_it(name="database query")
    """ 
    def decorator(f: F) -> F:
        timer_name = name or f.__qualname__
        
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any: 
            with Timer(timer_name, show=show):
                return f(*args, **kwargs) 
        
        return wrapper
    
    if f is not None:
        return decorator(f)
    
    return decorator
