"""
    logs / logger.py
    ----------------
    Lightweight wrapper around python's standard-library
    ``logging`` module. This module provides a small,
    project-oriented logging API while keeping the underlying
    ``logging.Logger`` available for general use.
"""
import logging
import sys
import threading

from typing import Any, Literal
from logs.formatters import ConsoleFormatter
from logs.handlers import ConsoleHandler
from logs.levels import Level




Profile = Literal["runtime", "debug"]

class Logger:
    """
    A small light wrapper around :class:`logging.Logger` for
    console logging. Two output profiles are supported:

        - ``"runtime"`` * Compact output skips any stacking
        - ``"debug"``   
            * Includes the source filename, line number and 
              calling function to provide additional information
              during development.
    
    The wrapper passes ``stacklevel=2`` to the underlying logger
    methods. This causes the logging system to report to the 
    caller of this wrapper rather than the wrapper method instead.
    """
    _lock = threading.RLock()

    def __init__(self,
        name: str = "logger", level: Level = Level.INFO,
        profile: Profile = "runtime"
    ):
        """
            Initialize Logger Instance
        """
        self.name = name
        
        self._logger = logging.getLogger(name)
        self._logger.propagate = False

        self._configure(level, profile)

    
    def _configure(self, level: Level, profile: Profile) -> None:
        """
        Configure the underlying logger and its console handler. A 
        :class;`ConsoleHandler` is created only when the named logger 
        does not already have one. Existing console handlers are 
        instead reused.
        """
        with self._lock:
            self._logger.setLevel(int(level))
        
        handler: ConsoleHandler | None = None
        for h in self._logger.handlers:
            
            if isinstance(h, ConsoleHandler):
                handler = h
                break
        
        if handler is None:
            handler = ConsoleHandler()
            self._logger.addHandler(handler)
        
        handler.setLevel(int(level))
        handler.setFormatter(ConsoleFormatter(profile))
    

    @property
    def level(self) -> int:
        """ Return the current minimum logging level """
        return self._logger.level
    
    @level.setter
    def level(self, value: Level | int) -> None:
        """ Sets the minimum logging level """
        self._logger.setLevel(int(value))
        
        for handler in self._logger.handlers:
            
            if isinstance(handler, ConsoleHandler):
                handler.setLevel(int(value))
    

    def set_level(self, value: Level | int) -> None:
        """
        Set the minimum logging level. This is equivalent to 
        assign to :attr:`level`.
        """
        self.level = value
    
    
    def disable(self) -> None:
        """Disable all normal log output for this logger."""
        self.level = logging.CRITICAL + 1
    
    def enable(self, level: Level = Level.INFO) -> None:
        """
        Re-enable logging at the specified minimum severity 
        level.
        -----
        Args:
        level: Minimum severity level to emit.
        """
        self.level = level
    


    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """ Log a DEBUG-level message """
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg, *args, stacklevel=2, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """ Log a INFO-level message """
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(msg, *args, stacklevel=2, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """ Log a WARNING-level message """
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(msg, *args, stacklevel=2, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """ Log a ERROR-level message """
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.error(msg, *args, stacklevel=2, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """ Log a CRITICAL-level message """
        if self._logger.isEnabledFor(logging.CRITICAL):
            self._logger.critical(msg, *args, stacklevel=2, **kwargs)

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """
        Log an ERROR-level message with exception information. This 
        method should normally be called from an exception handler.
        """
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.exception(msg, *args, stacklevel=2, **kwargs)
    

    def get_logger(self) -> logging.Logger:
        """
        Return the underlying :class:`logging.Logger`. This provides 
        access to the full standard-library logging API when functionality 
        not exposed by this wrapper is required. 
        """
        return self._logger





def get_logger(
    name: str = "logger", level: Level = Level.INFO, profile: Profile = "runtime",
) -> Logger: return Logger(name, level, profile)
