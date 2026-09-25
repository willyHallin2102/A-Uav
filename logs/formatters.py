"""
    logs / formatters.py
    --------------------
    Console formatting utilities for the logging package.
    The module provides the formatters to the logging 
    system for the project. It provides:

        - ANSI escape-code constants for terminal styling.
        - Severity - specific colors and icons.
        - ``ConsoleFormatter`` for compact, readable log output.
    
    ``ConsoleFormatter`` supports two distinct profiles each
    with their own individual output terminal.

        * ``"runtime"`` --  Minimal output applied for execution
        * ``"debug"``   -- Includes filename, line number, and 
                           calling function for simplifying
                           debugging purposes.
"""

import logging
from logs.levels import Level


# ============================================================================
#       ANSI Terminal Customization
# ============================================================================

class ANSI:
    """
    ANSI escape sequences used to customize terminal output. The
    constants are grouped into three namespaces, for various features:

        * ``ANSI.FG``:      Foreground / Text Colors
        * ``ANSI.BG``:      Background Colors
        * ``ANSI.Text``:    Text formatting attributes such as bold, dim,
                            and underline.
        * ``ANSI.RESET``:   Resets all active terminal formatting.
    """
    RESET = "\033[0m"

    class FG:
        """ ANSI foreground (text) color escape sequences """
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
    
    class BG:
        """ ANSI background color escape sequences """
        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
    
    class Text:
        """ ANSI text-style escape sequences """
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        STRIKE = "\033[9m"



# ============================================================================
#       Level Representation for the Severity 
# ============================================================================

_COLOR: dict[Level, str] = {
    Level.DEBUG:    ANSI.Text.DIM + ANSI.FG.GREEN,
    Level.INFO:     ANSI.Text.ITALIC + ANSI.FG.CYAN,
    Level.WARNING:  ANSI.Text.DIM + ANSI.FG.YELLOW,
    Level.ERROR:    ANSI.Text.BOLD + ANSI.FG.RED,
    Level.CRITICAL: (
        ANSI.Text.BOLD + ANSI.Text.UNDERLINE + ANSI.BG.RED + ANSI.FG.WHITE
    )
}

_ICON: dict[Level, str] = {
    Level.DEBUG:    "🔧", 
    Level.INFO:     "ℹ️ ",
    Level.WARNING:  "⚠️ ",
    Level.ERROR:    "❌",
    Level.CRITICAL: "💥",
}



# ============================================================================
#       Level Representation for the Severity 
# ============================================================================

class ConsoleFormatter(logging.Formatter):
    """
    FOrmat log record for readable console output. The formatter always 
    applies ANSI styling to the severity level and displays an icon 
    corresponding to the log level. Two formatting profiles supported:

        ``"runtime"``   : HH:MM:SS | app | INFO | message
        ---------------------------------------------------------------
        ``"debug"``     : HH:MM:SS | app INFO | app.py:lineno | message
        ---------------------------------------------------------------
    
    The debug profile requires caller information from the
    :class`logging.LogRecord` and therefore produces more detailed 
    output:
    -----
    Args:
    profile: Formatting profile to use. must be either `"runtime"` or \
        `"debug"` else it raise an error.
    -------
    Raises:
    ValueError: If an unsupported profile is provided.
    """
    __slots__ = ("_debug", "_style_cache")


    def __init__(self, profile: str="runtime"):
        """
            Initialize Console-Formatter Instance
        """
        if profile not in ("runtime", "debug"):
            raise ValueError(f"Unknown profile: {profile!r}")
        
        super().__init__(datefmt="%H:%M:%S")
        self._debug = profile == "debug"

        self._style_cache: dict[int, tuple[str, str]] = {}


    def _resolve(self, levelno: int) -> tuple[str, str]:
        """
        Resolves the ANSI style and icon for a logging level. Results
        are cached by numerical logging level because the same levels 
        are typically formatted repeatedly.
        -----
        Args:
        levelno: Numeric logging level from :attr:`logging.LogRecord.levelno`
        --------
        Returns:
        tuple[str, str] of `(color,icon)` pair. Unknown logging levels \
            receive no color and the ``❓`` fallback icon.
        """
        cached = self._style_cache.get(levelno)
        
        if cached is not None:
            return cached
        
        try:
            level = Level(levelno)
            style = (_COLOR.get(level, ""), _ICON.get(level, "❓"))
        
        except ValueError:
            style = ("", "❓")
        
        self._style_cache[levelno] = style
        return style
    
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a single logging record. The resulting format contains: 
            * timestamp 
            * logger name 
            * severity icon 
            * colored severity name
            * optional source location 
            * log message 
            * exception traceback, when present 
            
        -----
        Args:
        record: The :class:`logging.LogRecord` being formatted. 
        --------
        Returns: 
            The fully formatted console message as a string.
        """ 
        record.asctime = self.formatTime(record, self.datefmt) 
        color, icon = self._resolve(record.levelno) 
        
        level = f"{color}{record.levelname:<8}{ANSI.RESET}" 
        message = record.getMessage()
        
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        
        if self._debug:
            where = f"{record.filename}: {record.lineno} in {record.funcName}"
            return (
                f"{record.asctime} | "
                f"{record.name} | "
                f"{icon} {level} | "
                f"{where} | "
                f"{message}"
            )
        
        return (
            f"{record.asctime} | "
            f"{record.name} | "
            f"{icon} {level} | "
            f"{message}"
        )
