import logging
import sys


class ConsoleHandler(logging.StreamHandler):
    """
    StreamHandler that always writes to the *current* sys.stdout.

    This matters for pytest's sys, contextlib.redirect_stdout, and
    interactive REPLs where sys.stdout is swapped. A plain StreamHandler
    captures sys.stdout at construction time and misses the swap.
    """

    def emit(self, record: logging.LogRecord) -> None:

        # Rebind the stream every emit. Cheap attribute write.
        self.stream = sys.stdout
        super().emit(record)
