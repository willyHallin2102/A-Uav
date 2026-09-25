"""
    tests / logger.py
    -----------------
    Test script for the logger class
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import re
import logging
import io
import time
import threading

from argparse import Namespace
from contextlib import redirect_stdout

from logs.levels import Level
from logs.formatters import ConsoleFormatter, ANSI, _COLOR, _ICON
from logs.logger import Profile, Logger, get_logger
from logs.handlers import ConsoleHandler

from tests._utils import runner, build_cli, CommandSpec

# ============================================================================
#       Helpers and Constants
# ============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

def strip_ansi(text: str) -> str:
    """ Remove all ANSI escape sequences from *text* """
    return _ANSI_RE.sub("", text)


def unique_name(prefix: str = "logger") -> str:
    """ Return a unique logger name to avoid cross-test handler reuse """
    return f"{prefix}_{time.time_ns()}"


def capture(f, *args, **kwargs) -> str:
    """ 
    Run `f` while capturing all written to sys.stdout. The returned 
    string output the captured while preserve ANSI escape codes.
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        f(*args, **kwargs)
    
    return buffer.getvalue()

# ============================================================================
#       Classes and functions to trial the logger
# ============================================================================

class A:
    def __init__(self):
        self.logger = get_logger(name="A", level=Level.DEBUG, profile="runtime")
    
    def debug(self): self.logger.debug("debug")
    def info(self): self.logger.info("info")
    def warning(self): self.logger.warning("warning")
    def error(self): self.logger.error("error")
    def critical(self): self.logger.critical("critical")


class B:
    def __init__(self):
        self.logger = get_logger(name="B", level=Level.INFO, profile="debug")
    
    def debug(self): self.logger.debug("debug")
    def info(self): self.logger.info("info")
    def warning(self): self.logger.warning("warning")
    def error(self): self.logger.error("error")
    def critical(self): self.logger.critical("critical")


def _a():
    Logger(name="a", profile="debug").warning("DEBUG")



# ============================================================================
#       Test Method 
# ============================================================================

def test_levels(args: Namespace):
    """ Test the Level IntEnum mirrors stdlib logging values """
    levels = {
        Level.DEBUG: logging.DEBUG, Level.INFO: logging.INFO, 
        Level.WARNING: logging.WARNING, Level.ERROR: logging.ERROR,
        Level.CRITICAL: logging.CRITICAL
    }

    print()
    for level, std in levels.items():
        print(f"{level.name:<10}: Level={int(level):>3},\tstdlib={std:>3}")
        assert int(level) == std, f"{level.name} mismatch: {int(level) != {std}}"

    assert Level.DEBUG < Level.INFO < Level.WARNING < Level.ERROR < Level.CRITICAL

    print()
    for std in levels.values():
        level = Level(std)
        print(f"Level({std}) -> {level.name}")
        assert int(level) == std
    
    print()
    try:
        Level(9999)
        raise AssertionError("Level(9999) should have raised ValueError")
    
    except ValueError as ve:
        print(f"Raised as expected: {ve}")



def test_ansi_constants(args: Namespace):
    """ Testing ANSI escape code constants exists with expected prefixes """
    for name in ("RESET",):
        value = getattr(ANSI, name)
        assert value.startswith("\033["), f"ANSI.{name}, malformed: {value!r}"
        print(f"ANSI.{name} = {value!r}")
    
    print("\nForeground Colors\n")
    for attr in ("BLACK","RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN","WHITE"):
        value = getattr(ANSI.FG, attr)
        assert value.startswith("\033[3"), f"FG.{attr} malformed: {value!r}"
        print(f"ANSI.FG.{attr} = {value!r}")
    
    print("\nBackground Colors\n")
    for attr in ("BLACK","RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN","WHITE"):
        value = getattr(ANSI.BG, attr)
        assert value.startswith("\033[4"), f"FG.{attr} malformed: {value!r}"
        print(f"ANSI.FG.{attr} = {value!r}")
    
    print("\nTesting ANSI.Text\n")
    for attr in ("BOLD","DIM","ITALIC","UNDERLINE","BLINK","REVERSE","STRIKE"):
        value = getattr(ANSI.Text, attr)
        assert value.startswith("\033["), f"Text.{attr} malformed: {value!r}"
        print(f"ANSI.Text.{attr} = {value!r}")



def test_level_maps(args: Namespace):
    """ Test _COLOR and _ICON maps cover every level of severity """
    for level in Level:

        assert level in _COLOR, f"Missing color for {level.name}"
        assert level in _ICON, f"Missing icon for {level.name}"

        print(f"{level.name:<8} color={_COLOR[level]!r} icon={_ICON[level]!r}")
    
    crit = _COLOR[Level.CRITICAL]
    assert ANSI.BG.RED in crit, "CRITICAL should be having red background"
    assert ANSI.Text.BOLD in crit, "CRITICAL should be bold"

    print(f"CRITICAL color contains BOLD + BG.RED")



def test_formatter_profiles(args: Namespace):
    """ Test ConsoleFormatter construction and profile validation """
    for profile in ("runtime", "debug"):

        fmt = ConsoleFormatter(profile)
        print(f"ConsoleFormatter({profile!r}) created: {fmt}")

        assert fmt._debug is (profile == "debug")
    
    try:
        ConsoleFormatter("abc")
        raise AssertionError("Invalid profile should raise 'ValueError'")
    
    except ValueError as ve:
        print(f"Raised as expected: {ve}")



def test_handler_stream_rebinding(args: Namespace):
    """ Test ConsoleHandler picks up a swapped sys.stdout on every emit """
    handler = ConsoleHandler()
    handler.setFormatter(ConsoleFormatter("runtime"))

    record = logging.LogRecord(
        name="app", level=logging.INFO, pathname=__file__,
        lineno=1, msg="rebound", args=(), exc_info=None,
    )

    buffer_1 = io.StringIO()
    stdout = sys.stdout

    try:
        sys.stdout = buffer_1
        handler.emit(record)
    
    finally:
        sys.stdout = stdout
    
    captured = buffer_1.getvalue()
    print(f"Captured: {captured!r}")

    assert "rebound" in strip_ansi(captured), "Output should go to swapped stdout"
    assert handler.stream is buffer_1, \
        "Handler stream should be rebounded to a swapped stdout"
    
    buffer_2 = io.StringIO()
    with redirect_stdout(buffer_2):
        handler.emit(record)
    
    assert "rebound" in strip_ansi(buffer_2.getvalue())
    print("contextlib.redirect_stdout captured output")



def test_logger_initialization(args: Namespace):
    """ Test Logger construction and configuration """
    logger = get_logger(name="test_logger_init", level=Level.CRITICAL)
    print(f"Logger: {logger}\nname: {logger.name}\nlevel: {logger.level}")

    assert logger.name.startswith("test_logger_")

    underlying = logger.get_logger()
    print(f"Underlying: {underlying}")

    assert isinstance(underlying, logging.Logger)

    console_handlers = [
        h for h in underlying.handlers if isinstance(h, ConsoleHandler)
    ]
    print(f"  ConsoleHandler count: {len(console_handlers)}")
    assert len(console_handlers) == 1, "Exactly one ConsoleHandler expected"

    fmt = console_handlers[0].formatter
    assert isinstance(fmt, ConsoleFormatter)



def test_logger_singleton_handlers(args: Namespace):
    """ Test that re-creating a logger for the same name reuse handlers. """
    name = unique_name("reuse")

    one = Logger(name=name, level=Level.DEBUG)
    two = Logger(name=name, level=Level.DEBUG)

    handlers_1 = [
        h for h in one.get_logger().handlers if isinstance(h, ConsoleHandler)
    ]
    handlers_2 = [
        h for h in two.get_logger().handlers if isinstance(h, ConsoleHandler)
    ]

    print(f"first handler: {len(handlers_1)}")
    print(f"second handler: {len(handlers_2)}")

    assert len(handlers_1) == 1, "first handler len isn't 1"
    assert len(handlers_2) == 1, "second handler len isn't 1"
    assert handlers_1[0] is handlers_2[0], \
        "Handler should be reused"



def test_logger_level_property(args: Namespace):
    """ Test the level property setter/getter propagates to handlers """
    logger = get_logger(name="liquorice", level=Level.INFO)
    assert logger.level == int(Level.INFO), "Level was suppose to be INFO"
    print(f"initial level: {int(Level.INFO)}: {logger.level}")

    logger.level = Level.WARNING
    for handler in logger.get_logger().handlers:
        if isinstance(handler, ConsoleHandler):
            assert handler.level == int(Level.WARNING), "Handler was not WARNING"
    
    print(f" After assigning Level.WARNING: logger={logger.level} handler synced")

    logger.level = int(Level.ERROR)
    assert logger.level == int(Level.ERROR)
    print(f" After set_level(Level.Error): {logger.level}")

    logger.set_level(Level.CRITICAL)
    assert logger.level == int(Level.CRITICAL)
    print(f" After set_level(Level.CRITICAL): {logger.level}")
    


def test_logger_enable_disable(args: Namespace):
    """ Test disable()/enable() toggles output """
    logger = get_logger(name="ed", level=Level.DEBUG)

    logger.disable()
    print(f"level after disable: {logger.level}")
    assert logger.level == logging.CRITICAL + 1, \
        "Disable should raise level above CRITICAL"
    
    captured = capture(logger.critical, "should not appear")
    print(f"Captured critical after disable: {captured!r}")
    assert captured == "", "Critical should be suppressed after disable"

    logger.enable()
    assert logger.level == int(logger.level)
    print(f"level after enable(): {logger.level}")

    logger.enable(Level.DEBUG)
    assert logger.level == int(Level.DEBUG)
    print(f"level after enable(Level.DEBUG): {logger.level}")



def test_logger_methods(args: Namespace):
    """Test each logging method emits at the correct level."""

    logger = get_logger(level=Level.DEBUG)

    cases = [
        ("debug", "debug-msg", Level.DEBUG),
        ("info", "info-msg", Level.INFO),
        ("warning", "warning-msg", Level.WARNING),
        ("error", "error-msg", Level.ERROR),
        ("critical", "critical-msg", Level.CRITICAL),
    ]

    for method_name, msg, level in cases:

        method = getattr(logger, method_name)
        captured = capture(method, msg)
        plain = strip_ansi(captured)
        
        print(f"{method_name:<9} -> {plain!r}")
        
        assert msg in plain, f"{method_name} message missing"
        assert level.name in plain, f"{method_name} level name missing"

    quiet = get_logger(level=Level.ERROR)
    captured = capture(quiet.info, "should be filtered")
    print(f"info below ERROR threshold captured: {captured!r}")
    assert captured == "", "INFO below ERROR should not emit"

    logger = get_logger(level=Level.DEBUG)
    captured = capture(logger.info, "value=%d name=%s", 7, "x")
    plain = strip_ansi(captured)

    print(f"interpolated: {plain!r}")
    assert "value=7 name=x" in plain

    logger = get_logger(level=Level.DEBUG)
    try:
        raise RuntimeError("kaboom")

    except RuntimeError:
        captured = capture(logger.exception, "operation failed")

    plain = strip_ansi(captured)

    print(f"exception output: {plain!r}")

    assert "operation failed" in plain
    assert "RuntimeError: kaboom" in plain, "Traceback missing"

    logger = get_logger(level=Level.DEBUG, profile="debug")
    captured = capture(logger.error, "caller check")
    plain = strip_ansi(captured)

    print(f"debug output: {plain!r}")

    assert Path(__file__).name in plain, \
        f"stacklevel=2 should report caller file, got: {plain!r}"
    assert "tests/logger.py" not in plain, "Should not report wrapper module"



def test_get_logger_factory(args: Namespace):
    """Test the module-level get_logger factory."""
    
    logger = get_logger(name=unique_name("factory"))
    
    print(f"  returned: {logger}")
    
    assert isinstance(logger, Logger)
    assert logger.level == int(Level.INFO)
    
    print(f"default level: {logger.level}")

    logger = get_logger(
        name=unique_name("factory"), level=Level.DEBUG, profile="debug",
    )

    assert logger.level == int(Level.DEBUG)
    handler = [
        h for h in logger.get_logger().handlers if isinstance(h, ConsoleHandler)
    ][0]
    
    assert handler.formatter._debug is True
    print(f"custom level={logger.level} debug_formatter={handler.formatter._debug}")



def test_logger_threads(args: Namespace):
    """Smoke-test concurrent logging from multiple threads."""
    logger = get_logger(level=Level.DEBUG)

    n_threads, per_thread = 4, 50
    errors: list[BaseException] = []

    def worker(thread_id: int) -> None:
        try:
        
            for i in range(per_thread):
                logger.info("thread-%d msg-%d", thread_id, i)
        
        except BaseException as e:
            errors.append(e)


    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    start = time.time()
    
    for t in threads:
        t.start()

    for t in threads:
        t.join()
    
    elapsed = time.time() - start
    print(
        f"{n_threads * per_thread} messages from {n_threads} "
        f"threads in {elapsed:.3f}s"
    )
    assert not errors, f"Worker errors: {errors}"



def test_logger_class_calls(args: Namespace):
    """ Testing initialize classes invoking the logger instance """
    a, b = A(), B()

    a.debug()
    b.debug()
    print()

    a.info()
    b.info()
    print()
    
    a.warning()
    b.warning()
    print()
    
    a.error()
    b.error()
    print()
    
    a.critical()
    b.critical()
    print()
    _a()


# ============================================================================
#       Main Runner (Running the script)
# ============================================================================



COMMON = [
    {"flags": ["--verbose", "-v"], "kwargs": {"action": "store_true"}},
]

@runner
def main():
    parser = build_cli([
        CommandSpec(
            "levels", "Test level enum values and ordering", test_levels, [*COMMON]
        ),
        CommandSpec(
            "ansi", "Test ANSI escape code constants", 
            test_ansi_constants, [*COMMON]
        ),
        CommandSpec(
            "maps", "Testing _COLOR and _ICON maps cover all levels",
            test_level_maps, [*COMMON]
        ),
        CommandSpec(
            "formatter", "The `ConsoleFormatter` profile validation",
            test_formatter_profiles, [*COMMON]
        ),
        CommandSpec(
            "rebinding", "Test ConsoleHandler rebind sys.stdout on emit",
            test_handler_stream_rebinding, [*COMMON]
        ),
        CommandSpec(
            "init", "Test Logger initialization and configuration",
            test_logger_initialization, [*COMMON]
        ),
        CommandSpec(
            "singleton", "Test Logger reuses handlers for the same name",
            test_logger_singleton_handlers, [*COMMON]
        ),
        CommandSpec(
            "level_prop", "Testing Logger level property getter / setter",
            test_logger_level_property, [*COMMON]
        ),
        CommandSpec(
            "enable_disable", "Test Logger disable()/enable() toggles",
            test_logger_enable_disable, [*COMMON]
        ),
        CommandSpec(
            "methods", "Test each logger emission method",
            test_logger_methods, [*COMMON]
        ),
        CommandSpec(
            "factory", "Test module-level get_logger factory",
            test_get_logger_factory, [*COMMON],
        ),
        CommandSpec(
            "threads", "Smoke-test concurrent logging from threads",
            test_logger_threads, [*COMMON],
        ),
        CommandSpec(
            "class", "Smoke-test concurrent logging from threads",
            test_logger_class_calls, [*COMMON],
        ),
    ])

    args = parser.parse_args()
    args._handler(args)



if __name__ == "__main__":
    main()