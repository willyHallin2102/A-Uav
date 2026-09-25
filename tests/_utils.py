"""
"""
from __future__ import annotations

import argparse
import logging
import sys

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable



@dataclass
class CommandSpec:
    """ Specification for a CLI command. """

    name: str
    help: str
    handler: Callable[..., Any]
    args: list[dict[str, Any]] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the command specification."""

        if not self.name.isidentifier():
            raise ValueError(f"Command-name {self.name!r} must be identifier")

        if not callable(self.handler):
            raise TypeError(f"Command handler {self.handler!r} must be callable")

        for alias in self.aliases:
            if not alias.isidentifier():
                raise ValueError(f"Command alias {alias!r} must be identifier")



def build_cli(
    commands: list[CommandSpec], *, description: str = "Test CLI",
) -> argparse.ArgumentParser:
    """
    Build an argparse parser from command specifications.
    """

    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True,)

    for command in commands:
        subparser = subparsers.add_parser(
            command.name, help=command.help, aliases=command.aliases,
        )

        for arg in command.args:
            subparser.add_argument(*arg["flags"], **arg["kwargs"])

        subparser.set_defaults(_handler=command.handler)

    return parser



def runner(f: Callable[..., Any]) -> Callable[..., Any]:
    """
    Wrap a test entry point with a more standardized error management, it
    allows for `KeyboardInterruption` with a status 130 which prevent infinite
    loops issues. Other exceptions are logged with their traceback and result
    in exit status 1. 
    """
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return f(*args, **kwargs)

        except KeyboardInterrupt:
            print("\n⚠️ Aborted by user")
            raise SystemExit(130)

        except Exception as exc:
            logging.error("Test failed: %s", exc, exc_info=True,)
            print(f"\n⛔ Test failed: {exc}")
            raise SystemExit(1)

    return wrapper
