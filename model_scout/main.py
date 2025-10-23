"""
main.py — thin wrapper that defers to CLI module.
"""

from .cli import cli_entry

if __name__ == "__main__":
    cli_entry()
