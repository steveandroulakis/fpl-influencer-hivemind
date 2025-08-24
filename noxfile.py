"""Nox configuration for testing across Python versions."""

import nox


@nox.session(python=["3.11", "3.12", "3.13"])
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run("pytest", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Run linting checks."""
    session.install(".[dev]")
    session.run("ruff", "check", ".")
    session.run("ruff", "format", "--check", ".")


@nox.session
def typecheck(session: nox.Session) -> None:
    """Run type checking."""
    session.install(".[dev]")
    session.run("mypy", "src")


@nox.session
def coverage(session: nox.Session) -> None:
    """Run tests with coverage reporting."""
    session.install(".[dev]")
    session.run("pytest", "--cov-report=html", "--cov-report=xml")
