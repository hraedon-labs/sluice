"""Behavioural pin for the identifier gate's publication-visibility contract.

The gate (``scripts/check_committed_identifiers.py``) turns a missing denylist
into a hard failure on a repo whose ``publication.toml`` declares it public, and
into a quiet skip on a private-until-review one. Which branch it takes is decided
by reading ``visibility``, so that read is the whole safety property: if it
mistakes a public declaration for a private one, the gate skips, exits 0, and CI
is green having scanned nothing.

It used to compare against the bare string "public", so "Public", a typo, a
missing key and a non-string value all fell into the fail-OPEN branch. These
tests run the gate exactly as CI does -- as a subprocess, against a throwaway git
repo, with the denylist removed from the environment -- and read the exit code.
They deliberately do not import the gate or call its private helpers: this file
is shared across repos whose gates differ internally, and the contract it pins is
the exit code, not the implementation.

Stdlib only (``unittest``), so it runs under pytest and also as
``python3 tests/test_identifier_gate_visibility.py`` in a job with no test
dependencies installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _find_gate() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "scripts" / "check_committed_identifiers.py"
        if candidate.is_file():
            return candidate
    raise RuntimeError("could not locate scripts/check_committed_identifiers.py")


GATE = _find_gate()
# Resolved once, absolutely, so the fixture never runs whatever "git" happens to
# be first on PATH inside the scratch directory.
_GIT = shutil.which("git") or "git"

# The denylist variable name differs per repo (a shared org secret, or a
# per-repo prefixed one). Read it from the gate rather than hardcoding it, so a
# rename cannot leave this file setting a variable the gate never reads.
_ENV_NAMES = sorted(
    set(
        re.findall(
            r"""environ\.get\(\s*["']([A-Z0-9_]*FORBIDDEN_IDENTIFIERS)["']""",
            GATE.read_text(encoding="utf-8"),
        )
    )
)
if len(_ENV_NAMES) != 1:
    raise RuntimeError(f"expected exactly one denylist variable in the gate, found {_ENV_NAMES}")
DENYLIST_VAR = _ENV_NAMES[0]


_EMPTY_HOME = tempfile.mkdtemp(prefix="gate-visibility-home-")


def _clean_env() -> dict[str, str]:
    """The caller's environment minus every denylist-shaped variable.

    Inheriting a developer's or CI job's real denylist would arm the gate and
    turn every "unconfigured" case below into a configured one -- the tests
    would pass without exercising the branch they exist for.
    """
    env = {k: v for k, v in os.environ.items() if not k.endswith("FORBIDDEN_IDENTIFIERS")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Some gates fall back to a denylist FILE under $HOME when the variable is
    # unset (a developer convenience). A developer box that has one would arm
    # the gate exactly as an inherited variable would, so HOME points at an
    # empty directory for the duration of the run.
    env["HOME"] = _EMPTY_HOME
    return env


class VisibilityContract(unittest.TestCase):
    """The gate's exit code for each shape of the publication declaration."""

    def setUp(self) -> None:
        """Create the scratch directory and the first fixture repo."""
        self._tmp = tempfile.TemporaryDirectory()
        self._count = 0
        self._fresh()

    def _fresh(self) -> None:
        """Point self.root at a new, empty git repo (one per subTest case)."""
        self._count += 1
        self.root = Path(self._tmp.name) / f"repo{self._count}"
        self.root.mkdir()
        self._git("init", "-q")
        (self.root / "README.md").write_text("nothing forbidden here\n", encoding="utf-8")

    def tearDown(self) -> None:
        """Remove every fixture repo this test created."""
        self._tmp.cleanup()

    def _git(self, *args: str) -> None:
        """Run git in the fixture repo with an isolated, deterministic config."""
        subprocess.run(
            [
                _GIT,
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=Test",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=self.root,
            check=True,
            env=_clean_env(),
            capture_output=True,
        )

    def _run(
        self, declaration: str | None, denylist: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Commit the fixture and run the gate on it, returning the completed process."""
        if declaration is not None:
            (self.root / "publication.toml").write_text(declaration, encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "fixture")
        env = _clean_env()
        if denylist is not None:
            env[DENYLIST_VAR] = denylist
        return subprocess.run(
            [sys.executable, str(GATE)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    @staticmethod
    def _declare(visibility: str) -> str:
        """A minimal declaration body naming *visibility*."""
        return f'[publication]\nremote_owner = "someone"\nvisibility = "{visibility}"\n'

    # -- public, in any case or padding, arms the gate -----------------------

    def test_public_any_spelling_fails_closed_without_a_denylist(self) -> None:
        """Any casing or padding of "public" arms the gate: no denylist is a failure."""
        for spelling in ("public", "Public", "PUBLIC", "PuBlIc", "  public  "):
            with self.subTest(spelling=spelling):
                self._fresh()
                result = self._run(self._declare(spelling))
                self.assertEqual(result.returncode, 1, result.stderr)

    # -- anything outside the closed set is an error, never "not public" -----

    def test_unrecognised_declaration_fails_rather_than_guessing(self) -> None:
        """A declaration outside the closed set is an error, never a quiet "not public"."""
        bodies = {
            "missing key": '[publication]\nremote_owner = "someone"\n',
            "empty string": '[publication]\nvisibility = ""\n',
            "boolean": "[publication]\nvisibility = true\n",
            "integer": "[publication]\nvisibility = 1\n",
            "typo": '[publication]\nvisibility = "publik"\n',
            "outside the set": '[publication]\nvisibility = "internal"\n',
            "no table": 'visibility = "public"\n',
            "unparseable": "[publication\nvisibility = public\n",
        }
        for label, body in bodies.items():
            with self.subTest(case=label):
                self._fresh()
                result = self._run(body)
                self.assertEqual(result.returncode, 1, result.stderr)

    # -- the private side still no-ops, so tightening did not over-block -----

    def test_private_until_review_any_spelling_skips_without_a_denylist(self) -> None:
        """Tightening the public side must not block a private-until-review repo."""
        for spelling in (
            "private-until-review",
            "Private-Until-Review",
            "PRIVATE-UNTIL-REVIEW",
            "  private-until-review  ",
        ):
            with self.subTest(spelling=spelling):
                self._fresh()
                result = self._run(self._declare(spelling))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_absent_declaration_still_skips(self) -> None:
        """A repo that never opted into the publication system is not blocked."""
        result = self._run(None)
        self.assertEqual(result.returncode, 0, result.stderr)

    # -- recognised-and-scanned is distinguishable from "could not read it" --

    def test_oddly_cased_public_with_a_denylist_scans_and_passes_clean(self) -> None:
        """Recognised-and-scanned is distinguishable from could-not-read-the-declaration."""
        result = self._run(self._declare("Public"), denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_oddly_cased_public_with_a_denylist_still_catches_a_hit(self) -> None:
        """The scan behind a recognised "Public" really runs: a planted token fails it."""
        (self.root / "notes.txt").write_text("mentions zzzsynthetictoken here\n", encoding="utf-8")
        result = self._run(self._declare("Public"), denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 1, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
