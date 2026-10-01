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

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
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


def _clean_env(home: Path) -> dict[str, str]:
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
    # empty directory for the duration of the run. It lives inside the test's
    # own temporary directory so it is removed with it (a module-level mkdtemp
    # here used to leave one gate-visibility-home-* directory behind per run).
    env["HOME"] = str(home)
    return env


# Run the gate as Windows does: a locale codec (cp1252) that accepts almost any
# byte, so text decoded with the locale default can never raise. On Linux the
# locale is UTF-8 and a missing explicit UTF-8 decode is invisible; this makes it
# visible. Patches the two hooks Python consults when no encoding is given:
# locale.getencoding (subprocess text mode) and io.text_encoding (pathlib and
# io.open). A bare builtin open() still uses the C-level locale; CI's Windows
# legs cover that.
_FORCED_LOCALE_WRAPPER = """
import io, locale, os, runpy, sys
_codec = sys.argv[1]
locale.getencoding = lambda: _codec
locale.getpreferredencoding = lambda do_setlocale=True: _codec
def _text_encoding(encoding, stacklevel=2):
    return _codec if encoding in (None, "locale") else encoding
io.text_encoding = _text_encoding
sys.argv = sys.argv[2:]
# As `python script.py` would: the script's directory, not the cwd, is sys.path[0]
# (the gate imports check_publication_plumbing from beside itself).
sys.path[0] = os.path.dirname(os.path.abspath(sys.argv[0]))
runpy.run_path(sys.argv[0], run_name="__main__")
"""


class VisibilityContract(unittest.TestCase):
    """The gate's exit code for each shape of the publication declaration."""

    def setUp(self) -> None:
        """Create the scratch directory (cleaned up even if setUp fails) and a repo."""
        self._tmp = tempfile.TemporaryDirectory(prefix="gate-visibility-")
        self.addCleanup(self._tmp.cleanup)
        self._home = Path(self._tmp.name) / "empty-home"
        self._home.mkdir()
        self._count = 0
        self._fresh()

    def _env(self) -> dict[str, str]:
        """The isolated environment for this test (see _clean_env)."""
        return _clean_env(self._home)

    def _fresh(self) -> None:
        """Point self.root at a new, empty git repo (one per subTest case)."""
        self._count += 1
        self.root = Path(self._tmp.name) / f"repo{self._count}"
        self.root.mkdir()
        self._git("init", "-q")
        (self.root / "README.md").write_text("nothing forbidden here\n", encoding="utf-8")

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
            env=self._env(),
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
        return self._gate(denylist=denylist)

    def _gate(
        self, *args: str, denylist: str | None = None, path_env: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run the gate in the fixture repo as-is (no add/commit) with *args*."""
        env = self._env()
        if denylist is not None:
            env[DENYLIST_VAR] = denylist
        if path_env is not None:
            env["PATH"] = path_env
        return subprocess.run(
            [sys.executable, str(GATE), *args],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def _gate_forced_locale(self, codec: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run the gate with *codec* forced as the locale default (no denylist)."""
        return subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", _FORCED_LOCALE_WRAPPER, codec, str(GATE), *args],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
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


class DeclarationSourceContract(VisibilityContract):
    """Where the verdict and the scanned bytes come from: the index in --staged.

    Subclasses VisibilityContract only to share its fixtures; the inherited tests
    are blanked out after the class body so they do not run twice.
    """

    # -- --staged judges the index, not the worktree --------------------------

    def test_staged_public_index_with_private_worktree_fails_closed(self) -> None:
        """The publication-flip commit: public staged, worktree still private."""
        self._run(self._declare("private-until-review"))
        (self.root / "publication.toml").write_text(self._declare("public"), encoding="utf-8")
        self._git("add", "publication.toml")
        (self.root / "publication.toml").write_text(
            self._declare("private-until-review"), encoding="utf-8"
        )
        result = self._gate("--staged")
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_staged_private_index_with_public_worktree_skips(self) -> None:
        """The inverse divergence proves the index, not the worktree, decides."""
        self._run(self._declare("private-until-review"))
        (self.root / "publication.toml").write_text(self._declare("public"), encoding="utf-8")
        result = self._gate("--staged")
        self.assertEqual(result.returncode, 0, result.stderr)

    # -- only genuine absence is "never opted in" ------------------------------

    def test_directory_declaration_fails_rather_than_reading_as_absent(self) -> None:
        """A directory named publication.toml is present, not absent."""
        self._run(None)
        (self.root / "publication.toml").mkdir()
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_symlinked_declaration_fails_rather_than_being_followed(self) -> None:
        """A symlink (here to a private declaration, or dangling) is not a declaration."""
        for target_exists in (True, False):
            with self.subTest(target_exists=target_exists):
                self._fresh()
                self._run(None)
                target = self.root / "elsewhere.toml"
                if target_exists:
                    target.write_text(self._declare("private-until-review"), encoding="utf-8")
                (self.root / "publication.toml").symlink_to(target.name)
                result = self._gate()
                self.assertEqual(result.returncode, 1, result.stderr)

    def _stage_raw_declaration(self, mode: str, stage_lines: list[str]) -> None:
        """Write index entries for publication.toml directly (mode/oid/stage)."""
        info = "".join(f"{mode} {line}\tpublication.toml\n" for line in stage_lines)
        subprocess.run(
            [_GIT, "update-index", "--index-info"],
            cwd=self.root,
            input=info.encode(),
            check=True,
            env=self._env(),
            capture_output=True,
        )

    def _blob(self, data: bytes) -> str:
        """Write *data* as a blob and return its object id."""
        return (
            subprocess.run(
                [_GIT, "hash-object", "-w", "--stdin"],
                cwd=self.root,
                input=data,
                check=True,
                env=self._env(),
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )

    def test_staged_unreadable_declaration_entries_fail_closed(self) -> None:
        """A conflicted, gitlink, symlink or non-UTF-8 staged declaration is not absence.

        The "symlink to private TOML" case stages a 120000 entry whose target
        string is itself a valid private declaration: it can only fail on the
        mode check, not incidentally on a parse error as the plain symlink does.
        """
        public = b'[publication]\nvisibility = "public"\n'
        private = b'[publication]\nvisibility = "private-until-review"\n'
        private_bad_utf8 = b'[publication]\nvisibility = "private-until-review"\n# \xff\n'
        for label in (
            "conflicted",
            "gitlink",
            "symlink",
            "symlink to private TOML",
            "executable regular file (control)",
            "non-utf-8",
        ):
            with self.subTest(case=label):
                self._fresh()
                self._run(None)
                if label == "conflicted":
                    oid = self._blob(public)
                    self._stage_raw_declaration("100644", [f"{oid} 1", f"{oid} 2", f"{oid} 3"])
                elif label == "gitlink":
                    head = (
                        subprocess.run(
                            [_GIT, "rev-parse", "HEAD"],
                            cwd=self.root,
                            check=True,
                            env=self._env(),
                            capture_output=True,
                        )
                        .stdout.decode()
                        .strip()
                    )
                    self._stage_raw_declaration("160000", [f"{head} 0"])
                elif label == "symlink":
                    self._stage_raw_declaration("120000", [f"{self._blob(b'elsewhere.toml')} 0"])
                elif label == "symlink to private TOML":
                    self._stage_raw_declaration("120000", [f"{self._blob(private)} 0"])
                elif label == "executable regular file (control)":
                    # Control: the same bytes as a regular 100755 file ARE a
                    # declaration, so the case above fails on mode alone.
                    self._stage_raw_declaration("100755", [f"{self._blob(private)} 0"])
                    result = self._gate("--staged")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    continue
                else:
                    self._stage_raw_declaration("100644", [f"{self._blob(private_bad_utf8)} 0"])
                result = self._gate("--staged")
                self.assertEqual(result.returncode, 1, result.stderr)

    # -- --staged scans the staged bytes, not the worktree copy ----------------

    def test_staged_token_hidden_by_a_clean_worktree_copy_is_caught(self) -> None:
        """Stage a forbidden token, then overwrite the file: the commit still records it."""
        self._run(self._declare("private-until-review"))
        notes = self.root / "notes.txt"
        notes.write_text("mentions zzzsynthetictoken here\n", encoding="utf-8")
        self._git("add", "notes.txt")
        notes.write_text("clean in the worktree now\n", encoding="utf-8")
        result = self._gate("--staged", denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_unstaged_token_in_the_worktree_does_not_block_a_clean_commit(self) -> None:
        """The inverse: unstaged junk in the worktree is not what the commit records."""
        self._run(self._declare("private-until-review"))
        notes = self.root / "notes.txt"
        notes.write_text("clean\n", encoding="utf-8")
        self._git("add", "notes.txt")
        notes.write_text("mentions zzzsynthetictoken here\n", encoding="utf-8")
        result = self._gate("--staged", denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_staged_scan_reads_raw_blobs_not_smudged_copies(self) -> None:
        """A smudge filter must not be able to strip a token from what is scanned."""
        self._run(self._declare("private-until-review"))
        (self.root / ".gitattributes").write_text("*.txt filter=strip\n", encoding="utf-8")
        self._git("config", "filter.strip.smudge", "sed s/zzzsynthetictoken/redacted/")
        self._git("config", "filter.strip.clean", "cat")
        notes = self.root / "notes.txt"
        notes.write_text("mentions zzzsynthetictoken here\n", encoding="utf-8")
        self._git("add", ".gitattributes", "notes.txt")
        notes.write_text("clean in the worktree now\n", encoding="utf-8")
        result = self._gate("--staged", denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_staged_type_change_to_a_token_bearing_symlink_is_caught(self) -> None:
        """Regular file re-staged as a symlink is a type change (T), not A/C/M."""
        self._run(self._declare("private-until-review"))
        link = self.root / "notes.txt"
        link.write_text("clean\n", encoding="utf-8")
        self._git("add", "notes.txt")
        self._git("commit", "-q", "--no-verify", "-m", "regular file")
        link.unlink()
        link.symlink_to("zzzsynthetictoken.example.invalid")
        self._git("add", "notes.txt")
        result = self._gate("--staged", denylist="zzzsynthetictoken")
        self.assertEqual(result.returncode, 1, result.stderr)

    # -- removing a public declaration is not "never opted in" ----------------

    def test_removing_a_public_declaration_does_not_disarm_the_gate(self) -> None:
        """Deleting publication.toml does not make the remote private."""
        for how in ("staged removal", "committed removal"):
            with self.subTest(how=how):
                self._fresh()
                self._run(self._declare("public"))
                self._git("rm", "-q", "--cached", "publication.toml")
                if how == "committed removal":
                    (self.root / "publication.toml").unlink()
                    self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                    self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "later")
                    result = self._gate()
                else:
                    result = self._gate("--staged")
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_a_merge_that_deletes_a_public_declaration_does_not_disarm_the_gate(self) -> None:
        """Both parents still have the file; the conflict is resolved by deleting it."""
        self._run(self._declare("public"))
        self._git("checkout", "-q", "-b", "side")
        self._run(self._declare("Public"))
        self._git("checkout", "-q", "-")
        self._run(self._declare("PUBLIC"))
        merge = subprocess.run(
            [
                _GIT,
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=Test",
                "merge",
                "--no-edit",
                "side",
            ],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertIn("CONFLICT", merge.stdout + merge.stderr)  # fixture really is a merge
        self._git("rm", "-q", "-f", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "resolve by deleting")
        self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "later")
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_laundering_a_public_declaration_through_garbage_does_not_disarm(self) -> None:
        """Public, then invalid, then deleted: only a clean private declaration opts out."""
        for garbage in ('[publication]\nvisibility = "publik"\n', "[publication\n", "[x]\n"):
            with self.subTest(garbage=garbage):
                self._fresh()
                self._run(self._declare("public"))
                self._run(garbage)
                self._git("rm", "-q", "publication.toml")
                self.assertEqual(self._gate("--staged").returncode, 1)
                self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                result = self._gate()
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_a_non_utf8_last_declaration_is_not_an_opt_out(self) -> None:
        """A corrupt last declaration must not launder a public one into absence."""
        self._run(self._declare("public"))
        (self.root / "publication.toml").write_bytes(
            b'[publication]\nvisibility = "private-until-review"\n# \xff\n'
        )
        self._git("add", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "corrupt")
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("private-until-review", result.stderr)

    def test_a_private_branch_cannot_launder_a_deleted_public_lineage(self) -> None:
        """Base, then (public, deleted) merged with (private) into an absent merge."""
        self._run(None)
        self._git("checkout", "-q", "-b", "public-line")
        self._run(self._declare("public"))
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "delete public")
        self._git("checkout", "-q", "-")
        self._run(self._declare("private-until-review"))
        merge = subprocess.run(
            [
                _GIT,
                "-c",
                "user.email=t@example.invalid",
                "-c",
                "user.name=Test",
                "merge",
                "--no-commit",
                "--no-ff",
                "public-line",
            ],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if (self.root / "publication.toml").exists():
            self._git("rm", "-q", "-f", "publication.toml")
        self.assertIn(merge.returncode, (0, 1), merge.stderr)
        # The MERGE commit itself records no declaration: one parent is private,
        # the other is an absent lineage whose last declaration was public.
        self._git("commit", "-q", "--no-verify", "-m", "merge, declaration absent")
        self.assertEqual(
            len(
                subprocess.run(
                    [_GIT, "rev-list", "--parents", "-1", "HEAD"],
                    cwd=self.root,
                    env=self._env(),
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
            ),
            3,
        )
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_a_symlink_declaration_in_history_is_not_a_private_opt_out(self) -> None:
        """A symlink whose target string reads as private TOML is not a declaration."""
        self._run(self._declare("public"))
        decl = self.root / "publication.toml"
        decl.unlink()
        decl.symlink_to(self._declare("private-until-review"))
        self._git("add", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "symlink declaration")
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)

    def test_removing_a_private_declaration_is_a_clean_opt_out(self) -> None:
        """Leaving the publication system is allowed once the declaration said private."""
        for history in (["private-until-review"], ["public", "private-until-review"]):
            with self.subTest(history=history):
                self._fresh()
                for visibility in history:
                    self._run(self._declare(visibility))
                self._git("rm", "-q", "publication.toml")
                self.assertEqual(self._gate("--staged").returncode, 0)
                self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                result = self._gate()
                self.assertEqual(result.returncode, 0, result.stderr)

    # -- an unusable git is an error, not a verdict ----------------------------

    def test_unusable_git_fails_closed_in_message_mode(self) -> None:
        """Message mode returns straight after the verdict, so a broken git must fail."""
        self._run(self._declare("public"))
        message = self.root / "msg.txt"
        message.write_text("an ordinary message\n", encoding="utf-8")
        empty_bin = Path(self._tmp.name) / "empty-bin"
        empty_bin.mkdir(exist_ok=True)
        result = self._gate("--message-file", str(message), path_env=str(empty_bin))
        self.assertEqual(result.returncode, 1, result.stderr)

    # -- the declaration is decoded as UTF-8 explicitly, never the locale ------

    def test_forced_locale_wrapper_really_changes_the_default_codec(self) -> None:
        """Self-check: under the wrapper, text-mode subprocess output decodes as cp1252."""
        probe = Path(self._tmp.name) / "probe_codec.py"
        probe.write_text(
            "import subprocess, sys\n"
            "out = subprocess.run([sys.executable, '-c', "
            "'import sys; sys.stdout.buffer.write(bytes([0xff]))'], "
            "capture_output=True, text=True).stdout\n"
            "sys.exit(0 if out == '\\u00ff' else 3)\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8=0", "-c", _FORCED_LOCALE_WRAPPER, "cp1252", str(probe)],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_non_utf8_declaration_fails_closed_under_a_permissive_locale(self) -> None:
        """Under cp1252 (Windows) a locale decode accepts the byte; UTF-8 must not.

        Each body is a clean private declaration plus one 0xFF byte in a comment,
        so a gate that decodes with the locale codec reads it as private and
        skips (exit 0). Covered: the staged blob, the worktree file, and the last
        declaration in history before a removal.
        """
        bad = b'[publication]\nvisibility = "private-until-review"\n# \xff\n'
        for where in ("staged", "worktree", "history"):
            with self.subTest(where=where):
                self._fresh()
                if where == "history":
                    self._run(self._declare("public"))
                (self.root / "publication.toml").write_bytes(bad)
                self._git("add", "publication.toml")
                if where == "staged":
                    (self.root / "publication.toml").write_text(
                        self._declare("private-until-review"), encoding="utf-8"
                    )
                    result = self._gate_forced_locale("cp1252", "--staged")
                elif where == "worktree":
                    self._git("commit", "-q", "--no-verify", "-m", "declaration")
                    result = self._gate_forced_locale("cp1252")
                else:
                    self._git("commit", "-q", "--no-verify", "-m", "corrupt")
                    self._git("rm", "-q", "publication.toml")
                    self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                    result = self._gate_forced_locale("cp1252")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_clean_declarations_still_pass_under_a_permissive_locale(self) -> None:
        """Control for the test above: the wrapper alone does not make the gate fail."""
        for history in (["private-until-review"], ["public", "private-until-review"]):
            with self.subTest(history=history):
                self._fresh()
                for visibility in history:
                    self._run(self._declare(visibility))
                self.assertEqual(self._gate_forced_locale("cp1252").returncode, 0)
                self.assertEqual(self._gate_forced_locale("cp1252", "--staged").returncode, 0)

    def test_absence_verdict_survives_the_strictest_locale(self) -> None:
        """The absence path's git reads decode under ASCII, so no decode arm is needed.

        Pins the argument that removed _git_or_none's UnicodeDecodeError arm: its
        only caller reads `rev-parse --verify -q HEAD`, whose output is a hex id or
        nothing. Under a strict ASCII locale, never-opted-in still skips, a deleted
        public declaration still fails, and neither raises.
        """
        for history, expected in (([], 0), (["public"], 1), (["private-until-review"], 0)):
            with self.subTest(history=history):
                self._fresh()
                self._run(None)
                for visibility in history:
                    self._run(self._declare(visibility))
                if history:
                    self._git("rm", "-q", "publication.toml")
                    self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                result = self._gate_forced_locale("ascii")
                self.assertEqual(result.returncode, expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    # -- a shallow clone cannot judge absence ---------------------------------

    def _shallow_clone(self, depth: int = 1) -> Path:
        """Clone self.root with --depth *depth* and point self.root at the clone."""
        source = self.root
        clone = Path(self._tmp.name) / f"shallow{self._count}"
        subprocess.run(
            [_GIT, "clone", "-q", f"--depth={depth}", source.resolve().as_uri(), str(clone)],
            check=True,
            env=self._env(),
            capture_output=True,
        )
        self.root = clone
        is_shallow = subprocess.run(
            [_GIT, "rev-parse", "--is-shallow-repository"],
            cwd=clone,
            check=True,
            env=self._env(),
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(is_shallow, "true")  # the fixture really is shallow
        return clone

    def test_shallow_clone_cannot_launder_a_deleted_public_declaration(self) -> None:
        """actions/checkout is depth 1 by default: the deleting commit is all it sees.

        The full clone fails closed; the shallow one used to see a root with no
        declaration and skip (exit 0). It must refuse instead, in tree and staged
        mode, whatever the clone depth.
        """
        for depth in (1, 2):
            with self.subTest(depth=depth):
                self._fresh()
                self._run(self._declare("public"))
                self._git("rm", "-q", "publication.toml")
                self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
                self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "later")
                self.assertEqual(self._gate().returncode, 1)  # full history: closed
                self._shallow_clone(depth)
                result = self._gate()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("shallow", result.stderr)
                self.assertEqual(self._gate("--staged").returncode, 1)

    def test_shallow_clone_with_a_declaration_is_not_blocked(self) -> None:
        """Only the absence verdict reads history: a declared shallow clone is judged as usual."""
        for visibility, expected in (("private-until-review", 0), ("public", 1)):
            with self.subTest(visibility=visibility):
                self._fresh()
                self._run(self._declare("public"))
                self._run(self._declare(visibility))
                self._shallow_clone()
                result = self._gate()
                self.assertEqual(result.returncode, expected, result.stderr)
                if visibility == "public":
                    self.assertNotIn("shallow", result.stderr)

    # -- a history the gate cannot see whole is refused ------------------------

    def _deleted_public_history(self) -> None:
        """Commit a public declaration, delete it, then commit again (no denylist)."""
        self._run(self._declare("public"))
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "remove declaration")
        self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "later")

    def test_partial_clone_cannot_launder_a_deleted_public_declaration(self) -> None:
        """A --filter=blob:none clone lacks the declaration blob: "missing" is not absence."""
        self._deleted_public_history()
        self._git("config", "uploadpack.allowFilter", "true")
        source = self.root
        clone = Path(self._tmp.name) / f"partial{self._count}"
        subprocess.run(
            [
                _GIT,
                "clone",
                "-q",
                "--filter=blob:none",
                "--no-local",
                source.resolve().as_uri(),
                str(clone),
            ],
            check=True,
            env=self._env(),
            capture_output=True,
        )
        self.root = clone
        # Without the promisor remote nothing can be lazily fetched: the blob is gone.
        self._git("remote", "remove", "origin")
        result = self._gate()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_grafts_and_replace_refs_cannot_cut_away_a_public_ancestor(self) -> None:
        """Rewriting the parent graph locally must not turn the history into a safe root."""
        for how in ("replace --graft", "info/grafts"):
            with self.subTest(how=how):
                self._fresh()
                self._deleted_public_history()
                if how == "replace --graft":
                    self._git("replace", "--graft", "HEAD")
                else:
                    head = self._rev("HEAD")
                    grafts = self.root / ".git" / "info" / "grafts"
                    grafts.parent.mkdir(parents=True, exist_ok=True)
                    grafts.write_text(head + "\n", encoding="ascii")
                result = self._gate()
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def _rev(self, rev: str) -> str:
        """Resolve *rev* to a commit id in the fixture repo."""
        return subprocess.run(
            [_GIT, "rev-parse", rev],
            cwd=self.root,
            check=True,
            env=self._env(),
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_an_unresolvable_head_with_history_is_not_unborn(self) -> None:
        """A broken HEAD or an orphan branch is not the first commit of a fresh repo."""
        for how in ("broken symref", "orphan branch"):
            with self.subTest(how=how):
                self._fresh()
                self._deleted_public_history()
                if how == "broken symref":
                    self._git("symbolic-ref", "HEAD", "refs/heads/does-not-exist")
                else:
                    self._git("checkout", "-q", "--orphan", "fresh-start")
                self.assertEqual(self._gate().returncode, 1)
                result = self._gate("--staged")
                self.assertEqual(result.returncode, 1, result.stderr)

    def test_the_first_commit_of_a_fresh_repo_is_still_never_opted_in(self) -> None:
        """Control: no commits at all, nothing staged as a declaration -> skip."""
        (self.root / "README.md").write_text("first\n", encoding="utf-8")
        self._git("add", "README.md")
        result = self._gate("--staged")
        self.assertEqual(result.returncode, 0, result.stderr)

    def _forge_commit_graph_roots(self) -> None:
        """Write a commit-graph, blank every commit's parents, and re-seal its checksum."""
        self._git("commit-graph", "write", "--reachable")
        path = self.root / ".git" / "objects" / "info" / "commit-graph"
        data = bytearray(path.read_bytes())
        hash_len = 20 if data[5] == 1 else 32
        chunks: dict[bytes, int] = {}
        for i in range(data[6] + 1):
            entry = 8 + 12 * i
            chunks[bytes(data[entry : entry + 4])] = int.from_bytes(data[entry + 4 : entry + 12])
        count = int.from_bytes(data[chunks[b"OIDF"] + 255 * 4 : chunks[b"OIDF"] + 256 * 4])
        for n in range(count):
            row = chunks[b"CDAT"] + n * (hash_len + 16) + hash_len
            data[row : row + 8] = (0x70000000).to_bytes(4) * 2  # GRAPH_PARENT_NONE x2
        body = bytes(data[:-hash_len])
        # Git's own object-format checksum, not a security use.
        if hash_len == 20:
            digest = hashlib.sha1(body, usedforsecurity=False)
        else:
            digest = hashlib.sha256(body)
        path.chmod(0o644)
        path.write_bytes(body + digest.digest())

    def test_a_forged_commit_graph_cannot_cut_away_a_public_ancestor(self) -> None:
        """A checksum-valid commit-graph whose parent slots were blanked is not history."""
        self._deleted_public_history()
        self._forge_commit_graph_roots()
        count = subprocess.run(
            [_GIT, "rev-list", "--count", "HEAD"],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(count, "1")  # the fixture really does fool plain git
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 1, result.stderr)

    def test_an_object_whose_bytes_do_not_match_its_id_is_not_trusted(self) -> None:
        """A loose public-declaration blob rewritten to say private is corruption, not history."""
        self._deleted_public_history()
        oid = self._rev("HEAD~2:publication.toml")
        loose = self.root / ".git" / "objects" / oid[:2] / oid[2:]
        body = self._declare("private-until-review").encode()
        loose.chmod(0o644)
        loose.write_bytes(zlib.compress(b"blob %d\0" % len(body) + body))
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 1, result.stderr)

    def test_dangling_history_under_a_broken_head_is_not_unborn(self) -> None:
        """HEAD unresolvable and every ref deleted: the commits are still there."""
        self._deleted_public_history()
        branch = subprocess.run(
            [_GIT, "symbolic-ref", "HEAD"],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self._git("symbolic-ref", "HEAD", "refs/heads/unborn")
        self._git("update-ref", "-d", branch)
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 1, result.stderr)

    def test_an_orphan_branch_with_its_own_commit_is_still_the_public_repo(self) -> None:
        """A fresh root cut from a public repo is judged with the repo's other refs."""
        self._deleted_public_history()
        self._git("checkout", "-q", "--orphan", "fresh-start")
        self._git("commit", "-q", "--no-verify", "-m", "fresh root, no declaration")
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 1, result.stderr)

    def test_an_orphan_branch_in_a_private_repo_is_still_fine(self) -> None:
        """Control: when every ref's last declaration is cleanly private, absence skips."""
        self._run(self._declare("private-until-review"))
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "opt out")
        self._git("checkout", "-q", "--orphan", "fresh-start")
        self._git("commit", "-q", "--no-verify", "--allow-empty", "-m", "fresh root")
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 0, result.stderr)

    def _break_head_and_drop_refs(self) -> None:
        """Point HEAD at a missing branch and delete every ref (history left dangling)."""
        refs = subprocess.run(
            [_GIT, "for-each-ref", "--format=%(refname)"],
            cwd=self.root,
            env=self._env(),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        self._git("symbolic-ref", "HEAD", "refs/heads/unborn")
        for ref in refs:
            self._git("update-ref", "-d", ref)

    def test_a_damaged_object_store_is_not_an_unborn_repo(self) -> None:
        """Commits disguised as blobs, a pack without its index, or a foreign object dir."""
        for how in ("commits rewritten as blobs", "pack index removed", "GIT_OBJECT_DIRECTORY"):
            with self.subTest(how=how):
                self._fresh()
                self._deleted_public_history()
                objects = self.root / ".git" / "objects"
                commits = subprocess.run(
                    [_GIT, "rev-list", "--all"],
                    cwd=self.root,
                    env=self._env(),
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
                if how == "pack index removed":
                    self._git("repack", "-adq")
                    self._git("prune-packed")
                self._break_head_and_drop_refs()
                env_extra: dict[str, str] = {}
                if how == "commits rewritten as blobs":
                    for oid in commits:
                        loose = objects / oid[:2] / oid[2:]
                        loose.chmod(0o644)
                        loose.write_bytes(zlib.compress(b"blob 1\0x"))
                elif how == "pack index removed":
                    for idx in (objects / "pack").glob("*.idx"):
                        idx.chmod(0o644)
                        idx.unlink()
                else:
                    empty = Path(self._tmp.name) / f"empty-objects{self._count}"
                    empty.mkdir()
                    env_extra["GIT_OBJECT_DIRECTORY"] = str(empty)
                for args in ((), ("--staged",)):
                    env = self._env()
                    env.update(env_extra)
                    result = subprocess.run(
                        [sys.executable, str(GATE), *args],
                        cwd=self.root,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 1, result.stderr)

    def test_old_public_refs_and_stashes_do_not_block_a_clean_opt_out(self) -> None:
        """public, then cleanly private, then removed: a public-era tag or stash is superseded."""
        for keep in ("tag at the public commit", "stash made while public"):
            with self.subTest(keep=keep):
                self._fresh()
                self._run(self._declare("public"))
                if keep == "tag at the public commit":
                    self._git("tag", "v-public")
                else:
                    (self.root / "README.md").write_text("work in progress\n", encoding="utf-8")
                    self._git("stash", "push", "-q")
                self._run(self._declare("private-until-review"))
                self._git("rm", "-q", "publication.toml")
                self._git("commit", "-q", "--no-verify", "-m", "opt out")
                for args in ((), ("--staged",)):
                    result = self._gate(*args)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_public_side_branch_still_counts(self) -> None:
        """A ref that is NOT an ancestor of the opt-out (here a public side branch) is a tip."""
        self._run(self._declare("public"))
        self._git("branch", "still-public")
        self._run(self._declare("private-until-review"))
        self._git("checkout", "-q", "still-public")
        (self.root / "side.txt").write_text("side\n", encoding="utf-8")
        self._run(None)
        self._git("checkout", "-q", "-")
        self._git("rm", "-q", "publication.toml")
        self._git("commit", "-q", "--no-verify", "-m", "opt out")
        for args in ((), ("--staged",)):
            result = self._gate(*args)
            self.assertEqual(result.returncode, 1, result.stderr)


# The subclass exists for its own tests; do not run the inherited ones twice.
for _name in [n for n in vars(VisibilityContract) if n.startswith("test_")]:
    setattr(DeclarationSourceContract, _name, None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
