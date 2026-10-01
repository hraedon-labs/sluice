"""Mechanical gate against committing work-domain identifiers.

Two complementary checks:

1. Always-on (no configuration): no tracked file may live under ``samples/``.
   ``.gitignore`` is advisory — ``git add -f`` bypasses it — so this guard makes
   an accidental force-add of a real identifier-bearing data file fail CI. The
   ``samples/`` directory holds real environment data (hostnames, service
   accounts, principal handles) that must never be committed (AGENTS.md).

2. Secret-driven: when ``FORBIDDEN_IDENTIFIERS`` is set (a
   whitespace-separated list of real identifiers — hostnames, emails, service
   accounts, principal handles, personal names), every tracked text file
   outside ``samples/`` is scanned for those identifiers. This catches real
   names that leaked into docs, tests, or reflections.

   Unconfigured behaviour depends on ``publication.toml``. In a repo declaring
   ``private-until-review`` (or with no declaration at all) a missing secret is a
   no-op (exit 0), so a fresh clone or a fork without the secret is never
   blocked. In a repo declaring ``visibility = "public"`` it is a **failure**
   (exit 1): an unconfigured gate there prints "skipping" and exits 0, which is
   indistinguishable from a clean tree — a silent pass on exactly the repos where
   a leak is irreversible. That asymmetry was documented in publication.toml for
   months before it was implemented here.

   **Multi-word identifiers are double-quoted** (``"two words"``) and match any
   separator run — spaced, hyphenated, underscored, dotted, or wrapped across a
   line break. Before this, the parser split unconditionally on whitespace, so a
   multi-word identifier could not be expressed at all: its halves became short
   tokens that the length filter dropped. A real two-word work-domain name sat
   undetected in sixteen repositories — eight of them public — because of that
   blind spot. Any denylist entry containing a space must stay quoted.

Run locally: python scripts/check_committed_identifiers.py
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

MIN_IDENTIFIER_LENGTH = 4
# Separators a multi-word identifier may be written with. A two-word domain name
# appears in the wild as "two words", "two-words", "two_words", "two.words", and
# — in wrapped prose — with a line break between the words. A phrase entry
# matches all of those forms; see _phrase_pattern.
_PHRASE_SEPARATOR = r"[\s._\-]+"
_BINARY_SNIFF_LEN = 8192
# Dirs skipped by the identifier scan: .venv is build output. The always-on
# guard below handles root-level samples/ (which holds real identifier-bearing
# data); nested directories named samples/ (e.g. tests/samples/) are legitimate
# code dirs and SHOULD be scanned.
_SKIP_DIRS = frozenset({".venv"})
# Root-level gitignored data dirs that must never contain a tracked file. The
# guard matches the first path component so a legitimate nested code dir named
# ``samples`` (e.g. ``tests/samples/``) is not a false positive.
_GUARDED_DIRS = frozenset({"samples"})

# The plumbing declaration. Read here for ONE purpose: deciding whether an
# unconfigured gate is a benign no-op or a silent pass. check_publication_plumbing.py
# remains the authority on everything else in this file.
_DECLARATION_FILENAME = "publication.toml"


@dataclass(frozen=True)
class Violation:
    identifier: str
    path: Path
    line_number: int
    line: str


class GateError(Exception):
    """A condition that prevents the gate from judging the tree.

    Raised instead of letting a traceback escape: a publication gate that cannot
    complete its scan must fail *clean* (exit 1), never look like a pass and
    never bury the reason in a stack trace.
    """


def _filter_identifiers(identifiers: frozenset[str]) -> frozenset[str]:
    """Lowercase, collapse internal whitespace, drop empty or short identifiers.

    Internal whitespace is collapsed to a single space so a phrase entry is
    normalized regardless of how it was spaced in the denylist; scan_text then
    matches any separator run.
    """
    return frozenset(
        " ".join(token.lower().split())
        for token in (i.strip() for i in identifiers)
        if len(" ".join(token.split())) >= MIN_IDENTIFIER_LENGTH
    )


def parse_identifier_set(raw: str) -> frozenset[str]:
    """Build a normalized set of identifiers from the raw denylist.

    Accepts whitespace-separated tokens (the CI-secret form) and/or one token
    per line. Full-line and trailing ``#`` comments are stripped, so a
    human-maintained denylist file may document itself without every comment
    word becoming a forbidden token.

    **Multi-word identifiers must be double-quoted** (``"two words"``). Before
    this, the parser split unconditionally on whitespace, so a multi-word
    identifier could not be *expressed* — the two halves became two short
    tokens, each dropped by the length filter. A real two-word work-domain name
    sat undetected in sixteen repositories because of that. Quoted entries are
    kept whole and matched with a flexible separator (see scan_text). The audit
    that found the leak is recorded in docs/publication-review.md.

    Raises ValueError on unbalanced quoting: a denylist we cannot parse must
    fail the gate loudly, never degrade to a partial token set.
    """
    tokens: set[str] = set()
    for line in raw.splitlines() or [raw]:
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        try:
            tokens.update(shlex.split(content))
        except ValueError as exc:  # unbalanced quote
            raise ValueError(f"denylist entry could not be parsed (check quoting): {exc}") from exc
    return _filter_identifiers(frozenset(tokens))


def _phrase_pattern(identifier: str) -> re.Pattern[str]:
    """Compile a multi-word identifier into a flexible-separator regex.

    Internal whitespace matches any run of whitespace, ``.``, ``_``, or ``-``,
    so one denylist entry covers the spaced, hyphenated, underscored, dotted,
    and line-wrapped spellings. Everything else is escaped literally.
    """
    parts = [re.escape(word) for word in identifier.split()]
    return re.compile(_PHRASE_SEPARATOR.join(parts), re.IGNORECASE)


def scan_text(text: str, identifiers: frozenset[str]) -> Iterator[Violation]:
    """Yield a violation for every occurrence of one of *identifiers*.

    The match is case-insensitive and counts any substring occurrence; real
    identifiers such as ``WORK-DOMAIN`` can legitimately appear inside longer
    tokens.

    Single-word identifiers are matched line by line. Multi-word identifiers are
    matched against the whole text with a flexible separator, so a phrase that
    prose wrapped across a line break is still caught; the reported line is the
    one the match starts on.
    """
    identifiers = _filter_identifiers(identifiers)
    if not identifiers:
        return
    words = frozenset(i for i in identifiers if " " not in i)
    phrases = frozenset(i for i in identifiers if " " in i)

    lines = text.splitlines()
    for line_number, line in enumerate(lines, start=1):
        lower = line.lower()
        for identifier in words:
            start = 0
            while True:
                offset = lower.find(identifier, start)
                if offset == -1:
                    break
                yield Violation(
                    identifier=identifier,
                    path=Path("."),
                    line_number=line_number,
                    line=line,
                )
                start = offset + len(identifier)

    if not phrases:
        return
    for identifier in phrases:
        for match in _phrase_pattern(identifier).finditer(text):
            line_number = text.count("\n", 0, match.start()) + 1
            yield Violation(
                identifier=identifier,
                path=Path("."),
                line_number=line_number,
                line=lines[line_number - 1] if line_number <= len(lines) else "",
            )


def _sniff_encoding(chunk: bytes) -> str | None:
    """Return the text encoding if *chunk* starts with a known BOM, else None."""
    if chunk.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if chunk.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if chunk.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    return None


def _is_binary(chunk: bytes) -> bool:
    """Heuristic: null byte present without a recognized text BOM → binary."""
    if _sniff_encoding(chunk) is not None:
        return False
    return b"\x00" in chunk


def scan_files(
    identifiers: frozenset[str],
    paths: list[Path],
    *,
    unreadable: list[Path] | None = None,
) -> list[Violation]:
    """Scan every readable text file in *paths* for forbidden identifiers.

    UTF-16 files (common in Windows tooling output) are detected via BOM and
    decoded correctly rather than misclassified as binary by the null-byte
    heuristic.

    Returns the violations. A tracked file the gate could not read is collected
    into *unreadable* when a list is supplied (WI-027): silently skipping an
    unreadable file lets one containing a forbidden identifier pass, which is
    precisely the fails-open case this gate exists to prevent.

    The out-parameter is deliberate. This script is COPIED into every repo in the
    estate and several of them test ``scan_files`` directly, so returning a tuple
    instead of a list broke seven repositories' test suites at once. An optional
    keyword collector keeps the signature backward compatible while still letting
    the CLI fail closed on an unreadable file.
    """
    violations: list[Violation] = []
    if unreadable is None:
        unreadable = []
    for path in paths:
        # A tracked symlink's blob content is its target path, not file data.
        # Scan the target string without following the link: following it either
        # leaves the repo (wrong thing to scan) or fails on a broken link and
        # looks like an unreadable file. The target itself can carry a forbidden
        # identifier, so it is scanned rather than skipped.
        if path.is_symlink():
            target = os.readlink(path)
            for violation in scan_text(target, identifiers):
                violations.append(replace(violation, path=path, line=target))
            continue
        try:
            with path.open("rb") as f:
                chunk = f.read(_BINARY_SNIFF_LEN)
        except OSError:
            unreadable.append(path)
            continue
        if _is_binary(chunk):
            continue
        encoding = _sniff_encoding(chunk) or "utf-8"
        try:
            text = path.read_text(encoding=encoding, errors="replace")
        except OSError:
            unreadable.append(path)
            continue
        for violation in scan_text(text, identifiers):
            violations.append(replace(violation, path=path))
    return violations


def _run_git(args: list[str]) -> str:
    """Run a git command and return stdout.

    A git failure raises GateError so the gate exits 1 with a readable reason
    (WI-027): a CI gate must fail clean, not emit a CalledProcessError traceback
    that reads as an infrastructure crash rather than a blocked publication.
    """
    # argv is passed through verbatim, bare "git" included. Resolving it to an
    # absolute path via shutil.which is arguably better hygiene, but this script is
    # COPIED into every repo and several of them assert on the exact argv
    # (`== ["git", "diff", "--cached"]`), so absolute paths broke three test
    # suites. The S607 partial-path finding that motivated it only ever applied to
    # check_publication_plumbing.py, whose literal argv ruff can see statically;
    # here the list is a parameter, so the rule does not fire. Respect the
    # fleet-wide contract.
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise GateError(
            f"git command failed ({' '.join(args)}): "
            f"exit {exc.returncode}: {(exc.stderr or '').strip()}"
        ) from exc
    except OSError as exc:
        raise GateError(f"could not run git ({' '.join(args)}): {exc}") from exc
    return result.stdout


def _paths_from_git(args: list[str]) -> list[Path]:
    """Run a NUL-delimited git path command and return Paths.

    No filtering is applied here — the always-on samples/ guard needs to see
    every tracked path so it can detect a force-add. The identifier scan
    filters out _SKIP_DIRS separately.
    """
    paths: list[Path] = []
    for raw in _run_git(args).split("\0"):
        if not raw:
            continue
        paths.append(Path(raw))
    return paths


def collect_tracked_paths() -> list[Path]:
    """Return tracked file paths from ``git ls-files``, excluding obvious skips."""
    return _paths_from_git(["git", "ls-files", "-z"])


def collect_range_messages(rev_range: str) -> list[tuple[str, str]]:
    """Return ``(sha, message)`` for every commit in *rev_range*.

    Commit messages are a publication channel the content gate never covered:
    the tracked-tree scan reads files, so an identifier named only in a message
    is invisible to it. That blind spot is not hypothetical — a public repo
    carried work-domain identifiers in three commit messages, two of which were
    the very commits that redacted those identifiers from the files. The message
    described what the diff removed.
    """
    # rev_range may carry several git-log arguments (the pre-push new-branch case
    # passes "<sha> --not --remotes=<name>"), so it is split rather than passed
    # as one opaque argument.
    result = _run_git(
        ["git", "log", "--format=%H%x1f%B%x1e", *rev_range.split()],
    )
    messages: list[tuple[str, str]] = []
    for record in result.split("\x1e"):
        if "\x1f" not in record:
            continue
        sha, body = record.split("\x1f", 1)
        messages.append((sha.strip(), body))
    return messages


def collect_staged_paths() -> list[Path]:
    """Return staged (added/copied/modified/renamed) paths for the pre-commit hook.

    Scans only what is about to be committed rather than the whole tree, so the
    local gate is fast enough to run on every commit. Deletions are excluded
    (``--diff-filter=ACMT``) because there is nothing to scan. Type changes (T) ARE
    included: re-staging a regular file as a symlink whose target names a forbidden
    identifier is otherwise invisible to the hook. ``--no-renames``
    decomposes renames into add+delete so the new path (e.g. a file moved into
    ``samples/``) is included as an addition and caught by the always-on guard.
    """
    return _paths_from_git(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMT",
            "--no-renames",
            "-z",
        ]
    )


def print_report(violations: list[Violation]) -> None:
    violations.sort(key=lambda v: (str(v.path), v.line_number, v.identifier))
    print("Committed identifier violations detected:", file=sys.stderr)
    for v in violations:
        print(f"  {v.path}:{v.line_number}: {v.identifier!r}", file=sys.stderr)
        print(f"      {v.line.rstrip()}", file=sys.stderr)
    print(f"\nTotal: {len(violations)} violation(s)", file=sys.stderr)


def leaked_tracked_files(paths: list[Path], guarded: frozenset[str]) -> list[Path]:
    """Tracked files whose root component is a guarded (gitignored) data dir.

    Matches only the first path component so a nested code directory that happens
    to be named ``samples`` (e.g. ``tests/samples/``) is not a false positive.
    """
    return [p for p in paths if p.parts and p.parts[0] in guarded]


# Set by main() from --staged. In staged mode the publication verdict must come
# from the INDEX -- the bytes the commit records -- not the worktree: otherwise a
# commit that stages visibility="public" while the worktree still says
# "private-until-review" (the publication flip, exactly where this matters) is
# judged private and skipped. A one-element list so main() can set it without a
# global statement.
_DECLARATION_FROM_INDEX: list[bool] = [False]


def _staged_declaration_text() -> str | None:
    """The stage-0 index content of the declaration, or None if it is not staged.

    Absence from the index is the only None. A conflicted entry, a non-regular
    entry (symlink, submodule) or an undecodable blob is a GateError: those are
    present-but-unreadable, not "never opted in".
    """
    listing = _run_git(
        ["git", "ls-files", "--stage", "-z", "--", f":(top,literal){_DECLARATION_FILENAME}"]
    )
    entries = [e for e in listing.split("\0") if e]
    if not entries:
        return None
    if len(entries) != 1:
        raise GateError(
            f"{_DECLARATION_FILENAME} has a conflicted index entry; the gate cannot "
            "tell whether this repo is public, so it will not pass."
        )
    meta = entries[0].split("\t", 1)[0].split()
    if len(meta) != 3 or meta[2] != "0" or meta[0] not in ("100644", "100755"):
        raise GateError(
            f"{_DECLARATION_FILENAME} is staged but is not a regular file; the gate "
            "cannot tell whether this repo is public, so it will not pass."
        )
    # Bytes, decoded as UTF-8 here: _run_git decodes with the LOCALE codec, which
    # on Windows (cp1252) accepts any byte and would hide a non-UTF-8 blob.
    blob_argv = ["git", "cat-file", "blob", meta[1]]
    try:
        blob = subprocess.run(blob_argv, capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        raise GateError(
            f"could not read the staged {_DECLARATION_FILENAME} ({exc}); the gate "
            "cannot tell whether this repo is public, so it will not pass."
        ) from exc
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(
            f"the staged {_DECLARATION_FILENAME} is not valid UTF-8 ({exc}); the gate "
            "cannot tell whether this repo is public, so it will not pass."
        ) from exc


def _git_or_none(args: list[str]) -> str | None:
    """Return the stdout of a git command, or None when it fails (optional lookups)."""
    # No UnicodeDecodeError arm. The only caller asks `rev-parse --verify -q HEAD`,
    # whose stdout is a hex object id or nothing and whose stderr -q silences, so
    # no locale codec can fail on it. An arm here used to return a non-None
    # placeholder; were it ever reached, that would CONTINUE the absence verdict,
    # so dropping it can only turn such a case into an exit 1, never a pass. Do
    # not translate UnicodeDecodeError into GateError either: here that would read
    # as "no HEAD", which is the never-opted-in skip.
    try:
        return _run_git(args)
    except GateError:
        return None


def _text_declares_private(text: str) -> bool:
    """True only when a declaration cleanly names "private-until-review".

    Anything else -- public, an unknown value, a missing key, unparseable text --
    is not a safe last word before a deletion: a public -> garbage -> delete
    sequence would otherwise launder a public declaration into "never opted in".
    """
    try:
        section = tomllib.loads(text).get("publication")
    except tomllib.TOMLDecodeError:
        return False
    declared = section.get("visibility") if isinstance(section, dict) else None
    if not isinstance(declared, str):
        return False
    return declared.strip().casefold() == "private-until-review"


def _absent_declaration_verdict() -> bool:
    """Verdict for a repo whose declaration is ABSENT: False, unless that is unsafe.

    Absence is the "never opted in" skip. But deleting a declaration that said
    public does not make the remote private: it only disarms the gate. So the
    whole history decides, not one log query: a commit WITH a declaration is safe
    only if it cleanly says "private-until-review"; a commit WITHOUT one is safe
    only if every parent is safe (a root commit without one is safe -- never opted
    in). The state being judged (the index or worktree with no declaration) has
    HEAD as its parent, so absence is a skip exactly when HEAD is safe. This covers
    plain and merge deletions, laundering through an invalid declaration, and a
    merge that joins an unsafe absent lineage to a private one. To leave the
    publication system, declare "private-until-review" and remove the file in a
    later commit.

    A shallow clone is refused, not judged: history beyond the graft is invisible
    and the graft would count as a root, so a declaration deleted in an unfetched
    commit read as "never opted in" -- exit 0 on a public repo. actions/checkout
    is shallow by default, so this was the CI default, not a corner case.
    """
    if _git_or_none(["git", "rev-parse", "--verify", "-q", "HEAD"]) is None:
        return False
    # Anything but a literal "false" (including a git too old to know the flag,
    # which echoes it back) is treated as shallow.
    if _run_git(["git", "rev-parse", "--is-shallow-repository"]).strip() != "false":
        raise GateError(
            f"{_DECLARATION_FILENAME} is absent and this is a shallow clone, so the "
            "history that decides whether it was ever declared public cannot be "
            "read; the gate will not treat it as never opted in. Fetch full "
            "history (git fetch --unshallow; in CI, actions/checkout with "
            "fetch-depth: 0), or restore the declaration."
        )
    graph: dict[str, list[str]] = {}
    order: list[str] = []
    for line in _run_git(["git", "rev-list", "--topo-order", "--parents", "HEAD"]).splitlines():
        if line.strip():
            commit, *parents = line.split()
            graph[commit] = parents
            order.append(commit)
    head = order[0]
    # Bare "git", as _run_git passes it: the argv is a variable, as there.
    # %(objectmode) (git >= 2.45): a symlink or gitlink at publication.toml is
    # reported as a blob too, and its target string could read as a private
    # declaration. Only a regular file is a declaration, as on every other path.
    check_argv = ["git", "cat-file", "--batch-check=%(objectmode) %(objecttype) %(objectname)"]
    batch_argv = ["git", "cat-file", "--batch"]
    try:
        check_proc = subprocess.run(
            check_argv,
            input="".join(f"{c}:{_DECLARATION_FILENAME}\n" for c in order).encode(),
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise GateError(f"could not read the declaration history ({exc})") from exc
    checked = check_proc.stdout.decode("utf-8", "replace").splitlines()
    if len(checked) != len(order):
        raise GateError("could not read the declaration history (short batch-check output)")
    present: dict[str, str | None] = {}
    for commit, row in zip(order, checked, strict=True):
        fields = row.split()
        if row.endswith(" missing"):
            present[commit] = None
        elif len(fields) != 3 or fields[0] not in ("100644", "100755") or fields[1] != "blob":
            present[commit] = ""  # present but not a regular file: never a clean private
        else:
            present[commit] = fields[2]
    blobs = sorted({oid for oid in present.values() if oid})
    private_blob: dict[str, bool] = {}
    if blobs:
        try:
            out = subprocess.run(
                batch_argv,
                input="".join(f"{oid}\n" for oid in blobs).encode(),
                capture_output=True,
                check=True,
            ).stdout
        except (subprocess.CalledProcessError, OSError) as exc:
            raise GateError(f"could not read the declaration history ({exc})") from exc
        pos = 0
        for oid in blobs:
            header_end = out.index(b"\n", pos)
            size = int(out[pos:header_end].split()[2])
            data = out[header_end + 1 : header_end + 1 + size]
            pos = header_end + 1 + size + 1
            try:
                private_blob[oid] = _text_declares_private(data.decode("utf-8"))
            except UnicodeDecodeError:
                private_blob[oid] = False
    safe: dict[str, bool] = {}
    for commit in reversed(order):  # --topo-order lists children first
        entry = present[commit]
        if entry is not None:
            safe[commit] = bool(entry) and private_blob.get(entry, False)
        else:
            safe[commit] = all(safe.get(p, True) for p in graph[commit])
    if safe[head]:
        return False
    raise GateError(
        f"{_DECLARATION_FILENAME} is absent, but this history declared a visibility "
        'other than "private-until-review" without a later clean private '
        "declaration; removing a declaration does not make the remote private, so "
        "the gate will not treat it as never opted in. Restore it, or declare "
        '"private-until-review" before removing it.'
    )


def _declares_public() -> bool:
    """True when this repo's publication.toml declares public visibility.

    Governs whether a missing denylist is a no-op or a hard failure. The
    distinction is the whole point: a private-until-review repo must stay
    clonable and committable without the secret, but a PUBLIC repo whose gate is
    unconfigured is a silent pass — the scan prints "skipping" and exits 0, and
    nothing downstream can tell that apart from a clean tree.

    Absence of the file is False (fail-open): a repo that never opted into the
    publication system is not suddenly blocked. A file that is PRESENT but
    unparseable is a GateError, not False — that repo did opt in, and guessing
    its visibility is exactly the coin-flip this function exists to remove.
    """
    try:
        repo_root = Path(_run_git(["git", "rev-parse", "--show-toplevel"]).strip())
    except GateError as exc:
        # Fail closed. This used to return False ("not public"), which turned a
        # broken or missing git into a skip in --message-file / --rev-range mode:
        # those modes return straight after the verdict, so nothing later
        # surfaced the error and a public repo exited 0 having scanned nothing.
        raise GateError(
            "could not resolve the repository root, so the gate cannot read the "
            f"publication declaration and will not pass: {exc}"
        ) from exc

    text: str | None
    if _DECLARATION_FROM_INDEX[0]:
        text = _staged_declaration_text()
        if text is None:
            return _absent_declaration_verdict()
    else:
        path = repo_root / _DECLARATION_FILENAME
        # Only genuine absence is the "never opted in" skip. A path that exists
        # but is not a regular file (a directory, or a symlink -- dangling or
        # not) used to take the same branch via `not path.is_file()`, so a
        # stray directory or link silently disarmed a public repo's gate.
        if not os.path.lexists(path):
            return _absent_declaration_verdict()
        if path.is_symlink() or not path.is_file():
            raise GateError(
                f"{_DECLARATION_FILENAME} is present but is not a regular file; the "
                "gate cannot tell whether this repo is public, so it will not pass."
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GateError(
                f"{_DECLARATION_FILENAME} is present but could not be read ({exc}); "
                "the gate cannot tell whether this repo is public, so it will not pass."
            ) from exc
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise GateError(
            f"{_DECLARATION_FILENAME} is present but could not be parsed ({exc}); "
            "the gate cannot tell whether this repo is public, so it will not pass."
        ) from exc

    section = raw.get("publication")
    if not isinstance(section, dict):
        raise GateError(
            f"{_DECLARATION_FILENAME} has no [publication] table; the gate cannot "
            "tell whether this repo is public, so it will not pass."
        )
    # The set of legal visibilities is closed, and owned by Visibility in
    # check_publication_plumbing.py. Compare against it rather than against the
    # bare string "public": anything outside the set is a GateError, not a quiet
    # "not public".
    #
    # This used to be `str(section.get("visibility", "")).strip() == "public"`,
    # which coerced every other shape into the fail-OPEN branch. On a public repo
    # that silently disarmed the gate -- the exact "nothing was scanned and CI is
    # green" failure this function exists to prevent. A wrong-cased value
    # ("Public"), a missing visibility key, an empty string, and a non-string
    # (visibility = true) all took that branch. The declaration flip is precisely
    # the commit where this matters: a case typo or a dropped line in the
    # publication-review commit disarmed every later run.
    if "visibility" not in section:
        raise GateError(
            f"{_DECLARATION_FILENAME} has a [publication] table but no visibility "
            "key; the gate cannot tell whether this repo is public, so it will "
            "not pass."
        )
    declared = section["visibility"]
    if not isinstance(declared, str):
        raise GateError(
            f"{_DECLARATION_FILENAME} declares visibility={declared!r}, which is "
            f"{type(declared).__name__} rather than a string; the gate cannot tell "
            "whether this repo is public, so it will not pass."
        )
    normalised = declared.strip().casefold()
    if normalised == "public":
        return True
    if normalised == "private-until-review":
        return False
    raise GateError(
        f"{_DECLARATION_FILENAME} declares visibility={declared!r}, which is not "
        'one of "public" or "private-until-review"; the gate cannot tell whether '
        "this repo is public, so it will not pass."
    )


def _unconfigured(reason: str) -> None:
    """Handle a denylist that is unset or unusable.

    Returns quietly (caller no-ops) for a non-public repo; raises GateError for a
    public one.
    """
    if _declares_public():
        raise GateError(
            f'{reason} but {_DECLARATION_FILENAME} declares visibility="public". '
            "A public repo with an unconfigured gate is a silent pass, so this is "
            "a failure, not a skip. Provide the denylist via the FORBIDDEN_IDENTIFIERS "
            "environment variable (in CI, the shared organisation secret of that name)."
        )
    print(f"{reason}; skipping identifier gate.", file=sys.stderr)


def _resolve_identifiers() -> frozenset[str] | None:
    """Return the configured denylist, or None if the gate should no-op.

    Shared by the message-scanning modes so they honor exactly the same
    configured/unconfigured semantics as the tracked-tree scan.
    """
    raw = os.environ.get("FORBIDDEN_IDENTIFIERS", "")
    if not raw.strip():
        _unconfigured("FORBIDDEN_IDENTIFIERS is empty or unset")
        return None
    identifiers = parse_identifier_set(raw)
    if not identifiers:
        _unconfigured(
            "FORBIDDEN_IDENTIFIERS contained no usable identifiers "
            f"(minimum length is {MIN_IDENTIFIER_LENGTH} characters)"
        )
        return None
    return identifiers


def _report_message_violations(label: str, violations: list[Violation]) -> None:
    print(f"Forbidden identifier in {label}:", file=sys.stderr)
    for v in sorted(violations, key=lambda v: (v.line_number, v.identifier)):
        print(f"  line {v.line_number}: {v.identifier!r}", file=sys.stderr)
        print(f"      {v.line.rstrip()}", file=sys.stderr)
    print(
        "\nA commit message is published with the commit. Rewrite the message "
        "without the identifier (the canonical denylist is the authority on what "
        "may not appear).",
        file=sys.stderr,
    )


def _scan_message_file(path: Path) -> int:
    """commit-msg hook mode: scan the proposed commit message."""
    identifiers = _resolve_identifiers()
    if identifiers is None:
        return 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise GateError(f"could not read the commit message file {path}: {exc}") from exc
    # git puts everything after a scissors line out of the commit; comment lines
    # are stripped too. Scan only what will actually be recorded.
    kept = [ln for ln in text.splitlines() if not ln.startswith("#")]
    violations = list(scan_text("\n".join(kept), identifiers))
    if violations:
        _report_message_violations("the proposed commit message", violations)
        return 1
    return 0


def _scan_rev_range(rev_range: str) -> int:
    """pre-push mode: scan every commit message about to be published."""
    identifiers = _resolve_identifiers()
    if identifiers is None:
        return 0
    failed = False
    for sha, body in collect_range_messages(rev_range):
        violations = list(scan_text(body, identifiers))
        if violations:
            _report_message_violations(f"commit message {sha[:9]}", violations)
            failed = True
    return 1 if failed else 0


def _scan_index_snapshot(identifiers: frozenset[str], paths: list[Path], **kwargs: Any) -> Any:
    """``scan_files`` over the STAGED (stage-0 index) content of *paths*.

    --staged is the pre-commit hook, so it must judge the bytes the commit will
    record. Scanning the worktree let a staged forbidden identifier hide behind a
    clean unstaged copy (stage it, then overwrite the file) -- and blocked clean
    commits whose worktree held unstaged junk. The stage-0 blobs are read RAW with
    ``git cat-file --batch`` -- no smudge filter, no EOL conversion, exactly the
    bytes the commit records (``checkout-index`` applies both, so a smudge filter
    could strip a token from the scanned copy) -- written into a private temporary
    directory, and scanned by the unchanged ``scan_files``, so binary/encoding
    handling is the tree scan's. A staged symlink is recreated as a symlink to its
    staged target string, which scan_files scans without following. Reported paths
    are mapped back to repo-relative.

    Fail closed: a path with no stage-0 regular-file or symlink entry (an unmerged
    entry, a gitlink) is left out of the snapshot, which scan_files reports as
    unreadable.
    """
    repo_root = _run_git(["git", "rev-parse", "--show-toplevel"]).strip()
    wanted = {p.as_posix() for p in paths}
    entries: dict[str, tuple[str, str]] = {}
    for record in _run_git(["git", "-C", repo_root, "ls-files", "--stage", "-z"]).split("\0"):
        if not record or "\t" not in record:
            continue
        meta, name = record.split("\t", 1)
        mode, oid, stage = meta.split()
        if name in wanted and stage == "0" and mode in ("100644", "100755", "120000"):
            entries[name] = (mode, oid)
    blobs: dict[str, bytes] = {}
    if entries:
        names = sorted(entries)
        request = "".join(f"{entries[n][1]}\n" for n in names).encode()
        # Bare "git", as _run_git passes it: the argv is a variable, as there.
        batch_argv = ["git", "-C", repo_root, "cat-file", "--batch"]
        try:
            proc = subprocess.run(
                batch_argv,
                input=request,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, OSError) as exc:
            raise GateError(
                f"could not read the staged content to scan it ({exc}); the gate "
                "will not pass a commit it could not fully scan."
            ) from exc
        out, pos = proc.stdout, 0
        for name in names:
            header_end = out.index(b"\n", pos)
            header = out[pos:header_end].split()
            if len(header) != 3 or header[1] != b"blob":
                raise GateError(
                    f"could not read the staged blob for {name!r}; the gate will not "
                    "pass a commit it could not fully scan."
                )
            size = int(header[2])
            blobs[name] = out[header_end + 1 : header_end + 1 + size]
            pos = header_end + 1 + size + 1
    with tempfile.TemporaryDirectory(prefix="identifier-gate-index-") as tmp:
        base = Path(tmp)
        for name, data in blobs.items():
            target = base / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if entries[name][0] == "120000":
                    os.symlink(os.fsdecode(data), target)
                else:
                    target.write_bytes(data)
            except OSError as exc:
                raise GateError(
                    f"could not materialise the staged entry {name!r} to scan it ({exc}); "
                    "the gate will not pass a commit it could not fully scan."
                ) from exc

        def back(p: Path) -> Path:
            try:
                return Path(p).relative_to(base)
            except ValueError:
                return Path(p)

        unreadable = kwargs.get("unreadable")
        result = scan_files(identifiers, [base / p for p in paths], **kwargs)
        if isinstance(unreadable, list):
            unreadable[:] = [back(p) for p in unreadable]
        if isinstance(result, tuple):
            found, missed = result
            return [replace(v, path=back(v.path)) for v in found], [back(p) for p in missed]
        return [replace(v, path=back(v.path)) for v in result]


def _run(args: argparse.Namespace) -> int:
    if args.message_file is not None:
        return _scan_message_file(Path(args.message_file))
    if args.rev_range is not None:
        return _scan_rev_range(args.rev_range)

    paths = collect_staged_paths() if args.staged else collect_tracked_paths()

    # 1. Always-on: no tracked file under a guarded (gitignored) data dir. This
    #    catches a ``git add -f samples/...`` leak regardless of secret config.
    leaked = leaked_tracked_files(paths, _GUARDED_DIRS)
    if leaked:
        print("Tracked files under a gitignored data directory detected:", file=sys.stderr)
        for p in sorted(leaked, key=str):
            print(f"  {p}", file=sys.stderr)
        print(
            "\nThese paths are gitignored by convention (samples/ holds real "
            "identifier-bearing data — hostnames, service accounts, principal "
            "handles). Remove them from the index: git rm --cached -r <path>.",
            file=sys.stderr,
        )
        return 1

    # 2. Secret-driven: scan tracked text files (outside guarded dirs) for
    #    forbidden identifiers. A no-op until the secret is configured — EXCEPT in
    #    a repo declaring public visibility, where _resolve_identifiers raises
    #    rather than let an unconfigured gate report a green pass.
    #
    #    This path used to duplicate the resolver inline, so the tree scan and the
    #    message scans could drift apart in exactly the semantics that matter.
    identifiers = _resolve_identifiers()
    if identifiers is None:
        return 0

    scan_paths = [p for p in paths if not any(part in _SKIP_DIRS for part in p.parts)]
    unreadable: list[Path] = []
    # --staged judges the index blobs (what the commit records), never the worktree.
    scan = _scan_index_snapshot if args.staged else scan_files
    violations = scan(identifiers, scan_paths, unreadable=unreadable)
    if violations:
        print_report(violations)
        return 1
    if unreadable:
        print("Tracked files could not be read; the gate cannot clear them:", file=sys.stderr)
        for p in sorted(unreadable, key=str):
            print(f"  {p}", file=sys.stderr)
        print(
            "\nAn unreadable tracked file may contain a forbidden identifier. Fix the "
            "permissions (or untrack the file) and re-run; the gate will not pass a "
            "tree it could not fully scan.",
            file=sys.stderr,
        )
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate that prevents committing forbidden domain identifiers.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Scan only staged files (for the pre-commit hook) instead of the "
        "full tracked tree (the CI default).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--message-file",
        metavar="PATH",
        help="Scan a proposed commit message (for the commit-msg hook) instead "
        "of files. Comment lines are ignored, as git strips them.",
    )
    mode.add_argument(
        "--rev-range",
        metavar="RANGE",
        help="Scan the commit messages in a git rev range (for the pre-push "
        "hook), e.g. origin/main..HEAD.",
    )
    args = parser.parse_args(argv)
    _DECLARATION_FROM_INDEX[0] = bool(args.staged)

    try:
        return _run(args)
    except GateError as exc:
        print(f"identifier gate could not complete: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Unparseable denylist (bad quoting). Fail closed, loudly.
        print(f"identifier gate denylist is invalid: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
