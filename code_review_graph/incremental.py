"""Incremental graph update logic.

Detects changed files via git diff, re-parses only changed + impacted files,
and updates the graph accordingly. Also supports CLI invocation for hooks.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

from .graph import GraphStore
from .parser import CodeParser, normalize_file_path

_MAX_PARSE_WORKERS = int(os.environ.get("CRG_PARSE_WORKERS", str(min(os.cpu_count() or 4, 8))))

# Set only while the in-process FastMCP server is using stdio transport.
# This is deliberately separate from ``sys.stdin.isatty()``: CI, cron, and
# redirected CLI builds also have non-TTY stdin, but do not share the MCP
# transport's file-descriptor lifetime problem.
_MCP_STDIO_ACTIVE = False

# Each process-pool worker runs this module in its own process, while each
# thread-pool worker needs isolated parser state.  A thread-local cache covers
# both cases and avoids rebuilding CodeParser (including its grammar probes and
# parser caches) for every file in a parallel build.
_PARSE_WORKER_STATE = threading.local()


def _select_executor_kind() -> str:
    """Return 'process' or 'thread' for parallel parsing.

    Defaults to ``process`` (the original behavior, fastest on Linux/macOS).
    Auto-switches to ``thread`` for an active MCP stdio server on every
    platform, where ``ProcessPoolExecutor`` workers can inherit the transport
    pipe/socket and prevent EOF shutdown. The older Windows non-TTY fallback
    remains for direct integrations that predate the explicit transport flag
    (issues #46, #136, PR #615).

    Override explicitly with ``CRG_PARSE_EXECUTOR={process,thread}``.

    Tree-sitter parsing in the worker releases the GIL during native
    parsing, so the speedup loss for falling back to threads is small
    (typically <30% on the full-build path) and the trade is worth it
    to avoid the deadlock + zombie process accumulation.
    """
    explicit = os.environ.get("CRG_PARSE_EXECUTOR", "").strip().lower()
    if explicit in ("process", "thread"):
        return explicit
    if _MCP_STDIO_ACTIVE:
        return "thread"
    if sys.platform == "win32" and not sys.stdin.isatty():
        return "thread"
    return "process"


def _make_executor(max_workers: int):
    """Construct the parallel-parse executor selected by [_select_executor_kind]."""
    if _select_executor_kind() == "thread":
        return concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

logger = logging.getLogger(__name__)

CPP_IDENTITY_VERSION = "1"
_CPP_IDENTITY_METADATA_KEY = "cpp_identity_version"


def _run_python_resolver(store: GraphStore) -> Optional[dict]:
    """Run repository-wide Python import resolution without failing a build."""
    try:
        from .python_resolver import resolve_python_imports
        return resolve_python_imports(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("Python import resolver failed: %s", exc)
        return None


def _run_rescript_resolver(store: GraphStore) -> Optional[dict]:
    """Run the ReScript cross-module resolver, swallowing any failure so
    build never fails because of it. Returns stats or None on error.
    """
    try:
        from .rescript_resolver import resolve_rescript_cross_module
        return resolve_rescript_cross_module(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("ReScript cross-module resolver failed: %s", exc)
        return None


def _run_spring_resolver(store: GraphStore) -> Optional[dict]:
    """Run the Spring DI call resolver, swallowing any failure so
    build never fails because of it. Returns stats or None on error.
    """
    try:
        from .spring_resolver import resolve_spring_di_calls
        return resolve_spring_di_calls(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("Spring DI resolver failed: %s", exc)
        return None


def _run_spring_event_resolver(store: GraphStore) -> Optional[dict]:
    """Run the Spring application-event resolver without failing a build."""
    try:
        from .event_resolver import resolve_spring_events
        return resolve_spring_events(store)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Spring event resolver failed: %s", exc)
        return None


def _run_temporal_resolver(store: GraphStore) -> Optional[dict]:
    """Run the Temporal workflow/activity call resolver, swallowing any failure so
    build never fails because of it. Returns stats or None on error.
    """
    try:
        from .temporal_resolver import resolve_temporal_calls
        return resolve_temporal_calls(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("Temporal resolver failed: %s", exc)
        return None


def _run_hcl_resolver(store: GraphStore) -> Optional[dict]:
    """Run Terraform module-scope resolution without failing a build."""
    try:
        from .hcl_resolver import resolve_hcl_module_references
        return resolve_hcl_module_references(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("Terraform/HCL resolver failed: %s", exc)
        return None


def _run_scoped_resolver(store: GraphStore) -> Optional[dict]:
    """Resolve static/scoped ``Class::method`` calls without failing a build."""
    try:
        from .scoped_resolver import resolve_scoped_calls
        return resolve_scoped_calls(store)
    except Exception as exc:  # noqa: BLE001 - best-effort post-pass
        logger.warning("Scoped call resolver failed: %s", exc)
        return None


# Default ignore patterns (in addition to .gitignore).
#
# ``**/<dir>/**`` patterns are safe-anywhere directory exclusions.  A leading
# slash anchors a pattern to the repository root, which prevents ambiguous
# output names such as ``build`` and ``dist`` from hiding nested source
# directories.  See: #91 and PR #92.
DEFAULT_IGNORE_PATTERNS = [
    "**/.code-review-graph/**",
    "**/node_modules/**",
    "**/.git/**",
    "**/.svn/**",
    "**/__pycache__/**",
    "*.pyc",
    "**/.venv/**",
    "**/venv/**",
    "/dist/**",
    "/build/**",
    "/.next/**",
    "/.nuxt/**",
    "/target/**",
    "/bin/**",
    "/obj/**",
    # PHP / Laravel / Composer
    "**/vendor/**",
    "/storage/**",
    "/bootstrap/cache/**",
    "/public/build/**",
    # Ruby / Bundler
    "**/.bundle/**",
    # Java / Kotlin / Gradle
    "**/.gradle/**",
    "*.jar",
    # Dart / Flutter
    "**/.dart_tool/**",
    "**/.pub-cache/**",
    # AWS CDK
    "**/cdk.out/**",
    # General
    "/coverage/**",
    "**/.cache/**",
    "/.tmp/**",
    "/tmp/**",  # nosec B108 -- repo-relative ignore glob, not a temp-file path
    "*.min.js",
    "*.min.css",
    "*.map",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "*.db",
    "*.sqlite",
    "*.db-journal",
    "*.db-wal",
]

# Build-output directories that ``DEFAULT_IGNORE_PATTERNS`` only anchors at the
# repository root.  A nested copy is ignored as well, but only when a sibling
# manifest proves the directory is that module's build output — ``moduleA/pom.xml``
# next to ``moduleA/target/``.  Without that evidence the nested directory keeps
# being parsed and watched, so the root anchoring from #91/#92 still protects
# everyone whose nested ``build/`` or ``dist/`` holds real sources.  See: #811.
NESTED_OUTPUT_DIR_MARKERS: dict[str, frozenset[str]] = {
    "target": frozenset({"pom.xml", "Cargo.toml", "build.sbt"}),
    "build": frozenset({
        "build.gradle",
        "build.gradle.kts",
        "settings.gradle",
        "settings.gradle.kts",
    }),
    ".next": frozenset({
        "next.config.js",
        "next.config.mjs",
        "next.config.cjs",
        "next.config.ts",
    }),
    ".nuxt": frozenset({"nuxt.config.js", "nuxt.config.mjs", "nuxt.config.ts"}),
}

# Bounds for the nested build-output scan.  The scan only lists directories
# (no file stats), stops at ``CRG_MODULE_SCAN_DEPTH`` levels, never descends
# into an already-ignored tree, and its result is cached per repository so
# incremental updates never pay for it twice inside the TTL.
_MODULE_SCAN_DEPTH = int(os.environ.get("CRG_MODULE_SCAN_DEPTH", "3"))
_MODULE_SCAN_MAX_DIRS = int(os.environ.get("CRG_MODULE_SCAN_MAX_DIRS", "2000"))
_MAX_NESTED_OUTPUT_PATTERNS = 200
_NESTED_IGNORE_TTL_SECONDS = float(os.environ.get("CRG_NESTED_IGNORE_TTL", "300"))

_nested_ignore_cache: dict[str, tuple[float, list[str]]] = {}
_nested_ignore_lock = threading.Lock()


def find_svn_root(start: Path | None = None) -> Optional[Path]:
    """Walk up from start to find the SVN working copy root.

    For SVN 1.7+, there is a single ``.svn`` at the WC root.
    For older SVN, every directory has ``.svn`` — we return the topmost one
    found so that the WC root is correctly identified.
    """
    current = start or Path.cwd()
    candidate: Optional[Path] = None
    while current != current.parent:
        if (current / ".svn").exists():
            candidate = current
        current = current.parent
    if (current / ".svn").exists():
        candidate = current
    return candidate


def find_repo_root(
    start: Path | None = None,
    stop_at: Path | None = None,
) -> Optional[Path]:
    """Walk up from ``start`` to find the nearest ``.git`` directory or SVN working copy root.

    Args:
        start: Starting directory.  Defaults to ``Path.cwd()``.
        stop_at: Optional boundary — if provided, the walk examines
            ``stop_at`` for a ``.git`` directory and then stops without
            crossing above it.  Useful for tests that create a synthetic
            repo under ``tmp_path`` (so the walk does not accidentally
            climb into a developer's home-directory dotfiles repo) and
            for any production caller that wants to bound the ancestor
            walk — e.g. multi-repo orchestrators, CI containers with
            bind-mounted volumes, embedded sandboxes.  See #241.

    Returns:
        The first ancestor containing ``.git`` or an SVN working copy,
        or ``None`` if no ancestor up to and including ``stop_at`` (when
        set) or the filesystem root (when ``stop_at is None``) contains one.
    """
    current = start or Path.cwd()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        if stop_at is not None and current == stop_at:
            return None
        current = current.parent
    if (current / ".git").exists():
        return current
    # No Git root found — try SVN
    return find_svn_root(start)


def detect_vcs(root: Path) -> str:
    """Return ``'git'``, ``'svn'``, or ``'none'`` based on VCS markers at *root*."""
    if (root / ".git").exists():
        return "git"
    if (root / ".svn").exists():
        return "svn"
    return "none"


def find_project_root(
    start: Path | None = None,
    stop_at: Path | None = None,
) -> Path:
    """Find the project root.

    Resolution order (highest precedence first):

    1. ``CRG_REPO_ROOT`` environment variable — explicit override for
       anyone scripting the CLI from outside the repo (CI jobs, daemons,
       multi-repo orchestrators). See: #155
    2. Git repository root via :func:`find_repo_root` from ``start``,
       honoring ``stop_at`` if provided.
    3. ``start`` itself (or cwd if no start given).

    ``stop_at`` is forwarded to :func:`find_repo_root` so callers that
    want to bound the ancestor walk (typically tests; see #241) can do so
    without having to call ``find_repo_root`` directly.
    """
    env_override = os.environ.get("CRG_REPO_ROOT", "").strip()
    if env_override:
        p = Path(env_override).expanduser().resolve()
        if p.exists():
            return p
    root = find_repo_root(start, stop_at=stop_at)
    if root:
        return root
    return start or Path.cwd()


def _write_data_dir_gitignore(data_dir: Path) -> None:
    """Write .gitignore file in data directory if it doesn't exist.

    The gitignore contains a single '*' to prevent accidental commits.
    """
    inner_gitignore = data_dir / ".gitignore"
    if not inner_gitignore.exists():
        try:
            # `encoding="utf-8"` is REQUIRED — the em-dash in the header is
            # U+2014 which falls outside cp1252.  On Windows, calling
            # write_text without an encoding silently uses the system default
            # codepage, producing a file that subsequently fails to decode as
            # UTF-8 (see issue #239).
            inner_gitignore.write_text(
                "# Auto-generated by code-review-graph — do not commit database files.\n"
                "# The graph.db contains absolute paths and code structure metadata.\n"
                "*\n",
                encoding="utf-8",
            )
        except OSError:
            # Data dir might be read-only (rare); that's OK, it's a best-effort guard.
            pass


def get_data_dir(repo_root: Path, *, create: bool = True) -> Path:
    """Return the directory where this project's graph data lives.

    Resolution priority:
    1. Registry entry for this repo (set via --data-dir)
    2. CRG_DATA_DIR environment variable (global override)
    3. Default: <repo>/.code-review-graph/

    By default, ``<repo_root>/.code-review-graph``. If the
    ``CRG_DATA_DIR`` environment variable is set, it is used verbatim
    instead — letting you keep graphs outside the working tree (useful
    for ephemeral workspaces, Docker volumes, or shared caches). See: #155

    By default the directory is created if it does not already exist; an
    inner ``.gitignore`` (with ``*``) is written so any accidentally-nested
    files never get committed. Both are idempotent. Pass ``create=False``
    when resolving the path for a read-only existence check.
    """
    # Check registry first
    try:
        from .registry import Registry, default_registry_path

        # Registry construction creates its parent directory. A read-only
        # lookup must skip it entirely when no registry file exists.
        if create or default_registry_path().is_file():
            registry_data_dir = Registry().get_data_dir_for_repo(str(repo_root))
            if registry_data_dir:
                data_dir = Path(registry_data_dir).resolve()
                if create:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    _write_data_dir_gitignore(data_dir)
                return data_dir
    except Exception as exc:
        # If registry lookup fails, log and fall through to other methods
        logger.debug("Registry lookup failed for %s: %s", repo_root, exc)

    # Check environment variable
    env_override = os.environ.get("CRG_DATA_DIR", "").strip()
    if env_override:
        data_dir = Path(env_override).expanduser().resolve()
    else:
        data_dir = repo_root / ".code-review-graph"

    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
        _write_data_dir_gitignore(data_dir)

    return data_dir


def get_db_path(repo_root: Path, *, read_only: bool = False) -> Path:
    """Determine the database path for a repository.

    Respects ``CRG_DATA_DIR`` (see :func:`get_data_dir`). Migrates a
    legacy top-level ``.code-review-graph.db`` file into the new
    directory when it exists (WAL/SHM side-files are discarded). Pass
    ``read_only=True`` to resolve the current path without creating a data
    directory, migrating a legacy database, or deleting side-files.
    """
    crg_dir = get_data_dir(repo_root, create=not read_only)
    new_db = crg_dir / "graph.db"

    if read_only:
        return new_db

    # Migrate legacy database if present (only meaningful when the
    # legacy file sits at the repo root — if CRG_DATA_DIR is set we
    # skip the migration because there's no relationship between the
    # legacy location and the new one).
    legacy_db = repo_root / ".code-review-graph.db"
    if legacy_db.exists() and not new_db.exists():
        legacy_db.rename(new_db)
    # Discard stale WAL/SHM side-files from the old location
    for suffix in ("-wal", "-shm", "-journal"):
        side = repo_root / f".code-review-graph.db{suffix}"
        if side.exists():
            side.unlink()

    return new_db


def ensure_repo_gitignore_excludes_crg(repo_root: Path) -> str:
    """Ensure repo-level .gitignore excludes ``.code-review-graph/``.

    Returns one of:
    - ``created``: .gitignore was created with the entry
    - ``updated``: entry was appended to existing .gitignore
    - ``already-present``: no changes were needed
    """
    gitignore_path = repo_root / ".gitignore"
    existing = gitignore_path.read_text(encoding="utf-8") if gitignore_path.exists() else ""

    for raw_line in existing.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == ".code-review-graph" or line.startswith(".code-review-graph/"):
            return "already-present"

    block = "# Added by code-review-graph\n.code-review-graph/\n"
    prefix = "\n" if existing and not existing.endswith("\n") else ""
    gitignore_path.write_text(existing + prefix + block, encoding="utf-8")

    if existing:
        return "updated"
    return "created"


def _load_ignore_patterns(repo_root: Path) -> list[str]:
    """Load ignore patterns from .code-review-graphignore file."""
    patterns = list(DEFAULT_IGNORE_PATTERNS)
    ignore_file = repo_root / ".code-review-graphignore"
    if ignore_file.exists():
        for line in ignore_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                # Directory names without a slash match at any depth, as in
                # .gitignore. A leading slash remains an explicit root anchor.
                if line.endswith("/"):
                    prefix = line[:-1]
                    if prefix.startswith("/") or "/" in prefix:
                        line = f"{prefix}/**"
                    else:
                        line = f"**/{prefix}/**"
                elif line.endswith("/**") and not line.startswith(("/", "**/")):
                    prefix = line[:-3]
                    if "/" in prefix:
                        line = f"/{line}"
                    else:
                        line = f"**/{line}"
                if line:
                    patterns.append(line)
    patterns.extend(_nested_output_ignore_patterns(repo_root, patterns))
    return patterns


def _should_ignore(path: str, patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern.

    ``**/<dir>/**`` and unanchored single-directory patterns match at any
    depth. A leading slash anchors a pattern to the repository root.
    """
    normalized = path.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    for pattern in patterns:
        anchored = pattern.startswith("/")
        candidate = pattern[1:] if anchored else pattern

        if candidate.startswith("**/") and candidate.endswith("/**"):
            segment = candidate[3:-3]
            if segment and segment in parts:
                return True
            continue

        if candidate.endswith("/**"):
            prefix = tuple(part for part in candidate[:-3].split("/") if part)
            if not prefix:
                continue
            if anchored or len(prefix) > 1:
                if parts[: len(prefix)] == prefix:
                    return True
            elif prefix[0] in parts:
                return True
            continue

        if fnmatch.fnmatch(normalized, candidate):
            return True
    return False


def _child_directories(directory: Path) -> list[tuple[str, bool]]:
    """List ``directory`` once, returning ``(name, is_dir)`` for its entries.

    Symlinked directories are reported as non-directories so no walk follows
    them out of the repository.  Returns an empty list for anything that
    cannot be listed — an unreadable directory is not worth a failed watch.
    """
    try:
        with os.scandir(directory) as entries:
            listing: list[tuple[str, bool]] = []
            for entry in entries:
                try:
                    listing.append((entry.name, entry.is_dir(follow_symlinks=False)))
                except OSError:  # pragma: no cover - vanished mid-scan
                    continue
    except OSError as exc:
        logger.debug("Cannot list %s: %s", directory, exc)
        return []
    listing.sort()
    return listing


def _scan_nested_output_dirs(
    repo_root: Path,
    base_patterns: list[str],
    max_depth: int = _MODULE_SCAN_DEPTH,
    max_dirs: int = _MODULE_SCAN_MAX_DIRS,
) -> list[str]:
    """Find nested build-output directories that a sibling manifest confirms.

    Returns root-anchored ignore patterns such as ``/moduleA/target/**``.  The
    walk lists directories only, skips trees the base patterns already ignore,
    and is bounded by *max_depth* and *max_dirs*.
    """
    patterns: list[str] = []
    queue: list[tuple[Path, int]] = [(repo_root, 0)]
    visited = 0
    while queue:
        directory, depth = queue.pop()
        visited += 1
        if visited > max_dirs:
            logger.debug("Nested output scan hit the %d directory cap", max_dirs)
            break
        listing = _child_directories(directory)
        subdirectories = {name for name, is_dir in listing if is_dir}
        file_names = {name for name, is_dir in listing if not is_dir}
        flagged: set[str] = set()
        if depth > 0:
            for output_dir, markers in NESTED_OUTPUT_DIR_MARKERS.items():
                if output_dir in subdirectories and file_names & markers:
                    relative = (directory / output_dir).relative_to(repo_root).as_posix()
                    patterns.append(f"/{relative}/**")
                    flagged.add(output_dir)
                    if len(patterns) >= _MAX_NESTED_OUTPUT_PATTERNS:
                        logger.debug("Nested output scan hit the pattern cap")
                        return patterns
        if depth >= max_depth:
            continue
        for name in subdirectories:
            if name in flagged:
                continue
            child = directory / name
            if _should_ignore(child.relative_to(repo_root).as_posix(), base_patterns):
                continue
            queue.append((child, depth + 1))
    return patterns


def _nested_output_ignore_patterns(repo_root: Path, base_patterns: list[str]) -> list[str]:
    """Cached wrapper around :func:`_scan_nested_output_dirs`.

    The result is reused for ``_NESTED_IGNORE_TTL_SECONDS`` so a long-running
    watch pays for the scan once, not on every incremental update, while a
    module added later is still picked up without a restart.  Set
    ``CRG_NESTED_OUTPUT_SCAN=0`` to turn the whole thing off.
    """
    if os.environ.get("CRG_NESTED_OUTPUT_SCAN", "1").strip().lower() in ("0", "false", "no"):
        return []
    key = str(repo_root)
    now = time.monotonic()
    with _nested_ignore_lock:
        cached = _nested_ignore_cache.get(key)
        if cached is not None and now - cached[0] < _NESTED_IGNORE_TTL_SECONDS:
            return list(cached[1])
    patterns = _scan_nested_output_dirs(repo_root, base_patterns)
    with _nested_ignore_lock:
        _nested_ignore_cache[key] = (time.monotonic(), patterns)
    return list(patterns)


def clear_nested_ignore_cache() -> None:
    """Drop the cached nested build-output patterns (used by tests)."""
    with _nested_ignore_lock:
        _nested_ignore_cache.clear()


def _is_binary(path: Path) -> bool:
    """Quick heuristic: check if file appears to be binary."""
    try:
        chunk = path.read_bytes()[:8192]
        return b"\x00" in chunk
    except (OSError, PermissionError):
        return True


_GIT_TIMEOUT = int(os.environ.get("CRG_GIT_TIMEOUT", "30"))  # seconds, configurable

# When True, `git ls-files --recurse-submodules` is used so that files
# inside git submodules are included in the graph.  Opt-in via env var;
# can also be overridden per-call through function parameters.
_RECURSE_SUBMODULES = os.environ.get("CRG_RECURSE_SUBMODULES", "").lower() in ("1", "true", "yes")


def _git_branch_info(repo_root: Path) -> tuple[str, str]:
    """Return (branch_name, head_sha) for the current repo state."""
    branch = ""
    sha = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, UnicodeDecodeError):
        pass
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, UnicodeDecodeError):
        pass
    return branch, sha


def _svn_revision_info(repo_root: Path) -> tuple[str, str]:
    """Return (branch_path, revision_str) for the current SVN working copy."""
    branch = ""
    rev = ""
    try:
        result = subprocess.run(
            ["svn", "info", "--non-interactive"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(repo_root), timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("URL: "):
                    url = line[5:].strip()
                    # Extract trunk/branches/tags segment from SVN URL
                    for marker in ("/branches/", "/tags/", "/trunk"):
                        if marker in url:
                            idx = url.index(marker)
                            branch = url[idx:].lstrip("/")
                            break
                    if not branch and url:
                        branch = url.rstrip("/").split("/")[-1]
                elif line.startswith("Revision: "):
                    rev = line[10:].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return branch, rev


_SAFE_GIT_REF = re.compile(r"^[A-Za-z0-9_.~^/@{}\-]+$")
_SAFE_SVN_REV = re.compile(r"^r?\d+(:r?\d+|:HEAD|:BASE|:COMMITTED)?$", re.IGNORECASE)


def _decode_name_status_paths(output: bytes) -> list[str]:
    """Decode ``git diff --name-status -z`` output into a list of paths.

    Renames and copies (``R<score>``/``C<score>`` records) carry two paths —
    the old and the new one.  Both are emitted so the old path flows through
    the purge loop in :func:`incremental_update`; otherwise a rename leaves
    the old path's nodes and edges in the graph and the incremental result
    diverges from a full rebuild.
    """
    fields = [os.fsdecode(f) for f in output.split(b"\0") if f]
    paths: list[str] = []
    seen: set[str] = set()
    i = 0
    while i < len(fields):
        status = fields[i]
        takes_two = status[:1] in ("R", "C")
        entry = fields[i + 1 : i + (3 if takes_two else 2)]
        i += 3 if takes_two else 2
        for path in entry:
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def _store_vcs_metadata(repo_root: Path, store: "GraphStore") -> None:
    """Persist VCS branch/revision info into the graph metadata table."""
    vcs = detect_vcs(repo_root)
    if vcs == "git":
        branch, sha = _git_branch_info(repo_root)
        if branch:
            store.set_metadata("git_branch", branch)
        if sha:
            store.set_metadata("git_head_sha", sha)
    elif vcs == "svn":
        branch, rev = _svn_revision_info(repo_root)
        if branch:
            store.set_metadata("svn_branch", branch)
        if rev:
            store.set_metadata("svn_revision", rev)


def _commit_object_exists(repo_root: Path, ref: str) -> bool:
    """Return True if *ref* resolves to a commit object present in the repo.

    This is an object-existence check, not an ancestry check: a commit that is
    only reachable from a branch we have since switched away from is still a
    valid ``git diff`` base, so we must accept it. Any git failure (missing
    binary, timeout, unknown ref) is treated as "not usable".
    """
    if not ref or ref.startswith("-") or not _SAFE_GIT_REF.fullmatch(ref):
        return False
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def resolve_incremental_base(repo_root: Path, store: "GraphStore") -> str | None:
    """Resolve the automatic diff base for a default incremental update.

    The graph records the commit it was last built at (``git_head_sha``). Using
    that as the diff base lets a single ``update`` reconcile every change since
    the graph was last in sync, instead of only the most recent commit, which
    is what a fixed ``HEAD~1`` base does. That fixed base silently misses work
    that arrived through a multi-commit pull, rebase, or branch switch.

    Returns:
        - the stored commit SHA when it is still a usable diff base;
        - ``"HEAD~1"`` for SVN or non-git working copies, whose change
          discovery ignores or reinterprets the base anyway;
        - ``None`` for a git repo with no usable anchor (a fresh or legacy
          database, or a stored commit lost to a history rewrite or shallow
          clone), signalling the caller to do a full rebuild rather than
          diff against a wrong base.
    """
    if detect_vcs(repo_root) != "git":
        return "HEAD~1"
    stored = store.get_metadata("git_head_sha")
    if stored and _commit_object_exists(repo_root, stored):
        return stored
    return None


def get_changed_files(repo_root: Path, base: str = "HEAD~1") -> list[str]:
    """Get list of changed files via git diff or svn status.

    For SVN working copies the *base* parameter is ignored; modified/added/
    deleted files are detected from ``svn status``.  Pass an SVN revision
    range (e.g. ``"r100:HEAD"``) as *base* to compare against a specific
    revision instead.
    """
    if detect_vcs(repo_root) == "svn":
        return _get_svn_changed_files(repo_root, base if _SAFE_SVN_REV.match(base) else None)
    # Git path
    if base.startswith("-") or not _SAFE_GIT_REF.fullmatch(base):
        logger.warning("Invalid git ref rejected: %s", base)
        return []
    try:
        # --name-status (not --name-only): renames/copies must report BOTH
        # paths, or the old path never reaches the purge loop (issue #684).
        result = subprocess.run(
            ["git", "diff", "--name-status", "-z", base, "--"],
            capture_output=True,
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            # Fallback: try diff against empty tree (initial commit)
            result = subprocess.run(
                ["git", "diff", "--name-status", "-z", "--cached"],
                capture_output=True,
                cwd=str(repo_root),
                timeout=_GIT_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
        if result.returncode != 0:
            logger.warning("git diff failed while discovering changed files")
            return []
        return _decode_name_status_paths(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

def _get_svn_changed_files(repo_root: Path, rev_range: str | None = None) -> list[str]:
    """Return changed files in an SVN working copy.

    When *rev_range* is given (e.g. ``"r100:HEAD"``), ``svn diff --summarize``
    is used to list files changed between those revisions.  Otherwise
    ``svn status`` reports working-copy modifications.
    """
    try:
        if rev_range:
            result = subprocess.run(
                ["svn", "diff", "--summarize", "--non-interactive", "-r", rev_range],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(repo_root), timeout=_GIT_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode != 0:
                logger.warning("svn diff --summarize failed (rc=%d): %s",
                               result.returncode, result.stderr[:200])
                return []
            files = []
            for line in result.stdout.splitlines():
                # Format: "M       path/to/file"  (first char is status)
                if len(line) >= 2 and line[0] in ("M", "A", "D"):
                    files.append(line[1:].strip())
            return files
        else:
            result = subprocess.run(
                ["svn", "status", "--non-interactive"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(repo_root), timeout=_GIT_TIMEOUT,
                stdin=subprocess.DEVNULL,
            )
            files = []
            for line in result.stdout.splitlines():
                if len(line) < 2:
                    continue
                status_char = line[0]
                # M=modified, A=added, D=deleted, R=replaced, C=conflicted
                if status_char in ("M", "A", "D", "R", "C"):
                    # SVN status: 8 fixed-width columns then the path
                    path = line[8:].strip() if len(line) > 8 else line[1:].strip()
                    files.append(path)
            return files
    except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return []

def get_staged_and_unstaged(repo_root: Path) -> list[str]:
    """Get all modified files (staged + unstaged + untracked)."""
    if detect_vcs(repo_root) == "svn":
        return _get_svn_changed_files(repo_root)
    try:
        result = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            capture_output=True,
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            logger.warning("git status failed while discovering working-tree files")
            return []
        files: list[str] = []
        records = result.stdout.split(b"\0")
        index = 0
        while index < len(records):
            record = records[index]
            if len(record) > 3:
                status = record[:2]
                files.append(os.fsdecode(record[3:]))
                # With porcelain -z, a rename/copy record stores the
                # destination first and its source in the following record.
                if b"R" in status or b"C" in status:
                    index += 1
            index += 1
        return files
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

def get_all_tracked_files(
    repo_root: Path,
    recurse_submodules: bool | None = None,
) -> list[str]:
    """Get all files tracked by git or svn.

    Args:
        repo_root: Repository root directory.
        recurse_submodules: If True, pass ``--recurse-submodules`` to
            ``git ls-files`` so that files inside git submodules are
            included.  When *None* (default), falls back to the
            ``CRG_RECURSE_SUBMODULES`` environment variable.
            (Ignored for SVN working copies.)
    """
    if detect_vcs(repo_root) == "svn":
        return _get_svn_all_tracked_files(repo_root)

    if recurse_submodules is None:
        recurse_submodules = _RECURSE_SUBMODULES

    cmd = ["git", "ls-files"]
    if recurse_submodules:
        cmd.append("--recurse-submodules")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            cwd=str(repo_root),
            timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, UnicodeDecodeError):
        return []

def _get_svn_all_tracked_files(repo_root: Path) -> list[str]:
    """Return SVN-versioned files by walking the working copy.

    Uses ``svn list -R`` to get the server-side file list, falling back to
    a filesystem walk (which is also the fallback in :func:`collect_all_files`).
    """
    try:
        result = subprocess.run(
            ["svn", "list", "--recursive", "--non-interactive"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(repo_root), timeout=60,  # svn list queries the server
            stdin=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            # svn list returns paths relative to the WC URL; directories end with "/"
            files = [
                f.strip()
                for f in result.stdout.splitlines()
                if f.strip() and not f.strip().endswith("/")
            ]
            if files:
                return files
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: let collect_all_files do a filesystem walk
    return []


def collect_all_files(
    repo_root: Path,
    recurse_submodules: bool | None = None,
) -> list[str]:
    """Collect all parseable files in the repo, respecting ignore patterns.

    Args:
        repo_root: Repository root directory.
        recurse_submodules: If True, include files from git submodules.
            When *None*, falls back to ``CRG_RECURSE_SUBMODULES`` env var.
    """
    ignore_patterns = _load_ignore_patterns(repo_root)
    parser = CodeParser(repo_root)
    files = []

    # Prefer git ls-files for tracked files
    tracked = get_all_tracked_files(repo_root, recurse_submodules)
    if tracked:
        candidates = tracked
    else:
        # Fallback: walk directory
        candidates = [str(p.relative_to(repo_root)) for p in repo_root.rglob("*") if p.is_file()]

    for rel_path in candidates:
        if _should_ignore(rel_path, ignore_patterns):
            continue
        # Skip paths that would exceed OS filename limits (macOS: 255 bytes
        # per component, ~1024 total; Windows: 260 total).
        try:
            full_path = repo_root / rel_path
        except (OSError, ValueError):
            logger.debug("Skipping path that cannot be constructed: %s", rel_path)
            continue
        if len(str(full_path)) > 1000 or any(len(p.encode()) > 255 for p in full_path.parts):
            logger.debug("Skipping overlong path: %s", rel_path[:120])
            continue
        if not full_path.is_file():
            continue
        if full_path.is_symlink():
            continue
        if parser.detect_language(full_path) is None:
            continue
        if _is_binary(full_path):
            continue
        files.append(rel_path)

    return files


def _reconcile_stale_files(
    repo_root: Path,
    store: GraphStore,
    current_files: list[str] | None = None,
) -> list[str]:
    """Remove graph files absent from the current parseable repository inventory."""
    stored_files = set(store.get_all_files())
    current_paths: set[str]
    if current_files is not None:
        current_paths = {
            normalize_file_path(repo_root / file_path) for file_path in current_files
        }
    else:
        ignore_patterns = _load_ignore_patterns(repo_root)
        parser = CodeParser(repo_root)
        current_paths = set()
        for stored_file in stored_files:
            path = Path(stored_file)
            try:
                relative = str(path.relative_to(repo_root))
            except ValueError:
                continue
            if (
                path.is_file()
                and not path.is_symlink()
                and not _should_ignore(relative, ignore_patterns)
                and parser.detect_language(path) is not None
                and not _is_binary(path)
            ):
                current_paths.add(stored_file)
    stale_files = sorted(stored_files - current_paths)
    if stale_files:
        store.remove_files_permanently(stale_files)
    return stale_files


_MAX_DEPENDENT_HOPS = int(os.environ.get("CRG_DEPENDENT_HOPS", "2"))
_MAX_DEPENDENT_FILES = 500


def _single_hop_dependents(store: GraphStore, file_path: str) -> set[str]:
    """Find files that directly depend on *file_path* (single hop)."""
    dependents: set[str] = set()
    edges = store.get_edges_by_target(file_path)
    for e in edges:
        if e.kind == "IMPORTS_FROM":
            dependents.add(e.file_path)

    nodes = store.get_nodes_by_file(file_path)
    for node in nodes:
        for e in store.get_edges_by_target(node.qualified_name):
            if e.kind in ("CALLS", "IMPORTS_FROM", "INHERITS", "IMPLEMENTS"):
                dependents.add(e.file_path)

    dependents.discard(file_path)
    return dependents


class DependentList(list):
    """A ``list[str]`` with a ``.truncated`` flag.

    When :func:`find_dependents` hits ``_MAX_DEPENDENT_FILES`` it truncates
    the result and sets ``truncated = True`` so callers can distinguish a
    complete expansion from a capped one.  See issue #261.

    This is a transparent ``list`` subclass — existing callers that iterate,
    ``len()``, or slice continue to work unchanged; only callers that
    specifically check ``.truncated`` benefit from the signal.
    """

    truncated: bool

    def __init__(self, items: list, *, truncated: bool = False) -> None:
        super().__init__(items)
        self.truncated = truncated


def find_dependents(
    store: GraphStore,
    file_path: str,
    max_hops: int = _MAX_DEPENDENT_HOPS,
) -> DependentList:
    """Find files that import from or depend on the given file.

    Performs up to *max_hops* iterations of expansion (default 2).
    Stops early if the total exceeds 500 files.

    Returns a :class:`DependentList` — a regular ``list[str]`` that also
    carries a ``.truncated`` flag.  When ``truncated is True`` the
    returned list is capped at ``_MAX_DEPENDENT_FILES`` and the full
    set of dependents was not explored.  See issue #261.
    """
    all_dependents: set[str] = set()
    visited: set[str] = {file_path}
    frontier: set[str] = {file_path}
    for _hop in range(max_hops):
        next_frontier: set[str] = set()
        for fp in frontier:
            deps = _single_hop_dependents(store, fp)
            new_deps = deps - visited
            all_dependents.update(new_deps)
            next_frontier.update(new_deps)
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
        if len(all_dependents) > _MAX_DEPENDENT_FILES:
            logger.warning(
                "Dependent expansion capped at %d files for %s",
                len(all_dependents),
                file_path,
            )
            return DependentList(
                list(all_dependents)[:_MAX_DEPENDENT_FILES],
                truncated=True,
            )
    return DependentList(list(all_dependents))


def _parse_single_file(
    args: tuple[str, str],
) -> tuple[str, list, list, str | None, str]:
    """Parse one file in a process- or thread-pool worker.

    Returns ``(rel_path, nodes, edges, error_or_none, file_hash)``.
    Must be a module-level function so ``ProcessPoolExecutor`` can
    serialise it across processes.
    """
    rel_path, repo_root_str = args
    abs_path = Path(repo_root_str) / rel_path
    try:
        raw = abs_path.read_bytes()
        fhash = hashlib.sha256(raw).hexdigest()
        parser = getattr(_PARSE_WORKER_STATE, "parser", None)
        parser_repo_root = getattr(_PARSE_WORKER_STATE, "repo_root", None)
        if parser is None or parser_repo_root != repo_root_str:
            parser = CodeParser(Path(repo_root_str))
            _PARSE_WORKER_STATE.parser = parser
            _PARSE_WORKER_STATE.repo_root = repo_root_str
        nodes, edges = parser.parse_bytes(abs_path, raw)
        return (rel_path, nodes, edges, None, fhash)
    except Exception as e:
        return (rel_path, [], [], str(e), "")


def full_build(
    repo_root: Path,
    store: GraphStore,
    recurse_submodules: bool | None = None,
) -> dict:
    """Full rebuild of the entire graph.

    Args:
        repo_root: Repository root directory.
        store: Graph database store.
        recurse_submodules: If True, include files from git submodules.
            When *None*, falls back to ``CRG_RECURSE_SUBMODULES`` env var.
    """
    parser = CodeParser(repo_root)
    files = collect_all_files(repo_root, recurse_submodules)
    stale_files = _reconcile_stale_files(repo_root, store, files)

    total_nodes = 0
    total_edges = 0
    errors = []
    cpp_errors: set[str] = set()
    file_count = len(files)

    use_serial = os.environ.get("CRG_SERIAL_PARSE", "") == "1"

    if use_serial or file_count < 8:
        # Serial fallback (for debugging or tiny repos)
        for i, rel_path in enumerate(files, 1):
            full_path = repo_root / rel_path
            try:
                source = full_path.read_bytes()
                fhash = hashlib.sha256(source).hexdigest()
                nodes, edges = parser.parse_bytes(full_path, source)
                store.store_file_nodes_edges(str(full_path), nodes, edges, fhash)
                total_nodes += len(nodes)
                total_edges += len(edges)
            except (OSError, PermissionError) as e:
                errors.append({"file": rel_path, "error": str(e)})
                if parser.detect_language(full_path) == "cpp":
                    cpp_errors.add(str(rel_path))
            except Exception as e:
                logger.warning("Error parsing %s: %s", rel_path, e)
                errors.append({"file": rel_path, "error": str(e)})
                if parser.detect_language(full_path) == "cpp":
                    cpp_errors.add(str(rel_path))
            if i % 50 == 0 or i == file_count:
                logger.info("Progress: %d/%d files parsed", i, file_count)
    else:
        # Parallel parsing — store calls remain serial (SQLite single-writer).
        # Executor kind auto-selected: process for normal CLI/automation;
        # thread for MCP stdio to avoid pipe-handle inheritance deadlocks and
        # orphan workers (issues #46, #136, PR #615). Override via
        # CRG_PARSE_EXECUTOR env.
        args_list = [(rel_path, str(repo_root)) for rel_path in files]
        with _make_executor(_MAX_PARSE_WORKERS) as executor:
            for i, (rel_path, nodes, edges, error, fhash) in enumerate(
                executor.map(_parse_single_file, args_list, chunksize=20),
                1,
            ):
                if error:
                    logger.warning("Error parsing %s: %s", rel_path, error)
                    errors.append({"file": rel_path, "error": error})
                    if parser.detect_language(repo_root / rel_path) == "cpp":
                        cpp_errors.add(str(rel_path))
                    continue
                full_path = repo_root / rel_path
                store.store_file_nodes_edges(
                    str(full_path),
                    nodes,
                    edges,
                    fhash,
                )
                total_nodes += len(nodes)
                total_edges += len(edges)
                if i % 200 == 0 or i == file_count:
                    logger.info("Progress: %d/%d files parsed", i, file_count)

    store.set_metadata("last_updated", time.strftime("%Y-%m-%dT%H:%M:%S"))
    store.set_metadata("last_build_type", "full")
    if not cpp_errors:
        store.set_metadata(_CPP_IDENTITY_METADATA_KEY, CPP_IDENTITY_VERSION)
    _store_vcs_metadata(repo_root, store)
    store.commit()

    python_stats = _run_python_resolver(store)
    rescript_stats = _run_rescript_resolver(store)
    spring_stats = _run_spring_resolver(store)
    spring_event_stats = _run_spring_event_resolver(store)
    temporal_stats = _run_temporal_resolver(store)
    hcl_stats = _run_hcl_resolver(store)
    scoped_stats = _run_scoped_resolver(store)

    return {
        "files_parsed": len(files),
        "stale_files_removed": len(stale_files),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "errors": errors,
        "python_resolution": python_stats,
        "rescript_resolution": rescript_stats,
        "spring_resolution": spring_stats,
        "event_resolution": spring_event_stats,
        "temporal_resolution": temporal_stats,
        "hcl_resolution": hcl_stats,
        "scoped_resolution": scoped_stats,
    }


def incremental_update(
    repo_root: Path,
    store: GraphStore,
    base: str = "HEAD~1",
    changed_files: list[str] | None = None,
    reconcile_stale: bool = True,
) -> dict:
    """Incremental update: re-parse changed + dependent files only."""
    parser = CodeParser(repo_root)
    ignore_patterns = _load_ignore_patterns(repo_root)

    if (
        store.get_metadata(_CPP_IDENTITY_METADATA_KEY) != CPP_IDENTITY_VERSION
        and store.has_nodes_for_language("cpp")
    ):
        logger.info(
            "C++ identity format changed; rebuilding the graph before incremental update",
        )
        rebuilt = full_build(repo_root, store)
        return {
            "files_updated": rebuilt["files_parsed"],
            "total_nodes": rebuilt["total_nodes"],
            "total_edges": rebuilt["total_edges"],
            "changed_files": list(changed_files or []),
            "dependent_files": [],
            "errors": rebuilt["errors"],
            "identity_rebuild": True,
            "python_resolution": rebuilt["python_resolution"],
            "rescript_resolution": rebuilt["rescript_resolution"],
            "spring_resolution": rebuilt["spring_resolution"],
            "event_resolution": rebuilt["event_resolution"],
            "temporal_resolution": rebuilt["temporal_resolution"],
            "hcl_resolution": rebuilt["hcl_resolution"],
        }

    # Determine changed files
    if changed_files is None:
        changed_files = get_changed_files(repo_root, base)
    stale_files = _reconcile_stale_files(repo_root, store) if reconcile_stale else []

    if not changed_files and not stale_files:
        return {
            "files_updated": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "changed_files": [],
            "dependent_files": [],
            "stale_files_removed": 0,
            "errors": [],
        }

    # Find dependent files (files that import from changed files)
    dependent_files: set[str] = set()
    for rel_path in changed_files:
        full_path = normalize_file_path(repo_root / rel_path)
        deps = find_dependents(store, full_path)
        for d in deps:
            # Convert back to relative path if needed
            try:
                dependent_files.add(str(Path(d).relative_to(repo_root)))
            except ValueError:
                dependent_files.add(d)

    # Combine changed + dependent
    all_files = set(changed_files) | dependent_files

    total_nodes = 0
    total_edges = 0
    errors = []
    missing_paths: set[str] = set()

    # Separate deleted/unparseable files from files that need re-parsing
    to_parse: list[str] = []
    for rel_path in all_files:
        if _should_ignore(rel_path, ignore_patterns):
            continue
        abs_path = repo_root / rel_path
        if not abs_path.is_file():
            if normalize_file_path(abs_path) not in stale_files:
                missing_paths.add(normalize_file_path(abs_path))
            continue
        if parser.detect_language(abs_path) is None:
            continue
        # Quick hash check to skip unchanged files
        try:
            raw = abs_path.read_bytes()
            fhash = hashlib.sha256(raw).hexdigest()
            existing_nodes = store.get_nodes_by_file(str(abs_path))
            if existing_nodes and existing_nodes[0].file_hash == fhash:
                continue
        except (OSError, PermissionError):
            pass
        to_parse.append(rel_path)

    # Persist deletions before store_file_nodes_edges() opens its own
    # explicit transaction — avoids nested transaction errors.
    use_serial = os.environ.get("CRG_SERIAL_PARSE", "") == "1"
    parsed_files = 0

    if use_serial or len(to_parse) < 8:
        for rel_path in to_parse:
            abs_path = repo_root / rel_path
            try:
                source = abs_path.read_bytes()
                fhash = hashlib.sha256(source).hexdigest()
                nodes, edges = parser.parse_bytes(abs_path, source)
                store.store_file_nodes_edges(str(abs_path), nodes, edges, fhash)
                parsed_files += 1
                total_nodes += len(nodes)
                total_edges += len(edges)
            except (OSError, PermissionError) as e:
                errors.append({"file": rel_path, "error": str(e)})
            except Exception as e:
                logger.warning("Error parsing %s: %s", rel_path, e)
                errors.append({"file": rel_path, "error": str(e)})
    else:
        # See full-build comment above for executor kind rationale.
        args_list = [(rel_path, str(repo_root)) for rel_path in to_parse]
        with _make_executor(_MAX_PARSE_WORKERS) as executor:
            for rel_path, nodes, edges, error, fhash in executor.map(
                _parse_single_file,
                args_list,
                chunksize=20,
            ):
                if error:
                    logger.warning("Error parsing %s: %s", rel_path, error)
                    errors.append({"file": rel_path, "error": error})
                    continue
                store.store_file_nodes_edges(
                    str(repo_root / rel_path),
                    nodes,
                    edges,
                    fhash,
                )
                parsed_files += 1
                total_nodes += len(nodes)
                total_edges += len(edges)

    removed_files = store.remove_files_permanently(sorted(missing_paths)) if missing_paths else 0
    files_updated = parsed_files + len(stale_files) + removed_files
    if files_updated:
        store.set_metadata("last_updated", time.strftime("%Y-%m-%dT%H:%M:%S"))
        store.set_metadata("last_build_type", "incremental")
        store.set_metadata(_CPP_IDENTITY_METADATA_KEY, CPP_IDENTITY_VERSION)
        _store_vcs_metadata(repo_root, store)
        store.commit()

    # Only re-run language-specific resolvers when the relevant files changed.
    python_changed = any(
        path.endswith(".py")
        for path in set(all_files) | set(stale_files) | missing_paths
    )
    python_stats = _run_python_resolver(store) if python_changed else None

    rescript_changed = any(
        rp.endswith((".res", ".resi")) for rp in all_files
    )
    rescript_stats = (
        _run_rescript_resolver(store) if rescript_changed else None
    )

    # Like python_changed above, include stale/missing paths so a deletion
    # that only surfaces through reconciliation still clears derived state
    # (e.g. virtual Spring Event nodes — issue #474).
    spring_changed = any(
        path.endswith(".java")
        for path in set(all_files) | set(stale_files) | missing_paths
    )
    spring_stats = _run_spring_resolver(store) if spring_changed else None
    spring_event_stats = (
        _run_spring_event_resolver(store) if spring_changed else None
    )
    temporal_stats = _run_temporal_resolver(store) if spring_changed else None
    hcl_changed = any(rp.endswith((".tf", ".hcl")) for rp in all_files)
    hcl_stats = _run_hcl_resolver(store) if hcl_changed else None
    scoped_changed = any(rp.endswith((".php", ".rs", ".cs")) for rp in all_files)
    scoped_stats = _run_scoped_resolver(store) if scoped_changed else None

    return {
        "files_updated": files_updated,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "changed_files": list(changed_files),
        "dependent_files": list(dependent_files),
        "stale_files_removed": len(stale_files),
        "errors": errors,
        "python_resolution": python_stats,
        "rescript_resolution": rescript_stats,
        "spring_resolution": spring_stats,
        "event_resolution": spring_event_stats,
        "temporal_resolution": temporal_stats,
        "hcl_resolution": hcl_stats,
        "scoped_resolution": scoped_stats,
    }


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------


_DEBOUNCE_SECONDS = 1


def _raise_watch_update_errors(result: dict, context: str) -> None:
    """Fail the watch boundary when an incremental update reports errors."""
    errors = result.get("errors") or []
    if not errors:
        return
    details = "; ".join(
        f"{error.get('file', 'unknown')}: {error.get('error', 'unknown error')}"
        for error in errors
    )
    raise RuntimeError(f"{context} reported errors: {details}")


def _raise_watch_postprocess_warnings(result: object) -> None:
    """Treat structured post-processing warnings as a failed watch update."""
    if not isinstance(result, dict):
        return
    warnings = result.get("warnings") or []
    if warnings:
        details = "; ".join(str(warning) for warning in warnings)
        raise RuntimeError(f"post-processing reported warnings: {details}")


# ---------------------------------------------------------------------------
# Watch scheduling and supervision
# ---------------------------------------------------------------------------

# A single recursive watch on the repository root makes the OS register one
# watch per directory in the tree — including every temp directory a build tool
# churns through inside ``target/`` or ``node_modules/``.  Planning the watches
# ourselves keeps ignored trees off the OS watch list entirely.  See: #811.
_WATCH_PLAN_DEPTH = int(os.environ.get("CRG_WATCH_PLAN_DEPTH", "3"))
_MAX_WATCH_SCHEDULES = int(os.environ.get("CRG_MAX_WATCH_SCHEDULES", "24"))
# Splitting a watch costs one watchdog emitter, so it has to buy more than it
# costs: an ignored tree is only worth excluding once it holds this many
# directories.  A lone ``__pycache__`` is not worth a thread; ``target/`` is.
_WATCH_SPLIT_MIN_DIRS = int(os.environ.get("CRG_WATCH_SPLIT_MIN_DIRS", "4"))
_WATCH_HEALTH_INTERVAL = float(os.environ.get("CRG_WATCH_HEALTH_INTERVAL", "10"))
_MAX_PENDING_DIR_EVENTS = 1024
_WATCH_STOP_TIMEOUT = 10.0


def _watch_child_dirs(
    directory: Path,
    cache: dict[Path, list[Path]] | None = None,
) -> list[Path]:
    """Return the real (non-symlink) subdirectories of *directory*."""
    if cache is not None and directory in cache:
        return cache[directory]
    children = [directory / name for name, is_dir in _child_directories(directory) if is_dir]
    if cache is not None:
        cache[directory] = children
    return children


def _ignored_tree_weight(directory: Path, cap: int) -> int:
    """Count the directories inside an ignored tree, stopping at *cap*.

    Used to decide whether excluding the tree is worth a separate watch.  The
    count stops as soon as the cap is reached, so probing ``node_modules`` is
    barely more expensive than probing an empty ``__pycache__``.
    """
    if cap <= 0:
        return 0
    total = 1
    queue = [directory]
    while queue and total < cap:
        current = queue.pop()
        for name, is_dir in _child_directories(current):
            if not is_dir:
                continue
            total += 1
            if total >= cap:
                return total
            queue.append(current / name)
    return total


def _plan_watch_subtree(
    directory: Path,
    repo_root: Path,
    ignore_patterns: list[str],
    depth: int,
    max_depth: int,
    cache: dict[Path, list[Path]],
    split_threshold: int = _WATCH_SPLIT_MIN_DIRS,
) -> tuple[bool, list[tuple[Path, bool]]]:
    """Plan the watches covering *directory*.

    Returns ``(clean, plan)``.  ``clean`` means nothing below *directory*
    within *max_depth* is worth excluding, in which case a single recursive
    watch covers it.  Otherwise the directory is watched non-recursively and
    each surviving child is planned in turn, so the ignored subtree is never
    handed to the OS at all.
    """
    ignored_weight = 0
    kept: list[Path] = []
    for child in _watch_child_dirs(directory, cache):
        if _should_ignore(child.relative_to(repo_root).as_posix(), ignore_patterns):
            if ignored_weight < split_threshold:
                ignored_weight += _ignored_tree_weight(child, split_threshold - ignored_weight)
        else:
            kept.append(child)
    clean = ignored_weight < split_threshold
    if depth >= max_depth:
        if clean:
            return True, [(directory, True)]
        return False, [(directory, False)] + [(child, True) for child in kept]
    child_plans: list[tuple[Path, bool]] = []
    for child in kept:
        child_clean, child_plan = _plan_watch_subtree(
            child, repo_root, ignore_patterns, depth + 1, max_depth, cache, split_threshold
        )
        clean = clean and child_clean
        child_plans.extend(child_plan)
    if clean:
        return True, [(directory, True)]
    return False, [(directory, False)] + child_plans


def _plan_watch_paths(
    repo_root: Path,
    ignore_patterns: list[str],
    max_depth: int = _WATCH_PLAN_DEPTH,
    max_schedules: int = _MAX_WATCH_SCHEDULES,
    split_threshold: int = _WATCH_SPLIT_MIN_DIRS,
) -> list[tuple[Path, bool]]:
    """Return the ``(path, recursive)`` watches that cover the repo.

    Deeper plans exclude more ignored trees but cost one watchdog emitter each,
    so an over-budget plan is retried at a shallower depth before falling back
    to the single recursive root watch.
    """
    cache: dict[Path, list[Path]] = {}
    for depth in range(max(1, max_depth), 0, -1):
        _, plan = _plan_watch_subtree(
            repo_root, repo_root, ignore_patterns, 0, depth, cache, split_threshold
        )
        if len(plan) <= max(1, max_schedules):
            return plan
    logger.warning(
        "%s needs more than %d watches to skip its ignored trees; "
        "falling back to one recursive watch (raise CRG_MAX_WATCH_SCHEDULES to split it)",
        repo_root,
        max_schedules,
    )
    return [(repo_root, True)]


def _run_time_boxed(operation: Callable[[], Any], description: str, timeout: float = 10.0) -> None:
    """Run *operation* on a throwaway thread so a wedged watcher cannot hang exit.

    Watchdog's teardown joins its emitter threads, and the whole point of this
    module's health check is that one of those threads may be stuck.  Every
    watchdog thread is a daemon thread, so abandoning the join is safe.
    """

    def _call() -> None:
        try:
            operation()
        except Exception as exc:  # noqa: BLE001 - teardown must not mask the real error
            logger.debug("%s failed: %s", description, exc)

    thread = threading.Thread(target=_call, name="crg-watch-teardown", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        logger.warning(
            "%s did not finish in %.0fs; leaving it to process exit", description, timeout
        )


def _watch_health_path(repo_root: Path) -> Path | None:
    """Where this watcher publishes its health, or None if that is unavailable."""
    try:
        from .daemon import watch_health_path

        return watch_health_path(repo_root)
    except Exception as exc:  # noqa: BLE001 - health reporting is best-effort
        logger.debug("Watcher health reporting disabled: %s", exc)
        return None


class _WatchSupervisor:
    """Owns the observer's watches and reports whether they still work.

    Three jobs, all deliberately cheap:

    * schedule only the directories that survive the ignore patterns, so the OS
      never registers a watch inside an ignored tree;
    * adopt directories created later underneath a non-recursive watch;
    * notice a dead watchdog thread and publish watcher health, so a stalled
      watcher stops looking healthy to ``crg-daemon status``.
    """

    def __init__(
        self,
        observer: Any,
        repo_root: Path,
        ignore_patterns: list[str],
        health_path: Path | None = None,
        max_schedules: int = _MAX_WATCH_SCHEDULES,
    ) -> None:
        self._observer = observer
        self._repo_root = repo_root
        self._ignore_patterns = ignore_patterns
        self._health_path = health_path
        self._max_schedules = max(1, max_schedules)
        self._handler: Any = None
        self._watches: dict[str, Any] = {}
        self._shallow: set[str] = set()
        self._pending: deque[tuple[str, str]] = deque(maxlen=_MAX_PENDING_DIR_EVENTS)
        self._live_threads: dict[int, threading.Thread] = {}
        self._budget_warned = False
        self._last_health_write = 0.0
        self._last_health_state: tuple[bool, tuple[str, ...]] | None = None
        self._started_at = time.time()

    # -- scheduling -----------------------------------------------------

    @property
    def watched_paths(self) -> list[str]:
        return sorted(self._watches)

    def schedule_initial(self, handler: Any) -> None:
        """Schedule the planned watches for *handler*."""
        self._handler = handler
        plan = _plan_watch_paths(
            self._repo_root,
            self._ignore_patterns,
            max_schedules=self._max_schedules,
        )
        for path, recursive in plan:
            self._schedule(path, recursive=recursive)
        logger.info(
            "Watching %d path(s) under %s; ignored trees are never registered",
            len(self._watches),
            self._repo_root,
        )

    def _schedule(self, path: Path, *, recursive: bool) -> None:
        key = str(path)
        if key in self._watches:
            return
        try:
            handle = self._observer.schedule(self._handler, key, recursive=recursive)
        except OSError as exc:
            logger.warning("Could not watch %s: %s", key, exc)
            return
        self._watches[key] = handle
        if recursive:
            self._shallow.discard(key)
        else:
            self._shallow.add(key)

    def note_directory_event(self, event_type: str, path: str) -> None:
        """Record a directory event.  Runs on watchdog's dispatch thread.

        This must stay a bounded append: the scheduling work itself happens on
        the watch loop's thread so a slow syscall can never stall dispatch.
        """
        self._pending.append((event_type, path))

    def apply_pending(self) -> None:
        """Adopt or release directories seen since the last tick.

        A move arrives as its source path plus a separate ``created`` for the
        destination, so the old watch is released and the new one adopted.
        """
        while self._pending:
            event_type, path = self._pending.popleft()
            if event_type == "created":
                self._adopt_directory(path)
            elif event_type in ("deleted", "moved"):
                self._release_directory(path)

    def _adopt_directory(self, path: str) -> None:
        candidate = os.path.abspath(path)
        if candidate in self._watches:
            return
        if os.path.dirname(candidate) not in self._shallow:
            # Anything else is already inside a recursive watch, or outside the
            # repository entirely.
            return
        try:
            relative = Path(candidate).relative_to(self._repo_root).as_posix()
        except ValueError:
            return
        if _should_ignore(relative, self._ignore_patterns):
            logger.debug("Not watching ignored directory %s", relative)
            return
        if not os.path.isdir(candidate):
            return
        if len(self._watches) >= self._max_schedules:
            if not self._budget_warned:
                self._budget_warned = True
                logger.warning(
                    "Watch budget of %d reached; %s and later directories are not watched "
                    "(raise CRG_MAX_WATCH_SCHEDULES)",
                    self._max_schedules,
                    relative,
                )
            return
        self._schedule(Path(candidate), recursive=True)
        logger.info("Watching new directory %s", relative)

    def _release_directory(self, path: str) -> None:
        candidate = os.path.abspath(path)
        handle = self._watches.pop(candidate, None)
        self._shallow.discard(candidate)
        if handle is None:
            return
        try:
            self._observer.unschedule(handle)
        except Exception as exc:  # noqa: BLE001 - watchdog raises KeyError for stale watches
            logger.debug("Could not unschedule %s: %s", candidate, exc)

    # -- liveness -------------------------------------------------------

    def _watchdog_threads(self) -> list[threading.Thread]:
        """Every thread the observer depends on, across watchdog backends.

        The dispatcher, each emitter, and any reader thread an emitter owns
        (inotify keeps its buffer thread there) can die on their own; the
        process survives all three, which is what makes the failure silent.
        """
        threads: list[threading.Thread] = []
        observer = self._observer
        if isinstance(observer, threading.Thread):
            threads.append(observer)
        try:
            emitters = list(getattr(observer, "emitters", ()) or ())
        except TypeError:  # a stub or mock observer — nothing to inspect
            return threads
        for emitter in emitters:
            if isinstance(emitter, threading.Thread):
                threads.append(emitter)
            try:
                members = list(vars(emitter).values())
            except TypeError:
                continue
            threads.extend(
                member
                for member in members
                if isinstance(member, threading.Thread) and member is not emitter
            )
        return threads

    def dead_threads(self) -> list[str]:
        """Names of watchdog threads that were running and have since died.

        Only threads previously observed alive can be reported, so an emitter
        caught between construction and start is never mistaken for a corpse.
        """
        dead: list[str] = []
        still_present: dict[int, threading.Thread] = {}
        for thread in self._watchdog_threads():
            key = id(thread)
            if thread.is_alive():
                still_present[key] = thread
            elif key in self._live_threads:
                dead.append(thread.name)
        self._live_threads = still_present
        return dead

    # -- health reporting ------------------------------------------------

    def report_health(
        self,
        *,
        observer_alive: bool,
        last_event_at: float | None = None,
        events_seen: int = 0,
        dead_threads: tuple[str, ...] = (),
        force: bool = False,
    ) -> None:
        """Publish watcher health, rate-limited to one write per interval."""
        if self._health_path is None:
            return
        now = time.time()
        state = (observer_alive, tuple(dead_threads))
        if (
            not force
            and state == self._last_health_state
            and now - self._last_health_write < _WATCH_HEALTH_INTERVAL
        ):
            return
        payload = {
            "repo": str(self._repo_root),
            "pid": os.getpid(),
            "started_at": self._started_at,
            "updated_at": now,
            "observer_alive": observer_alive,
            "last_event_at": last_event_at,
            "events_seen": events_seen,
            "watched_paths": len(self._watches),
            "dead_threads": list(dead_threads),
        }
        try:
            self._health_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._health_path.with_name(f"{self._health_path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, self._health_path)
        except OSError as exc:
            logger.debug("Could not write watcher health to %s: %s", self._health_path, exc)
            return
        self._last_health_write = now
        self._last_health_state = state

    def clear_health(self) -> None:
        """Remove the health file on a clean shutdown."""
        if self._health_path is None:
            return
        try:
            self._health_path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


def _create_watch_handler(
    repo_root: Path,
    store: GraphStore,
    on_files_updated: Optional[Callable],
    on_directory_event: Optional[Callable[[str, str], None]] = None,
):
    """Create the debounced watchdog handler for one repository."""
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.utils.event_debouncer import EventDebouncer

    ignore_patterns = _load_ignore_patterns(repo_root)
    parser = CodeParser(repo_root)
    lexical_root = Path(os.path.abspath(repo_root))
    resolved_root = lexical_root.resolve()

    class WatchBatchProcessor:
        def __init__(self) -> None:
            self.failure: BaseException | None = None
            self.last_event_at: float | None = None
            self.events_seen: int = 0

        def _relative_path(self, path: str) -> str | None:
            candidate = Path(os.path.abspath(path))
            try:
                relative = candidate.relative_to(lexical_root)
            except ValueError:
                return None
            existing = candidate
            while not existing.exists() and existing != lexical_root:
                existing = existing.parent
            try:
                existing.resolve().relative_to(resolved_root)
            except ValueError:
                return None
            if any(
                component.is_symlink()
                for component in [
                    lexical_root / Path(*relative.parts[:index])
                    for index in range(1, len(relative.parts) + 1)
                ]
            ):
                return None
            if _should_ignore(str(relative), ignore_patterns):
                return None
            return str(relative)

        def _stored_descendants(self, relative_directory: str) -> set[str]:
            # Stored file paths use POSIX separators (#774).
            directory = normalize_file_path(repo_root / relative_directory) + "/"
            return {
                str(Path(file_path).relative_to(repo_root))
                for file_path in store.get_all_files()
                if file_path.startswith(directory)
            }

        def _parseable_file(self, relative_path: str) -> bool:
            absolute_path = repo_root / relative_path
            resolved_path = absolute_path.resolve()
            try:
                resolved_path.relative_to(resolved_root)
            except ValueError:
                return False
            return (
                absolute_path.is_file()
                and not absolute_path.is_symlink()
                and parser.detect_language(absolute_path) is not None
                and not _is_binary(absolute_path)
            )

        def _parseable_descendants(self, relative_directory: str) -> set[str]:
            directory = repo_root / relative_directory
            if not directory.is_dir() or directory.is_symlink():
                return set()
            return {
                str(path.relative_to(repo_root))
                for path in directory.rglob("*")
                if self._parseable_file(str(path.relative_to(repo_root)))
                and not _should_ignore(str(path.relative_to(repo_root)), ignore_patterns)
            }

        def _event_paths(self, event: FileSystemEvent) -> set[str]:
            paths: set[str] = set()
            source = self._relative_path(os.fsdecode(event.src_path))
            destination_path = getattr(event, "dest_path", "")
            destination = (
                self._relative_path(os.fsdecode(destination_path))
                if destination_path
                else None
            )
            if event.is_directory:
                if source is not None and event.event_type in {"deleted", "moved"}:
                    paths.update(self._stored_descendants(source))
                if destination is not None:
                    paths.update(self._parseable_descendants(destination))
                elif source is not None and event.event_type == "created":
                    paths.update(self._parseable_descendants(source))
            else:
                if source is not None and event.event_type in {"deleted", "moved"}:
                    paths.add(source)
                elif source is not None and self._parseable_file(source):
                    paths.add(source)
                if destination is not None and self._parseable_file(destination):
                    paths.add(destination)
            return paths

        def process(self, events: list[FileSystemEvent]) -> None:
            # Recorded before the work so a stalled watcher is distinguishable
            # from a watcher whose repository is simply quiet.
            self.last_event_at = time.time()
            self.events_seen += len(events)
            try:
                changed_files = sorted(
                    {path for event in events for path in self._event_paths(event)}
                )
                if not changed_files:
                    return
                result = incremental_update(
                    repo_root,
                    store,
                    changed_files=changed_files,
                    reconcile_stale=False,
                )
                _raise_watch_update_errors(result, "incremental update")
                if result["files_updated"] > 0 and on_files_updated is not None:
                    postprocess_result = on_files_updated(store)
                    _raise_watch_postprocess_warnings(postprocess_result)
            except BaseException as exc:
                self.failure = exc

        def raise_if_failed(self) -> None:
            if self.failure is not None:
                raise RuntimeError("watch update failed") from self.failure

    processor = WatchBatchProcessor()
    debouncer = EventDebouncer(_DEBOUNCE_SECONDS, processor.process)

    class GraphUpdateHandler(FileSystemEventHandler):
        def dispatch(self, event: FileSystemEvent) -> None:
            if event.event_type not in {"created", "modified", "deleted", "moved"}:
                return
            if event.is_directory:
                if event.event_type == "modified":
                    return
                if on_directory_event is not None:
                    # Directories created under a non-recursive watch need a
                    # watch of their own; the supervisor decides, off-thread.
                    on_directory_event(event.event_type, os.fsdecode(event.src_path))
                    destination = getattr(event, "dest_path", "")
                    if destination:
                        on_directory_event("created", os.fsdecode(destination))
            debouncer.handle_event(event)

        def start(self) -> None:
            debouncer.start()

        def stop(self) -> None:
            debouncer.stop()
            debouncer.join()

        def process(self, events: list[FileSystemEvent]) -> None:
            processor.process(events)

        def raise_if_failed(self) -> None:
            processor.raise_if_failed()

        @property
        def last_event_at(self) -> float | None:
            return processor.last_event_at

        @property
        def events_seen(self) -> int:
            return processor.events_seen

    return GraphUpdateHandler()


def watch(
    repo_root: Path,
    store: GraphStore,
    on_files_updated: Optional[Callable] = None,
) -> None:
    """Watch for file changes and auto-update the graph.

    Uses a one-second debounce to batch rapid-fire saves into a single update.

    Ignored trees are never handed to the OS watcher, and every tick checks
    that watchdog's own threads are still running: a dead one raises, so the
    process exits non-zero and the daemon restarts it instead of the graph
    going quietly stale.  See: #811.

    Args:
        repo_root: Repository root to watch.
        store: Graph database to update.
        on_files_updated: Optional callback invoked after each debounced
            batch of file updates completes.  Receives the store as its
            only argument.  Used by the CLI to run post-processing
            (FTS, flows, communities) after watch updates.

    Raises:
        RuntimeError: if a watch update fails, or if the filesystem observer
            stops running.
    """
    from watchdog.observers import Observer

    initial = incremental_update(repo_root, store, changed_files=[])
    _raise_watch_update_errors(initial, "initial watch reconciliation")
    if initial["files_updated"] > 0 and on_files_updated is not None:
        postprocess_result = on_files_updated(store)
        _raise_watch_postprocess_warnings(postprocess_result)
    observer = Observer()
    supervisor = _WatchSupervisor(
        observer,
        repo_root,
        _load_ignore_patterns(repo_root),
        health_path=_watch_health_path(repo_root),
    )
    handler = _create_watch_handler(
        repo_root,
        store,
        on_files_updated,
        on_directory_event=supervisor.note_directory_event,
    )
    supervisor.schedule_initial(handler)
    handler.start()
    observer.start()
    supervisor.report_health(observer_alive=True, force=True)

    logger.info("Watching %s for changes... (Ctrl+C to stop)", repo_root)
    try:
        import time as _time

        while True:
            _time.sleep(1)
            handler.raise_if_failed()
            supervisor.apply_pending()
            dead = supervisor.dead_threads()
            if dead:
                names = ", ".join(dead)
                supervisor.report_health(
                    observer_alive=False,
                    last_event_at=handler.last_event_at,
                    events_seen=handler.events_seen,
                    dead_threads=tuple(dead),
                    force=True,
                )
                logger.error(
                    "Filesystem watcher thread(s) died (%s); %s would stop updating "
                    "silently, so this watcher is exiting for the daemon to restart it",
                    names,
                    repo_root,
                )
                raise RuntimeError(f"watch observer stopped: dead thread(s) {names}")
            supervisor.report_health(
                observer_alive=True,
                last_event_at=handler.last_event_at,
                events_seen=handler.events_seen,
            )
    except KeyboardInterrupt:
        supervisor.clear_health()
        _run_time_boxed(observer.stop, "observer stop")
    finally:
        _run_time_boxed(observer.stop, "observer stop")
        observer.join(timeout=_WATCH_STOP_TIMEOUT)
        handler.stop()
    logger.info("Watch stopped.")


def start_watch_thread(
    repo_root: Path,
    store: GraphStore,
    daemon: bool = True,
) -> threading.Thread | None:
    """Start watch mode in a background thread.

    Returns the started thread, or None if watchdog is unavailable.
    """
    try:
        import watchdog  # noqa: F401
    except ImportError:
        logger.warning("watchdog not installed; auto-watch disabled")
        return None

    def _run() -> None:
        # A thread cannot take the process down, so the one thing it must not
        # do is die quietly: the server would keep serving a frozen graph.
        try:
            watch(repo_root, store)
        except RuntimeError as exc:
            logger.error("Auto-watch for %s stopped: %s", repo_root, exc)

    thread = threading.Thread(
        target=_run,
        daemon=daemon,
        name="crg-watch",
    )
    thread.start()
    logger.info("Auto-watch started for %s", repo_root)
    return thread
