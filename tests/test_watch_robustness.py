"""Watch-mode robustness: ignore-aware scheduling and observer liveness.

Covers issue #811, where one recursive watch on the repository root made the
OS register a watch inside every ignored tree.  A build tool churning through
``target/`` then killed a watchdog thread, the process stayed up, and the graph
silently stopped updating while ``crg-daemon status`` still said "alive".

Every test here is deterministic: observers are fakes, and a "dead" watchdog
thread is a real thread that has been joined, never a crash we tried to
provoke through the filesystem.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_review_graph.daemon import (
    DaemonConfig,
    WatchRepo,
    read_watch_health,
    watch_health_path,
    watcher_status,
)
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import (
    _load_ignore_patterns,
    _plan_watch_paths,
    _should_ignore,
    _WatchSupervisor,
    clear_nested_ignore_cache,
    watch,
)


@pytest.fixture(autouse=True)
def _fresh_nested_ignore_cache():
    """Nested build-output patterns are cached per repo; tests build repos."""
    clear_nested_ignore_cache()
    yield
    clear_nested_ignore_cache()


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeWatch:
    """Stand-in for watchdog's ObservedWatch."""

    def __init__(self, path: str, recursive: bool) -> None:
        self.path = path
        self.is_recursive = recursive


class FakeEmitter:
    """Stand-in for a watchdog emitter that owns a backend reader thread.

    inotify keeps its buffer thread exactly like this, and that buffer thread
    is the one that dies in #811 — the emitter itself stays alive.
    """

    def __init__(self, reader: threading.Thread | None = None) -> None:
        self._inotify = reader


class FakeObserver:
    """Records what would have been handed to the OS."""

    def __init__(self) -> None:
        self.scheduled: list[tuple[str, bool]] = []
        self.unscheduled: list[str] = []
        self.emitters: list[object] = []
        self.handler = None
        self.started = False
        self.stopped = False

    def schedule(self, handler, path, *, recursive=False, event_filter=None):
        self.handler = handler
        self.scheduled.append((path, recursive))
        return FakeWatch(path, recursive)

    def unschedule(self, watch_handle) -> None:
        self.unscheduled.append(watch_handle.path)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        return None


def _live_thread() -> tuple[threading.Thread, threading.Event]:
    """A thread that stays alive until its event is set."""
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, name="fake-inotify-buffer", daemon=True)
    thread.start()
    return thread, gate


def _tick_driver(*actions):
    """Replacement for ``time.sleep`` that runs one scripted action per tick.

    After the script is exhausted the watch loop is stopped with a
    KeyboardInterrupt, which is how a real ``Ctrl+C`` leaves it.
    """
    state = {"tick": 0}

    def _sleep(_seconds):
        index = state["tick"]
        state["tick"] += 1
        if index >= len(actions):
            raise KeyboardInterrupt
        actions[index]()

    return _sleep


def _maven_repo(root: Path) -> None:
    """A monorepo shaped like the one in #811: nested modules with target/."""
    (root / ".git" / "objects").mkdir(parents=True)
    for index in range(8):
        (root / ".git" / "objects" / f"{index:02d}").mkdir()
    (root / "pom.xml").write_text("<project/>", encoding="utf-8")
    module = root / "intranet-backend"
    (module / "src" / "main" / "java").mkdir(parents=True)
    (module / "pom.xml").write_text("<project/>", encoding="utf-8")
    for index in range(6):
        (module / "target" / f"surefire-{index}").mkdir(parents=True)
    (module / "target" / "classes").mkdir(parents=True)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestIgnoreAwareScheduling:
    def test_ignored_top_level_directories_are_never_scheduled(self, tmp_path):
        """node_modules and .git must not reach the OS watch list at all."""
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        (tmp_path / ".git" / "objects").mkdir(parents=True)

        plan = _plan_watch_paths(tmp_path, _load_ignore_patterns(tmp_path))
        paths = [str(path) for path, _ in plan]

        assert (tmp_path, False) in plan, "the repo root needs a non-recursive watch"
        assert (tmp_path / "src", True) in plan
        assert not any("node_modules" in path for path in paths)
        assert not any(".git" in path for path in paths)

    def test_root_watch_is_non_recursive_so_ignored_trees_stay_out(self, tmp_path):
        """A recursive root watch would re-register everything underneath it."""
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)

        plan = _plan_watch_paths(tmp_path, _load_ignore_patterns(tmp_path))

        assert (tmp_path, True) not in plan

    def test_clean_repository_keeps_a_single_recursive_watch(self, tmp_path):
        """Nothing to exclude means nothing to split: one watch, as before."""
        (tmp_path / "src" / "inner").mkdir(parents=True)

        plan = _plan_watch_paths(tmp_path, _load_ignore_patterns(tmp_path))

        assert plan == [(tmp_path, True)]

    def test_small_ignored_directory_does_not_buy_its_own_watch(self, tmp_path):
        """A lone __pycache__ costs one OS watch; a split costs a thread."""
        (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)

        plan = _plan_watch_paths(tmp_path, _load_ignore_patterns(tmp_path))

        assert plan == [(tmp_path, True)]

    def test_plan_falls_back_to_one_recursive_watch_when_over_budget(self, tmp_path):
        """The watch count is bounded; an over-budget repo keeps the old shape."""
        for index in range(10):
            (tmp_path / f"pkg{index}").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"dep{index}").mkdir(parents=True)

        plan = _plan_watch_paths(
            tmp_path, _load_ignore_patterns(tmp_path), max_schedules=3
        )

        assert plan == [(tmp_path, True)]

    def test_nested_module_output_is_ignored_and_never_watched(self, tmp_path):
        """`moduleA/target/` is build output when `moduleA/pom.xml` says so."""
        _maven_repo(tmp_path)
        patterns = _load_ignore_patterns(tmp_path)

        assert "/intranet-backend/target/**" in patterns
        assert _should_ignore("intranet-backend/target/surefire-0/a.xml", patterns)
        assert not _should_ignore("intranet-backend/src/main/java/A.java", patterns)

        plan = _plan_watch_paths(tmp_path, patterns)
        paths = [str(path) for path, _ in plan]

        assert not any("target" in path for path in paths)
        assert (tmp_path / "intranet-backend" / "src", True) in plan

    def test_nested_output_name_without_a_manifest_keeps_its_files(self, tmp_path):
        """Root anchoring stays intact for anyone whose nested target/ is source."""
        module = tmp_path / "moduleB"
        (module / "target").mkdir(parents=True)
        (module / "target" / "handler.py").write_text("x = 1\n", encoding="utf-8")

        patterns = _load_ignore_patterns(tmp_path)

        assert "/moduleB/target/**" not in patterns
        assert not _should_ignore("moduleB/target/handler.py", patterns)

    def test_root_level_output_dirs_still_match_the_anchored_pattern(self, tmp_path):
        """The nested scan adds patterns; it never removes existing ones."""
        patterns = _load_ignore_patterns(tmp_path)

        assert _should_ignore("target/classes/A.class", patterns)
        assert _should_ignore("build/output.js", patterns)
        assert not _should_ignore("src/build/output.js", patterns)

    def test_nested_scan_can_be_disabled(self, tmp_path, monkeypatch):
        _maven_repo(tmp_path)
        monkeypatch.setenv("CRG_NESTED_OUTPUT_SCAN", "0")
        clear_nested_ignore_cache()

        patterns = _load_ignore_patterns(tmp_path)

        assert "/intranet-backend/target/**" not in patterns


class TestNewDirectoryAdoption:
    def _supervisor(self, tmp_path) -> tuple[_WatchSupervisor, FakeObserver]:
        observer = FakeObserver()
        supervisor = _WatchSupervisor(
            observer, tmp_path, _load_ignore_patterns(tmp_path), health_path=None
        )
        supervisor.schedule_initial(MagicMock())
        return supervisor, observer

    def test_new_top_level_directory_is_picked_up(self, tmp_path):
        """A non-recursive root watch only helps if new children get scheduled."""
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        supervisor, observer = self._supervisor(tmp_path)
        before = list(observer.scheduled)

        created = tmp_path / "services"
        created.mkdir()
        supervisor.note_directory_event("created", str(created))
        supervisor.apply_pending()

        assert (str(created), True) in observer.scheduled
        assert (str(created), True) not in before

    def test_new_ignored_directory_is_not_picked_up(self, tmp_path):
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        supervisor, observer = self._supervisor(tmp_path)

        created = tmp_path / "dist"
        created.mkdir()
        supervisor.note_directory_event("created", str(created))
        supervisor.apply_pending()

        assert not any(path == str(created) for path, _ in observer.scheduled)

    def test_directory_inside_a_recursive_watch_is_not_rescheduled(self, tmp_path):
        """watchdog already covers those; a second watch would be waste."""
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        supervisor, observer = self._supervisor(tmp_path)

        created = tmp_path / "src" / "nested"
        created.mkdir()
        supervisor.note_directory_event("created", str(created))
        supervisor.apply_pending()

        assert not any(path == str(created) for path, _ in observer.scheduled)

    def test_deleted_directory_releases_its_watch(self, tmp_path):
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        supervisor, observer = self._supervisor(tmp_path)

        supervisor.note_directory_event("deleted", str(tmp_path / "src"))
        supervisor.apply_pending()

        assert observer.unscheduled == [str(tmp_path / "src")]

    def test_directory_events_do_not_schedule_on_the_dispatch_thread(self, tmp_path):
        """Dispatch must stay an append; scheduling happens on the watch loop."""
        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        supervisor, observer = self._supervisor(tmp_path)
        created = tmp_path / "services"
        created.mkdir()

        supervisor.note_directory_event("created", str(created))

        assert not any(path == str(created) for path, _ in observer.scheduled)


# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------


class TestObserverLiveness:
    def test_dead_backend_reader_thread_is_reported(self, tmp_path):
        """The inotify shape: the emitter lives on, its buffer thread does not."""
        thread, gate = _live_thread()
        observer = FakeObserver()
        observer.emitters = [FakeEmitter(thread)]
        supervisor = _WatchSupervisor(observer, tmp_path, [], health_path=None)

        assert supervisor.dead_threads() == []

        gate.set()
        thread.join(timeout=5)

        assert supervisor.dead_threads() == ["fake-inotify-buffer"]

    def test_dead_emitter_thread_is_reported(self, tmp_path):
        """The Windows shape: the dispatch/emitter thread itself ends."""
        gate = threading.Event()

        class ThreadEmitter(threading.Thread):
            def __init__(self) -> None:
                super().__init__(name="fake-emitter", daemon=True)

            def run(self) -> None:
                gate.wait()

        emitter = ThreadEmitter()
        emitter.start()
        observer = FakeObserver()
        observer.emitters = [emitter]
        supervisor = _WatchSupervisor(observer, tmp_path, [], health_path=None)

        assert supervisor.dead_threads() == []

        gate.set()
        emitter.join(timeout=5)

        assert supervisor.dead_threads() == ["fake-emitter"]

    def test_unstarted_thread_is_never_mistaken_for_a_dead_one(self, tmp_path):
        """A watch scheduled mid-tick must not read as a corpse."""
        observer = FakeObserver()
        observer.emitters = [FakeEmitter(threading.Thread(target=lambda: None))]
        supervisor = _WatchSupervisor(observer, tmp_path, [], health_path=None)

        assert supervisor.dead_threads() == []
        assert supervisor.dead_threads() == []

    def test_unschedulable_observer_reports_no_deaths(self, tmp_path):
        """A mock or stub observer must not fake a death every tick."""
        supervisor = _WatchSupervisor(MagicMock(), tmp_path, [], health_path=None)

        assert supervisor.dead_threads() == []


class TestWatchLoop:
    def _watch_with(self, tmp_path, store, observer, sleeper, health_interval=0.0, **kwargs):
        with (
            patch("watchdog.observers.Observer", return_value=observer),
            patch("time.sleep", side_effect=sleeper),
            patch(
                "code_review_graph.incremental._WATCH_HEALTH_INTERVAL",
                health_interval,
            ),
        ):
            watch(tmp_path, store, **kwargs)

    def test_dead_observer_exits_loudly_instead_of_stalling(self, tmp_path, caplog):
        """The whole point of #811: never keep running with a dead watcher."""
        (tmp_path / "src").mkdir()
        thread, gate = _live_thread()
        observer = FakeObserver()
        observer.emitters = [FakeEmitter(thread)]
        store = GraphStore(tmp_path / "graph.db")

        def kill_the_reader():
            gate.set()
            thread.join(timeout=5)

        try:
            with caplog.at_level(logging.ERROR):
                with pytest.raises(RuntimeError, match="watch observer stopped"):
                    self._watch_with(
                        tmp_path,
                        store,
                        observer,
                        _tick_driver(lambda: None, kill_the_reader),
                    )
        finally:
            store.close()

        assert "fake-inotify-buffer" in caplog.text
        assert str(tmp_path) in caplog.text
        assert observer.stopped is True

        health = read_watch_health(tmp_path)
        assert health is not None
        assert health["observer_alive"] is False
        assert health["stalled"] is True
        assert health["dead_threads"] == ["fake-inotify-buffer"]

    def test_cli_watch_turns_a_dead_observer_into_exit_code_1(self):
        """The daemon restarts on process exit, so the exit code has to be non-zero."""
        from code_review_graph import cli

        argv = ["code-review-graph", "watch", "--repo", "repo-root"]
        with (
            patch.object(sys, "argv", argv),
            patch("code_review_graph.graph.GraphStore", return_value=MagicMock()),
            patch("code_review_graph.incremental.get_db_path", return_value=MagicMock()),
            patch(
                "code_review_graph.incremental.watch",
                side_effect=RuntimeError("watch observer stopped: dead thread(s) Thread-3"),
            ),
            pytest.raises(SystemExit) as exit_info,
        ):
            cli.main()

        assert exit_info.value.code == 1

    def test_live_observer_publishes_health_for_daemon_status(self, tmp_path):
        (tmp_path / "src").mkdir()
        thread, gate = _live_thread()
        observer = FakeObserver()
        observer.emitters = [FakeEmitter(thread)]
        store = GraphStore(tmp_path / "graph.db")
        seen: list[dict] = []

        try:
            self._watch_with(
                tmp_path,
                store,
                observer,
                _tick_driver(
                    lambda: seen.append(read_watch_health(tmp_path) or {}),
                ),
            )
        finally:
            gate.set()
            store.close()

        assert seen and seen[0]["observer_alive"] is True
        assert seen[0]["stalled"] is False
        assert seen[0]["watched_paths"] >= 1
        # A clean Ctrl+C removes the file it published.
        assert not watch_health_path(tmp_path).exists()

    def test_normal_file_change_still_updates_the_graph(self, tmp_path):
        """End to end: scheduling changes must not break ordinary updates."""
        from watchdog.events import FileCreatedEvent

        (tmp_path / "src").mkdir()
        for index in range(6):
            (tmp_path / "node_modules" / f"pkg{index}").mkdir(parents=True)
        source = tmp_path / "src" / "app.py"
        source.write_text("def handler():\n    return 1\n", encoding="utf-8")

        observer = FakeObserver()
        store = GraphStore(tmp_path / "graph.db")
        callbacks: list[int] = []
        health: dict[str, object] = {}

        def deliver_event():
            observer.handler.process([FileCreatedEvent(str(source))])

        def capture_health():
            health.update(json.loads(watch_health_path(tmp_path).read_text("utf-8")))

        try:
            self._watch_with(
                tmp_path,
                store,
                observer,
                _tick_driver(deliver_event, capture_health),
                on_files_updated=lambda _store: callbacks.append(1),
            )
            assert store.get_nodes_by_file(str(source)), "watch never parsed the change"
        finally:
            store.close()

        assert callbacks, "the post-processing callback never ran"
        assert health["observer_alive"] is True
        assert isinstance(health["last_event_at"], float)
        assert health["events_seen"] == 1
        # src is watched, node_modules is not.
        assert (str(tmp_path / "src"), True) in observer.scheduled
        assert not any("node_modules" in path for path, _ in observer.scheduled)

    def test_health_is_not_rewritten_on_every_tick(self, tmp_path):
        """Watch mode runs for days; the heartbeat must stay rate-limited."""
        (tmp_path / "src").mkdir()
        supervisor = _WatchSupervisor(
            FakeObserver(), tmp_path, [], health_path=tmp_path / "health.json"
        )
        supervisor.report_health(observer_alive=True, force=True)
        first = (tmp_path / "health.json").read_text(encoding="utf-8")

        for _ in range(30):
            supervisor.report_health(observer_alive=True, last_event_at=time.time())

        assert (tmp_path / "health.json").read_text(encoding="utf-8") == first

        supervisor.report_health(observer_alive=False, dead_threads=("Thread-3",))

        assert (tmp_path / "health.json").read_text(encoding="utf-8") != first


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


def _write_health(repo_root: Path, **fields) -> None:
    payload = {
        "repo": str(repo_root),
        "pid": 4242,
        "started_at": time.time(),
        "updated_at": time.time(),
        "observer_alive": True,
        "last_event_at": time.time(),
        "events_seen": 3,
        "watched_paths": 4,
        "dead_threads": [],
    }
    payload.update(fields)
    path = watch_health_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestStatusSurfacesStalls:
    def test_missing_health_reads_as_unknown(self, tmp_path):
        assert read_watch_health(tmp_path) is None
        assert watcher_status(True, None) == "unknown"

    def test_corrupt_health_file_is_ignored(self, tmp_path):
        path = watch_health_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert read_watch_health(tmp_path) is None

    def test_dead_observer_reads_as_stalled(self, tmp_path):
        _write_health(tmp_path, observer_alive=False, dead_threads=["Thread-3"])

        health = read_watch_health(tmp_path)

        assert health is not None
        assert health["stalled"] is True
        assert watcher_status(True, health) == "stalled"

    def test_frozen_heartbeat_reads_as_stalled(self, tmp_path):
        _write_health(tmp_path, updated_at=time.time() - 600)

        health = read_watch_health(tmp_path)

        assert health is not None
        assert health["stalled"] is True
        assert health["age"] > 500

    def test_fresh_heartbeat_reads_as_ok(self, tmp_path):
        _write_health(tmp_path)

        health = read_watch_health(tmp_path)

        assert health is not None
        assert health["stalled"] is False
        assert watcher_status(True, health) == "ok"

    def test_daemon_status_reports_the_stall(self, tmp_path):
        from code_review_graph.daemon import WatchDaemon

        repo = tmp_path / "repo"
        repo.mkdir()
        _write_health(repo, observer_alive=False, dead_threads=["Thread-3"])
        config = DaemonConfig(
            repos=[WatchRepo(path=str(repo), alias="repo")],
            log_dir=tmp_path / "logs",
        )
        daemon = WatchDaemon(config=config, config_path=tmp_path / "watch.toml")

        with patch(
            "code_review_graph.daemon.load_state",
            return_value={"repo": {"pid": 4242, "path": str(repo)}},
        ), patch("code_review_graph.daemon._is_pid_alive", return_value=True):
            entry = daemon.status()["repos"][0]

        assert entry["alive"] is True, "the process really is still running"
        assert entry["watcher"] == "stalled"
        assert entry["observer_alive"] is False

    def test_daemon_cli_status_prints_a_stalled_watcher(self, tmp_path):
        from code_review_graph.daemon_cli import _handle_status

        repo = tmp_path / "repo"
        repo.mkdir()
        _write_health(repo, observer_alive=False, dead_threads=["Thread-3"])
        config = DaemonConfig(
            repos=[WatchRepo(path=str(repo), alias="repo")],
            log_dir=tmp_path / "logs",
        )

        with (
            patch("code_review_graph.daemon.is_daemon_running", return_value=True),
            patch("code_review_graph.daemon.load_config", return_value=config),
            patch("code_review_graph.daemon.read_pid", return_value=4242),
            patch(
                "code_review_graph.daemon.load_state",
                return_value={"repo": {"pid": 4242, "path": str(repo)}},
            ),
            patch("code_review_graph.daemon.pid_alive", return_value=True),
            patch("builtins.print") as printer,
        ):
            _handle_status(MagicMock())

        printed = "\n".join(str(call) for call in printer.call_args_list)
        assert "stalled" in printed
        assert "alive" in printed, "the process column still says alive"

    def test_daemon_health_check_warns_about_a_stalled_watcher(self, tmp_path, caplog):
        from code_review_graph.daemon import WatchDaemon

        repo = tmp_path / "repo"
        repo.mkdir()
        _write_health(repo, observer_alive=False)
        config = DaemonConfig(
            repos=[WatchRepo(path=str(repo), alias="repo")],
            log_dir=tmp_path / "logs",
        )
        daemon = WatchDaemon(config=config, config_path=tmp_path / "watch.toml")
        child = MagicMock()
        child.poll.return_value = None
        daemon._current_repos = {"repo": config.repos[0]}
        daemon._children = {"repo": child}

        with caplog.at_level(logging.WARNING):
            with patch.object(WatchDaemon, "_start_watcher") as restart:
                daemon._check_health()

        assert "stalled" in caplog.text
        restart.assert_not_called()
