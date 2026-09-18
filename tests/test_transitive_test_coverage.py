"""Test-gap coverage reached through a caller (#1047).

The gate on promotion PR #1047 called ``main.py::_offload``, ``_run_off_loop``
and 71 other symbols "test gaps" while the suite executed every one of them.
TESTED_BY is a direct link from a test to a production symbol, so a private
helper that tests only reach by calling the public function above it has no
such edge and was reported as untested.

The fix walks *up* the CALLS graph for a tested caller, and keeps the answer
in its own class. These tests pin both halves: the false alarms stop, and a
symbol nothing reaches at any depth is still reported -- including the shape
that made the naive "any tested ancestor means covered" rule a net loss on
that very delta, a helper sitting under two well-tested callers whose own
body no test executes.
"""

import tempfile
from pathlib import Path

from code_review_graph.changes import analyze_changes, compute_risk_score
from code_review_graph.graph import GraphStore
from code_review_graph.parser import EdgeInfo, NodeInfo


class _Fixture:
    """Shared graph-building helpers."""

    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.store = GraphStore(self.tmp.name)

    def teardown_method(self):
        self.store.close()
        Path(self.tmp.name).unlink(missing_ok=True)

    def _add_func(
        self,
        name: str,
        path: str = "app.py",
        is_test: bool = False,
        line_start: int = 1,
        line_end: int = 10,
    ) -> str:
        node = NodeInfo(
            kind="Test" if is_test else "Function",
            name=name,
            file_path=path,
            line_start=line_start,
            line_end=line_end,
            language="python",
            parent_name=None,
            is_test=is_test,
            extra={},
        )
        self.store.upsert_node(node, file_hash="abc")
        self.store.commit()
        return f"{path}::{name}"

    def _add_call(self, source_qn: str, target_qn: str, extra: dict | None = None) -> None:
        self.store.upsert_edge(EdgeInfo(
            kind="CALLS",
            source=source_qn,
            target=target_qn,
            file_path=source_qn.split("::", 1)[0],
            line=5,
            extra=extra or {},
        ))
        self.store.commit()

    def _add_tested_by(self, production_qn: str, test_qn: str) -> None:
        # TESTED_BY is stored source=production, target=test. See: #515
        self.store.upsert_edge(EdgeInfo(
            kind="TESTED_BY",
            source=production_qn,
            target=test_qn,
            file_path=test_qn.split("::", 1)[0],
            line=1,
        ))
        self.store.commit()

    def _analyze(self, path: str = "app.py", start: int = 1, end: int = 500) -> dict:
        return analyze_changes(
            self.store,
            changed_files=[path],
            changed_ranges={path: [(start, end)]},
        )

    def _gap(self, result: dict, name: str) -> dict:
        matches = [g for g in result["test_gaps"] if g["name"] == name]
        assert len(matches) == 1, (name, result["test_gaps"])
        return matches[0]


class TestGapClassification(_Fixture):
    def _chain(self, length: int) -> str:
        """``public`` (tested) -> mid1 -> ... -> ``helper``; return helper's qn.

        ``length`` is the number of CALLS hops between the tested caller and
        the helper, so length=1 means the tested function calls it directly.
        """
        public = self._add_func("public", line_start=1, line_end=10)
        test = self._add_func("test_public", path="test_app.py", is_test=True)
        self._add_tested_by(public, test)
        previous = public
        for i in range(1, length):
            mid = self._add_func(f"mid{i}", line_start=20 + i * 10, line_end=25 + i * 10)
            self._add_call(previous, mid)
            previous = mid
        helper = self._add_func("helper", line_start=200, line_end=210)
        self._add_call(previous, helper)
        return helper

    def test_direct_coverage_is_not_a_gap_at_all(self):
        """A symbol with its own TESTED_BY edge never reaches the report."""
        public = self._add_func("public")
        test = self._add_func("test_public", path="test_app.py", is_test=True)
        self._add_tested_by(public, test)

        result = self._analyze()
        assert result["test_gaps"] == []
        assert result["test_gaps_uncovered"] == 0
        assert result["test_gaps_indirect"] == 0

    def test_helper_under_a_tested_caller_is_indirect_not_untested(self):
        """The #1047 shape: ``_offload``, reached only via its caller."""
        self._chain(1)

        result = self._analyze()
        gap = self._gap(result, "helper")
        assert gap["coverage"] == "indirect"
        assert gap["covered_via"] == "app.py::public"
        assert gap["covered_depth"] == 1
        assert gap["covered_by"] == ["test_app.py::test_public"]
        assert result["test_gaps_indirect"] == 1

    def test_two_hops_still_counts_as_reached(self):
        """``_run_off_loop`` sits two CALLS hops below its tested caller."""
        self._chain(2)

        gap = self._gap(self._analyze(), "helper")
        assert gap["coverage"] == "indirect"
        assert gap["covered_depth"] == 2
        # ``via`` names the tested caller found at the end of the walk, not
        # the intermediate hop.
        assert gap["covered_via"] == "app.py::public"

    def test_three_hops_is_past_the_limit_and_stays_a_gap(self):
        """The limit has to bite somewhere, or every symbol reads as covered."""
        self._chain(3)

        gap = self._gap(self._analyze(), "helper")
        assert gap["coverage"] == "none"
        assert "covered_via" not in gap

    def test_symbol_with_no_caller_anywhere_is_still_reported(self):
        """The case the fix must never silence: nothing reaches it at all."""
        self._add_func("orphan", line_start=1, line_end=10)

        result = self._analyze()
        gap = self._gap(result, "orphan")
        assert gap["coverage"] == "none"
        assert result["test_gaps_uncovered"] == 1
        assert result["test_gaps_indirect"] == 0

    def test_indirect_coverage_never_removes_the_row(self):
        """The real gap this rule could have hidden.

        ``incremental.py::_get_svn_changed_files`` has two direct callers, both
        heavily tested, and zero of its own 29 statements executed. Suppressing
        a symbol because a tested caller reaches it would have hidden the one
        genuine gap on the delta while fixing six false alarms.
        """
        helper = self._add_func("svn_helper", line_start=200, line_end=229)
        for i, caller in enumerate(("get_changed_files", "get_staged_and_unstaged")):
            caller_qn = self._add_func(caller, line_start=1 + i * 20, line_end=10 + i * 20)
            self._add_call(caller_qn, helper)
            test_qn = self._add_func(f"test_{caller}", path="test_app.py", is_test=True)
            self._add_tested_by(caller_qn, test_qn)

        result = self._analyze()
        gap = self._gap(result, "svn_helper")
        assert gap["coverage"] == "indirect"
        assert len(result["test_gaps"]) == 1
        assert result["test_gaps_indirect"] == 1

    def test_unreached_gaps_sort_before_indirect_ones(self):
        """Every consumer truncates the list; the real gaps must survive it."""
        self._chain(1)
        self._add_func("orphan", line_start=300, line_end=310)

        gaps = self._analyze()["test_gaps"]
        assert [g["coverage"] for g in gaps] == ["none", "indirect"]

    def test_a_test_caller_does_not_launder_coverage_onto_its_callees(self):
        """A caller with no tests of its own passes nothing down."""
        untested_caller = self._add_func("caller", line_start=1, line_end=10)
        helper = self._add_func("helper", line_start=20, line_end=30)
        self._add_call(untested_caller, helper)

        result = self._analyze()
        assert {g["name"]: g["coverage"] for g in result["test_gaps"]} == {
            "caller": "none",
            "helper": "none",
        }


class TestSummaryText(_Fixture):
    def test_summary_separates_the_two_claims(self):
        public = self._add_func("public", line_start=1, line_end=10)
        test = self._add_func("test_public", path="test_app.py", is_test=True)
        self._add_tested_by(public, test)
        helper = self._add_func("helper", line_start=20, line_end=30)
        self._add_call(public, helper)
        self._add_func("orphan", line_start=40, line_end=50)

        summary = self._analyze()["summary"]
        assert (
            "  - 2 test gap(s) (1 with no test in reach, "
            "1 reached only through a caller)"
        ) in summary
        untested_line = next(
            line for line in summary.splitlines() if "Untested:" in line
        )
        assert "orphan" in untested_line
        assert "helper" not in untested_line
        indirect_line = next(
            line for line in summary.splitlines()
            if "Reached only through a caller:" in line
        )
        assert "helper" in indirect_line

    def test_summary_is_unchanged_when_nothing_is_indirect(self):
        """No breakdown clause and no extra line when the split is trivial."""
        self._add_func("orphan", line_start=1, line_end=10)

        summary = self._analyze()["summary"]
        assert "  - 1 test gap(s)\n" in summary + "\n"
        assert "reached only through a caller" not in summary
        assert "Reached only through a caller:" not in summary


class TestWalkSafety(_Fixture):
    def test_bare_caller_names_are_not_credited(self):
        """An unresolved caller has no identity, so it lends no coverage."""
        helper = self._add_func("helper", line_start=20, line_end=30)
        self._add_call("helper_caller", helper)
        self._add_tested_by("helper_caller", "test_app.py::test_thing")

        assert self.store.get_caller_test_routes([helper]) == {}

    def test_ambiguous_call_edges_are_not_credited(self):
        """An edge naming several candidates is not a fact about any one."""
        public = self._add_func("public", line_start=1, line_end=10)
        test = self._add_func("test_public", path="test_app.py", is_test=True)
        self._add_tested_by(public, test)
        helper = self._add_func("helper", line_start=20, line_end=30)
        self._add_call(public, helper, extra={"ambiguous_targets": ["a", "b"]})

        assert self.store.get_caller_test_routes([helper]) == {}

    def test_call_cycles_terminate(self):
        """Mutual recursion with no test anywhere must not loop."""
        a = self._add_func("a", line_start=1, line_end=10)
        b = self._add_func("b", line_start=20, line_end=30)
        self._add_call(a, b)
        self._add_call(b, a)

        assert self.store.get_caller_test_routes([a, b]) == {}
        result = self._analyze()
        assert {g["coverage"] for g in result["test_gaps"]} == {"none"}

    def test_depth_zero_walks_nothing(self):
        public = self._add_func("public", line_start=1, line_end=10)
        test = self._add_func("test_public", path="test_app.py", is_test=True)
        self._add_tested_by(public, test)
        helper = self._add_func("helper", line_start=20, line_end=30)
        self._add_call(public, helper)

        assert self.store.get_caller_test_routes([helper], max_depth=0) == {}
        assert self.store.get_caller_test_routes([helper], max_depth=1) != {}

    def test_frontier_cap_bounds_the_walk(self):
        """A hub with many callers must not turn one report into a scan."""
        helper = self._add_func("helper", line_start=1, line_end=5)
        for i in range(10):
            caller = self._add_func(f"c{i:02d}", line_start=10 + i * 10, line_end=15 + i * 10)
            self._add_call(caller, helper)
        # Only the last caller (sorted) has tests, so a cap of 1 keeps the
        # first one and finds nothing.
        self._add_tested_by("app.py::c09", "test_app.py::test_c09")

        assert self.store.get_caller_test_routes([helper], max_frontier=1) == {}
        assert self.store.get_caller_test_routes([helper], max_frontier=10) != {}

    def test_batching_survives_more_seeds_than_one_sql_batch(self):
        """450 is the per-query bind limit; the walk must chunk past it."""
        public = self._add_func("public", line_start=1, line_end=5)
        self._add_tested_by(public, "test_app.py::test_public")
        seeds = []
        for i in range(600):
            helper = self._add_func(f"h{i:04d}", line_start=10 + i * 10, line_end=15 + i * 10)
            self._add_call(public, helper)
            seeds.append(helper)

        routes = self.store.get_caller_test_routes(seeds)
        assert len(routes) == 600
        assert all(r["depth"] == 1 for r in routes.values())


class TestRiskScoring(_Fixture):
    def test_indirect_credit_sits_between_untested_and_one_direct_test(self):
        """Own tests must always beat borrowed ones, or the order lies."""
        untested = self._add_func("untested", line_start=1, line_end=10)
        tested = self._add_func("tested", line_start=20, line_end=30)
        test_qn = self._add_func("test_tested", path="test_app.py", is_test=True)
        self._add_tested_by(tested, test_qn)

        untested_node = self.store.get_node(untested)
        tested_node = self.store.get_node(tested)

        plain = compute_risk_score(self.store, untested_node)
        indirect = compute_risk_score(
            self.store, untested_node, reached_by_tested_caller=True
        )
        direct = compute_risk_score(self.store, tested_node)

        assert direct < indirect < plain

    def test_indirect_credit_does_not_stack_on_direct_tests(self):
        """A symbol with its own tests gets no extra discount for a caller."""
        tested = self._add_func("tested", line_start=1, line_end=10)
        test_qn = self._add_func("test_tested", path="test_app.py", is_test=True)
        self._add_tested_by(tested, test_qn)
        node = self.store.get_node(tested)

        assert compute_risk_score(self.store, node) == compute_risk_score(
            self.store, node, reached_by_tested_caller=True
        )
