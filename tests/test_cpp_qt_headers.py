"""Regression coverage for C++ and Qt header indexing (issue #463)."""

from pathlib import Path

from code_review_graph.parser import CodeParser


def _parse(tmp_path: Path, name: str, source: str):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path, *CodeParser().parse_file(path)


def _file_language(nodes) -> str:
    return next(node.language for node in nodes if node.kind == "File")


def test_h_file_uses_cpp_when_source_has_strong_cpp_evidence(tmp_path: Path) -> None:
    _, nodes, _ = _parse(
        tmp_path,
        "MyWidgetPlain.h",
        """#pragma once

class MyWidgetPlain {
 public:
  void reset();
};
""",
    )

    assert _file_language(nodes) == "cpp"
    assert any(node.kind == "Class" and node.name == "MyWidgetPlain" for node in nodes)


def test_h_file_without_cpp_evidence_remains_c(tmp_path: Path) -> None:
    _, nodes, _ = _parse(
        tmp_path,
        "plain.h",
        """#pragma once

typedef struct record {
  int value;
} record;

int read_record(const record *value);
""",
    )

    assert _file_language(nodes) == "c"


def test_c_header_cpp_compatibility_guard_is_not_cpp_evidence(tmp_path: Path) -> None:
    _, nodes, _ = _parse(
        tmp_path,
        "compat.h",
        """#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int library_version(void);

#ifdef __cplusplus
}
#endif
""",
    )

    assert _file_language(nodes) == "c"
