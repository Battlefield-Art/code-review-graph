"""Regression coverage for C++ and Qt header indexing (issue #463)."""

from pathlib import Path

from code_review_graph.parser import CodeParser

QT_HEADER = """#pragma once
#include <QMainWindow>

QT_BEGIN_NAMESPACE namespace Ui { class MyWidgetClass; };
QT_END_NAMESPACE

class MyWidget : public QMainWindow {
  Q_OBJECT

 public:
  MyWidget(QWidget* parent = nullptr);
  ~MyWidget();

 protected Q_SLOTS:
  void onButtonClicked();

 public Q_SLOTS:
  void onReset();

 Q_SIGNALS:
  void dataReady(int result);
  void errorOccurred(const QString& msg);
};
"""


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


def test_qt_structural_macros_do_not_hide_classes_or_become_functions(
    tmp_path: Path,
) -> None:
    _, nodes, _ = _parse(tmp_path, "MyWidget.hpp", QT_HEADER)

    class_names = {node.name for node in nodes if node.kind == "Class"}
    function_names = {node.name for node in nodes if node.kind == "Function"}

    assert {"MyWidget", "MyWidgetClass"} <= class_names
    assert function_names.isdisjoint({
        "QT_BEGIN_NAMESPACE",
        "QT_END_NAMESPACE",
        "Q_OBJECT",
        "Q_SLOTS",
        "Q_SIGNALS",
    })


def test_qt_macro_shielding_preserves_class_source_span(tmp_path: Path) -> None:
    _, nodes, _ = _parse(tmp_path, "MyWidget.hpp", QT_HEADER)

    widget = next(
        node for node in nodes if node.kind == "Class" and node.name == "MyWidget"
    )
    assert (widget.line_start, widget.line_end) == (7, 23)


def test_cpp_callable_declarations_are_indexed_without_variables(tmp_path: Path) -> None:
    _, nodes, _ = _parse(
        tmp_path,
        "Widget.hpp",
        """class Widget {
 public:
  Widget();
  ~Widget();
  void reset();
  int value() const;
  int count;
};

void top_level(int value);
extern int global_value;
""",
    )

    function_names = [node.name for node in nodes if node.kind == "Function"]
    assert function_names == ["Widget", "~Widget", "reset", "value", "top_level"]
    assert "count" not in function_names
    assert "global_value" not in function_names


def test_qt_member_declarations_survive_macro_shielding(tmp_path: Path) -> None:
    _, nodes, _ = _parse(tmp_path, "MyWidget.hpp", QT_HEADER)

    function_names = {node.name for node in nodes if node.kind == "Function"}
    assert {
        "MyWidget",
        "~MyWidget",
        "onButtonClicked",
        "onReset",
        "dataReady",
        "errorOccurred",
    } <= function_names
