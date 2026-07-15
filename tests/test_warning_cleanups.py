import ast
import unittest
import warnings
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPOSITORY_ROOT / "controller.py"


def controller_class():
    module = ast.parse(
        CONTROLLER_PATH.read_text(encoding="utf-8"),
        filename=str(CONTROLLER_PATH),
    )
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Controller"
    )


def class_method(class_name, method_name):
    module = ast.parse(
        CONTROLLER_PATH.read_text(encoding="utf-8"),
        filename=str(CONTROLLER_PATH),
    )
    class_node = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


class WarningCleanupTests(unittest.TestCase):
    def test_controller_compiles_without_deprecated_escape_warnings(self):
        source = CONTROLLER_PATH.read_text(encoding="utf-8")
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            compile(source, str(CONTROLLER_PATH), "exec")

    def test_plot_setup_does_not_create_an_empty_legend(self):
        method = next(
            node
            for node in controller_class().body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_plot_setup_overlay"
        )
        legend_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "legend"
        ]
        self.assertEqual(legend_calls, [])

    def test_reagent_info_slices_are_copied_before_column_assignment(self):
        method = class_method("ScanDataFrame", "AddReagentInfo")
        copied_names = set()
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Attribute):
                continue
            if node.value.func.attr != "copy":
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    copied_names.add(target.id)

        self.assertIn("base", copied_names)
        self.assertIn("temp", copied_names)


if __name__ == "__main__":
    unittest.main()
