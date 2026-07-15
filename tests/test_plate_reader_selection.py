import ast
import os
import types
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PATH = REPOSITORY_ROOT / "controller.py"


class DummyReaderStub:
    def __init__(self, data_path):
        self.data_path = data_path


class PlateReaderStub:
    def __init__(self, data_path, header_data, eve_files_path, simulate=False):
        self.data_path = data_path
        self.header_data = header_data
        self.eve_files_path = eve_files_path
        self.simulate = simulate


def load_init_plate_reader_method():
    """Load Controller._init_pr without importing legacy third-party dependencies."""
    source = CONTROLLER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(CONTROLLER_PATH))
    controller_class = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Controller"
    )
    method = next(
        node
        for node in controller_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_pr"
    )
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "os": os,
        "DummyReader": DummyReaderStub,
        "PlateReader": PlateReaderStub,
    }
    exec(compile(isolated_module, str(CONTROLLER_PATH), "exec"), namespace)
    return namespace["_init_pr"]


class PlateReaderSelectionTests(unittest.TestCase):
    def setUp(self):
        self.harness = types.SimpleNamespace(
            out_path="output",
            header_data=[["header"]],
            eve_files_path="eve-files",
        )
        self.init_plate_reader = load_init_plate_reader_method()

    def initialize(self, simulate, no_pr):
        self.init_plate_reader(self.harness, simulate=simulate, no_pr=no_pr)
        return self.harness.pr

    def test_simulation_always_uses_dummy_reader(self):
        reader = self.initialize(simulate=True, no_pr=False)
        self.assertIsInstance(reader, DummyReaderStub)

    def test_no_pr_always_uses_dummy_reader(self):
        reader = self.initialize(simulate=False, no_pr=True)
        self.assertIsInstance(reader, DummyReaderStub)

    def test_live_run_uses_physical_reader_path(self):
        reader = self.initialize(simulate=False, no_pr=False)
        self.assertIsInstance(reader, PlateReaderStub)
        self.assertFalse(reader.simulate)


if __name__ == "__main__":
    unittest.main()
