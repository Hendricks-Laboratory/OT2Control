'''Hardware-free tests for Auto setup transcript capture.'''

import ast
import io
import os
import unittest

from auto_terminal_transcript import AutoPreOutputTranscript


class _FakeSys:
    '''Minimal injectable sys-like object used without process-wide streams.'''

    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()


class AutoPreOutputTranscriptTests(unittest.TestCase):
    '''Verifies setup output is preserved without creating any files.'''

    def test_consume_preserves_stdout_and_stderr_then_restores_streams(self):
        fake_sys = _FakeSys()
        original_stdout = fake_sys.stdout
        original_stderr = fake_sys.stderr
        transcript = AutoPreOutputTranscript(fake_sys)

        transcript.start()
        fake_sys.stdout.write('Header setting parsed\\n')
        fake_sys.stderr.write('Header warning\\n')
        captured = transcript.consume()

        self.assertEqual(
            captured,
            'Header setting parsed\\nHeader warning\\n'
        )
        self.assertIs(fake_sys.stdout, original_stdout)
        self.assertIs(fake_sys.stderr, original_stderr)
        self.assertEqual(original_stdout.getvalue(), 'Header setting parsed\\n')
        self.assertEqual(original_stderr.getvalue(), 'Header warning\\n')

    def test_discard_restores_streams_without_needing_an_output_path(self):
        fake_sys = _FakeSys()
        original_stdout = fake_sys.stdout
        original_stderr = fake_sys.stderr
        transcript = AutoPreOutputTranscript(fake_sys)

        transcript.start()
        fake_sys.stdout.write('Declined collision\\n')
        transcript.discard()

        self.assertIs(fake_sys.stdout, original_stdout)
        self.assertIs(fake_sys.stderr, original_stderr)
        self.assertFalse(transcript.active)


class AutoTerminalTranscriptControllerPlacementTests(unittest.TestCase):
    '''Verifies Auto setup capture begins before Header parsing and folder use.'''

    def test_auto_hooks_surround_header_parsing_and_terminal_log_creation(self):
        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as source_file:
            module = ast.parse(source_file.read(), filename=controller_path)

        classes = {
            node.name: node
            for node in module.body
            if isinstance(node, ast.ClassDef)
        }
        controller_methods = {
            node.name: node
            for node in classes['Controller'].body
            if isinstance(node, ast.FunctionDef)
        }
        auto_methods = {
            node.name: node
            for node in classes['AutoContr'].body
            if isinstance(node, ast.FunctionDef)
        }

        for method_name in (
            '_start_pre_output_terminal_capture',
            '_consume_pre_output_terminal_capture',
            '_discard_pre_output_terminal_capture'
        ):
            self.assertIn(method_name, controller_methods)
            self.assertIn(method_name, auto_methods)

        controller_init = controller_methods['__init__']
        call_lines = {}
        for node in ast.walk(controller_init):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in (
                    '_start_pre_output_terminal_capture',
                    '_init_robo_header_params',
                    '_make_out_dirs'
                )
            ):
                call_lines[node.func.attr] = node.lineno

        self.assertLess(
            call_lines['_start_pre_output_terminal_capture'],
            call_lines['_init_robo_header_params']
        )
        self.assertLess(
            call_lines['_init_robo_header_params'],
            call_lines['_make_out_dirs']
        )

        start_log = controller_methods['_start_terminal_output_capture']
        consumed = [
            node for node in ast.walk(start_log)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_consume_pre_output_terminal_capture'
        ]
        self.assertEqual(len(consumed), 1)


if __name__ == '__main__':
    unittest.main()
