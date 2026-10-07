'''Hardware-free tests for Auto setup transcript capture.'''

import ast
import contextlib
import functools
import io
import os
import sys
import traceback
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

    @staticmethod
    def _load_terminal_output_capture_guard():
        '''Loads the decorator alone without importing hardware dependencies.'''
        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as source_file:
            source = source_file.read()
        module = ast.parse(source, filename=controller_path)
        guard = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'terminal_output_capture_guard'
        )
        namespace = {
            'functools': functools,
            'sys': sys,
            'traceback': traceback
        }
        exec(
            compile(
                ast.Module(body=[guard], type_ignores=[]),
                controller_path,
                'exec'
            ),
            namespace
        )
        return namespace['terminal_output_capture_guard']

    def test_run_guard_writes_traceback_before_capture_is_stopped(self):
        '''A saved terminal transcript contains the root failure traceback.'''
        guard = self._load_terminal_output_capture_guard()

        class FakeController:
            def __init__(self):
                self.terminal_log_file_handle = object()
                self.capture_stopped = False

            def _stop_terminal_output_capture(self):
                self.capture_stopped = True

        @guard
        def fail_run(_):
            raise ValueError('synthetic Auto failure')

        controller = FakeController()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stderr(captured_stderr):
            with self.assertRaisesRegex(ValueError, 'synthetic Auto failure'):
                fail_run(controller)

        self.assertTrue(controller.capture_stopped)
        self.assertIn(
            'unhandled Auto run traceback follows',
            captured_stderr.getvalue()
        )
        self.assertIn(
            'ValueError: synthetic Auto failure',
            captured_stderr.getvalue()
        )

    def test_run_guard_captures_unhandled_traceback_before_closing_log(self):
        '''The saved Auto transcript must retain a final traceback on failure.'''
        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as source_file:
            source = source_file.read()
        module = ast.parse(source, filename=controller_path)
        guard = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'terminal_output_capture_guard'
        )
        wrapper = next(
            node for node in guard.body
            if isinstance(node, ast.FunctionDef)
            and node.name == 'wrapper'
        )
        guarded_try = next(
            node for node in ast.walk(wrapper)
            if isinstance(node, ast.Try)
        )

        exception_writes = [
            node for handler in guarded_try.handlers
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'traceback'
            and node.func.attr == 'print_exception'
        ]
        self.assertEqual(1, len(exception_writes))
        self.assertTrue(guarded_try.finalbody)
        self.assertTrue(any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'stop_capture'
            for node in ast.walk(ast.Module(
                body=guarded_try.finalbody,
                type_ignores=[]
            ))
        ))

    def test_auto_hooks_surround_header_parsing_and_terminal_log_creation(self):
        controller_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'controller.py'
        )
        with open(controller_path, 'r', encoding='utf-8') as source_file:
            source = source_file.read()
        module = ast.parse(source, filename=controller_path)

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

        start_log_source = ast.get_source_segment(
            source,
            start_log
        )
        self.assertIn('terminal transcript initialized', start_log_source)
        self.assertIn('buffered pre-output setup transcript', start_log_source)
        self.assertIn("'follows.\\n'", start_log_source)
        self.assertIn('live terminal capture active', start_log_source)


if __name__ == '__main__':
    unittest.main()
