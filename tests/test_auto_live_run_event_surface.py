'''Source-level guard that controller journal calls remain schema-approved.'''

import ast
import os
import unittest

from auto_live_run_state import EVENT_TYPES


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLLER_PATH = os.path.join(REPOSITORY_ROOT, 'controller.py')


class AutoLiveRunEventSurfaceTests(unittest.TestCase):
    '''Avoid runtime journal failures from newly added controller milestones.'''

    def test_all_literal_controller_journal_events_are_schema_approved(self):
        with open(CONTROLLER_PATH, 'r', encoding='utf-8') as source_file:
            source = source_file.read()

        tree = ast.parse(source, filename=CONTROLLER_PATH)
        event_call_shapes = {
            '_record_auto_live_run_event': 0,
            '_record_auto_live_run_transition': 1,
        }
        literal_events = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            event_position = event_call_shapes.get(node.func.attr)
            if event_position is None or len(node.args) <= event_position:
                continue

            event_argument = node.args[event_position]
            if (
                    isinstance(event_argument, ast.Constant)
                    and isinstance(event_argument.value, str)):
                literal_events.append((event_argument.value, node.lineno))

        missing_events = [
            '{} (controller.py:{})'.format(event_name, line_number)
            for event_name, line_number in literal_events
            if event_name not in EVENT_TYPES
        ]
        self.assertEqual([], missing_events)


if __name__ == '__main__':
    unittest.main()
