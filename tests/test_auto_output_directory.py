'''Hardware-free tests for safe Auto output-directory resolution.'''

import ast
import os
import unittest

from auto_output_directory import (
    AutoOutputDirectoryConflictError,
    resolve_auto_output_directory
)


class AutoOutputDirectoryTests(unittest.TestCase):
    '''Verifies collision approval without creating any directories.'''

    OUTPUT_ROOT = os.path.join(os.sep, 'tmp', 'Protocol_Outputs')

    @staticmethod
    def _exists_for(paths):
        normalized_paths = {os.path.abspath(path) for path in paths}
        return lambda path: os.path.abspath(path) in normalized_paths

    def test_new_output_directory_is_retained_without_prompt(self):
        result = resolve_auto_output_directory(
            self.OUTPUT_ROOT,
            'RTG_020',
            input_func=lambda prompt: self.fail('prompt must not be called'),
            interactive=False,
            path_exists=self._exists_for([])
        )

        self.assertEqual(result['requested_data_dir'], 'RTG_020')
        self.assertEqual(result['effective_data_dir'], 'RTG_020')
        self.assertFalse(result['was_renamed_for_collision'])

    def test_existing_directory_requires_approval_for_first_unused_suffix(self):
        existing = [
            os.path.join(self.OUTPUT_ROOT, 'RTG_020'),
            os.path.join(self.OUTPUT_ROOT, 'RTG_020_1')
        ]
        prompts = []
        result = resolve_auto_output_directory(
            self.OUTPUT_ROOT,
            'RTG_020',
            input_func=lambda prompt: prompts.append(prompt) or 'yes',
            interactive=True,
            path_exists=self._exists_for(existing)
        )

        self.assertEqual(result['effective_data_dir'], 'RTG_020_2')
        self.assertTrue(result['was_renamed_for_collision'])
        self.assertEqual(len(prompts), 1)
        self.assertIn('RTG_020_2', prompts[0])

    def test_existing_directory_fails_closed_without_interactive_approval(self):
        existing = [os.path.join(self.OUTPUT_ROOT, 'RTG_020')]

        with self.assertRaisesRegex(
            AutoOutputDirectoryConflictError,
            'interactive terminal'
        ):
            resolve_auto_output_directory(
                self.OUTPUT_ROOT,
                'RTG_020',
                input_func=lambda prompt: self.fail('prompt must not be called'),
                interactive=False,
                path_exists=self._exists_for(existing)
            )

    def test_declining_proposed_name_creates_no_approved_result(self):
        existing = [os.path.join(self.OUTPUT_ROOT, 'RTG_020')]

        with self.assertRaisesRegex(
            AutoOutputDirectoryConflictError,
            'not approved'
        ):
            resolve_auto_output_directory(
                self.OUTPUT_ROOT,
                'RTG_020',
                input_func=lambda prompt: 'no',
                interactive=True,
                path_exists=self._exists_for(existing)
            )

    def test_absolute_and_parent_escape_paths_are_rejected(self):
        for invalid_name in (
            None,
            '',
            '.',
            '..',
            '../RTG_020',
            '/tmp/RTG_020'
        ):
            with self.subTest(invalid_name=invalid_name):
                with self.assertRaises(AutoOutputDirectoryConflictError):
                    resolve_auto_output_directory(
                        self.OUTPUT_ROOT,
                        invalid_name,
                        input_func=lambda prompt: 'yes',
                        interactive=True,
                        path_exists=self._exists_for([])
                    )


class AutoOutputDirectoryControllerPlacementTests(unittest.TestCase):
    '''Verifies the Auto-only resolver is reached before directory creation.'''

    def test_auto_overrides_the_base_output_directory_hook(self):
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

        self.assertIn('_resolve_output_directory_path', controller_methods)
        self.assertIn('_resolve_output_directory_path', auto_methods)
        make_out_dirs = controller_methods['_make_out_dirs']
        called_attributes = [
            node.func.attr
            for node in ast.walk(make_out_dirs)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn('_resolve_output_directory_path', called_attributes)

        resolver_call_lines = [
            node.lineno
            for node in ast.walk(make_out_dirs)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == '_resolve_output_directory_path'
        ]
        directory_creation_lines = [
            node.lineno
            for node in ast.walk(make_out_dirs)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'makedirs'
        ]
        self.assertEqual(len(resolver_call_lines), 1)
        self.assertTrue(directory_creation_lines)
        self.assertLess(
            resolver_call_lines[0],
            min(directory_creation_lines)
        )


if __name__ == '__main__':
    unittest.main()
