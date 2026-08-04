'''Static contract checks for the non-mutating Stage 9B Pi reservation path.'''

import ast
import os
import unittest


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_PATH = os.path.join(REPOSITORY_ROOT, 'ot2_robot.py')
ARMCHAIR_PATH = os.path.join(REPOSITORY_ROOT, 'Armchair', 'armchair.py')


class AutoPreparationReservationContractTests(unittest.TestCase):
    '''Keep the Stage 9B packet and no-liquid boundary explicit and reviewable.'''

    @classmethod
    def setUpClass(cls):
        with open(ROBOT_PATH, 'r', encoding='utf-8') as source_file:
            cls.robot_source = source_file.read()
        robot_tree = ast.parse(cls.robot_source, filename=ROBOT_PATH)
        cls.robot_class = next(
            node for node in robot_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'OT2Robot'
        )
        cls.methods = {
            node.name: node
            for node in cls.robot_class.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_stage_9b_methods_exist_once(self):
        for method_name in (
                '_validate_auto_preparation_reservation_request',
                '_auto_preparation_destination_candidates',
                '_build_auto_preparation_group_reservation',
                '_exec_reserve_auto_preparation_groups'):
            self.assertIn(method_name, self.methods)
            self.assertEqual(
                1,
                sum(
                    isinstance(node, ast.FunctionDef) and node.name == method_name
                    for node in self.robot_class.body
                )
            )

    def test_reservation_uses_copied_deck_view_and_stock_temperature_policy(self):
        source = ast.get_source_segment(
            self.robot_source,
            self.methods['_build_auto_preparation_group_reservation']
        )
        candidates = ast.get_source_segment(
            self.robot_source,
            self.methods['_auto_preparation_destination_candidates']
        )
        self.assertIn("'ColdWaterC1.0' if stock_is_cold", source)
        self.assertIn("'WaterC1.0'", source)
        self.assertGreaterEqual(source.count('_simulate_preflight_source('), 2)
        self.assertNotIn('self._exec_transfer(', source)
        self.assertNotIn('self._exec_make(', source)
        self.assertNotIn('self._exec_mix(', source)
        self.assertNotIn('self._exec_init_containers(', source)
        self.assertIn('list(', candidates)
        self.assertNotIn('pop_next_well(', candidates)

    def test_packet_codes_and_compatibility_version_are_registered(self):
        with open(ARMCHAIR_PATH, 'r', encoding='utf-8') as source_file:
            armchair_tree = ast.parse(source_file.read(), filename=ARMCHAIR_PATH)
        armchair_class = next(
            node for node in armchair_tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'Armchair'
        )
        assignments = {
            target.id: node.value
            for node in armchair_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        packet_types = ast.literal_eval(assignments['PACK_TYPES'].args[0])
        ghost_types = ast.literal_eval(assignments['GHOST_TYPES'])
        self.assertEqual(
            b'\x1B', packet_types['reserve_auto_preparation_groups']
        )
        self.assertEqual(
            b'\x1C', packet_types['auto_preparation_groups_reserved']
        )
        self.assertIn('reserve_auto_preparation_groups', ghost_types)
        self.assertIn('auto_preparation_groups_reserved', ghost_types)
        self.assertIn(
            "AUTO_MAIN_PROTOCOL_VERSION = 'auto-main-state-v6'",
            self.robot_source
        )


if __name__ == '__main__':
    unittest.main()
