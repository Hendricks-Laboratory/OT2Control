'''Hardware-free tests for the Stage 14G3 monitor-only condition contract.'''

import unittest

from auto_stability_monitor import (
    AutoStabilityMonitorValidationError,
    STABILITY_MONITOR_MODE_OFF,
    STABILITY_MONITOR_MODE_REQUIRED,
    build_stability_monitor_condition_manifest,
    parse_auto_stability_monitor_header_settings,
)


class Stage14G3MonitorContractTests(unittest.TestCase):
    def test_missing_header_defaults_off(self):
        settings = parse_auto_stability_monitor_header_settings({})
        self.assertEqual(
            STABILITY_MONITOR_MODE_OFF,
            settings['auto_stability_monitor_only']
        )

    def test_required_conditions_preserve_explicit_order_and_values(self):
        manifest = build_stability_monitor_condition_manifest(
            [
                {
                    'condition_id': 'stable_control',
                    'replicate_count': '3',
                    'silver_nitrate': '0.1',
                    'potassium_bromide': '0.002',
                    'notes': 'PI-approved control',
                },
                {
                    'condition_id': 'changing_control',
                    'replicate_count': 2,
                    'silver_nitrate': .2,
                    'potassium_bromide': .004,
                },
            ],
            ['silver_nitrate', 'potassium_bromide']
        )
        self.assertEqual(
            STABILITY_MONITOR_MODE_REQUIRED,
            parse_auto_stability_monitor_header_settings({
                'auto_stability_monitor_only': 'required'
            })['auto_stability_monitor_only']
        )
        self.assertEqual(
            ['stable_control', 'changing_control'],
            [row['condition_id'] for row in manifest['conditions']]
        )
        self.assertEqual(3, manifest['conditions'][0]['replicate_count'])
        self.assertEqual(
            .1,
            manifest['conditions'][0]['variable_concentrations_mM'][
                'silver_nitrate'
            ]
        )
        self.assertTrue(manifest['manifest_sha256'])

    def test_ambiguous_or_incomplete_rows_fail_closed(self):
        for rows in (
                [{'condition_id': 'a', 'replicate_count': 1,
                  'silver_nitrate': .1}],
                [{'condition_id': 'a', 'replicate_count': 1,
                  'silver_nitrate': -.1, 'potassium_bromide': .002}],
                [{'condition_id': 'a', 'replicate_count': 1,
                  'silver_nitrate': .1, 'potassium_bromide': .002},
                 {'condition_id': 'a', 'replicate_count': 1,
                  'silver_nitrate': .2, 'potassium_bromide': .003}],
                []):
            with self.subTest(rows=rows):
                with self.assertRaises(AutoStabilityMonitorValidationError):
                    build_stability_monitor_condition_manifest(
                        rows, ['silver_nitrate', 'potassium_bromide']
                    )


if __name__ == '__main__':
    unittest.main()
