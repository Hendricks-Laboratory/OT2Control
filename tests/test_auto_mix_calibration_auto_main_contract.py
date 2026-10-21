'''Read-only cross-branch packet contract checks for mix calibration.

These tests intentionally inspect the local ``Auto-main`` Git revision rather
than import it. They never contact a Pi, construct a robot, or open a socket.
They prove only that the paired repository branches retain the exact command,
acknowledgement, packet-code, handler, and snapshot contracts required by the
calibration controller before a supervised water debug is considered.
'''

import os
import pathlib
import subprocess
import unittest


_PACKETS = (
    ('preflight_auto_mix_calibration', "b'\\x21'"),
    ('auto_mix_calibration_preflight', "b'\\x22'"),
    ('mix_auto_calibration_well', "b'\\x23'"),
    ('auto_calibration_well_mixed', "b'\\x24'"),
    ('bind_auto_mix_calibration_well', "b'\\x25'"),
    ('auto_mix_calibration_well_bound', "b'\\x26'"),
    ('preflight_auto_mix_calibration_resources', "b'\\x27'"),
    ('auto_mix_calibration_resources_preflight', "b'\\x28'"),
    ('transfer_auto_mix_calibration_source', "b'\\x29'"),
    ('auto_mix_calibration_source_transferred', "b'\\x2A'"),
)


def _repository_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _read_auto_main(path):
    worktree = os.environ.get('OT2CONTROL_AUTO_MAIN_WORKTREE')
    if worktree:
        source_path = pathlib.Path(worktree).resolve() / path
        if not source_path.is_file():
            raise AssertionError(
                'Configured Auto-main worktree is missing {}: {}'
                .format(path, source_path)
            )
        return source_path.read_text(encoding='utf-8')
    result = subprocess.run(
        ['git', 'show', 'Auto-main:{}'.format(path)],
        cwd=str(_repository_root()), check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode:
        raise AssertionError(
            'Unable to read local Auto-main {}: {}'
            .format(path, result.stderr.strip())
        )
    return result.stdout


class AutoMixCalibrationAutoMainContractTests(unittest.TestCase):
    def test_controller_passes_only_the_preserved_deck_positions_well(self):
        source = (_repository_root() / 'controller.py').read_text(
            encoding='utf-8'
        )
        self.assertIn(
            "self.calibration_deck_reader_well = self.robo_params.get(",
            source
        )
        self.assertIn(
            "'platereader_input_first_usable'", source
        )
        self.assertIn(
            'deck_reader_well=self.calibration_deck_reader_well', source
        )

    def test_auto_main_registers_exact_ghost_packet_pairs(self):
        source = _read_auto_main('Armchair/armchair.py')
        for name, code in _PACKETS:
            self.assertIn("'{}':{}".format(name, code), source)
            self.assertIn("'{}'".format(name), source)

    def test_auto_main_registers_handlers_and_snapshot_capabilities(self):
        source = _read_auto_main('ot2_robot.py')
        commands = (
            'preflight_auto_mix_calibration',
            'mix_auto_calibration_well',
            'bind_auto_mix_calibration_well',
            'preflight_auto_mix_calibration_resources',
            'transfer_auto_mix_calibration_source',
        )
        acknowledgements = (
            'auto_mix_calibration_preflight',
            'auto_calibration_well_mixed',
            'auto_mix_calibration_well_bound',
            'auto_mix_calibration_resources_preflight',
            'auto_mix_calibration_source_transferred',
        )
        self.assertIn("'auto_mix_calibration_protocol_version'", source)
        self.assertIn("'auto-mix-calibration-v4'", source)
        for command in commands:
            self.assertIn("@exec_func('{}'".format(command), source)
            self.assertIn("'{}'".format(command), source)
        for acknowledgement in acknowledgements:
            self.assertIn("'{}'".format(acknowledgement), source)

    def test_calibration_source_transfer_cannot_use_generic_fallback(self):
        source = _read_auto_main('ot2_robot.py')
        self.assertIn(
            'def _exec_transfer_auto_mix_calibration_source', source
        )
        self.assertIn('allow_auto_make=False', source)
        self.assertIn('isinstance(source, (Well96, MultiContainer))', source)
        controller = (_repository_root() / 'controller.py').read_text(
            encoding='utf-8'
        )
        adapter_start = controller.index(
            'class AutoMixCalibrationExecutionAdapter'
        )
        adapter_end = controller.index('class AutoMixCalibrationContr',
                                       adapter_start)
        adapter_source = controller[adapter_start:adapter_end]
        self.assertIn('request_pi_transfer', adapter_source)
        self.assertNotIn("send_pack(\n            'transfer'", adapter_source)


if __name__ == '__main__':
    unittest.main()
