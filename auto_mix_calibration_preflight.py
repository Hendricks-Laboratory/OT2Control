'''Hardware-free preflight bridge for supervised well-mix calibration.

This module owns the required no-liquid Pi handshake for one already-built
calibration plan.  It deliberately does not import the controller, reader,
Armchair, or OT-2 implementation.  A thin controller transport may supply
the three request/response operations later, while this module keeps their
ordering and validation independently testable.
'''

import copy
import datetime
import json
import os
from tempfile import NamedTemporaryFile

from auto_mix_calibration_plan import validate_auto_mix_calibration_run_plan
from auto_mix_calibration_protocol import (
    validate_mix_calibration_preflight_acknowledgement,
)
from auto_mix_calibration_resources import (
    build_auto_mix_calibration_resource_preflight_request,
    build_auto_mix_calibration_well_binding_request,
    validate_auto_mix_calibration_resource_capability,
    validate_auto_mix_calibration_resource_preflight_acknowledgement,
    validate_auto_mix_calibration_well_binding_acknowledgement,
)


MIX_CALIBRATION_PREFLIGHT_RECORD_VERSION = 1


class AutoMixCalibrationPreflightError(RuntimeError):
    '''Raised when calibration preflight cannot safely reach execution.'''


def _require_reader_scan_protocol(reader_scan_protocol):
    '''Require the existing reader protocol name without invoking the reader.

    The legacy reader API does not expose a read-only wavelength query.  This
    stage can therefore verify only that a named established protocol exists.
    C2eB must prove 520/730 nm coverage from the first immutable raw scan;
    this helper must not claim otherwise.
    '''
    if not isinstance(reader_scan_protocol, str) or not reader_scan_protocol.strip():
        raise AutoMixCalibrationPreflightError(
            'Calibration requires one configured existing reader scan '
            'protocol before any Pi preflight packet is sent.'
        )
    return reader_scan_protocol.strip()


def _require_transport_methods(transport):
    required = (
        'get_robot_state_snapshot', 'request_pi_preflight',
        'request_pi_well_binding', 'request_pi_resource_preflight'
    )
    for method_name in required:
        if not callable(getattr(transport, method_name, None)):
            raise AutoMixCalibrationPreflightError(
                'Calibration preflight transport lacks {}().'.format(
                    method_name
                )
            )


def _require_accepted(acknowledgement, label):
    if not acknowledgement['accepted']:
        raise AutoMixCalibrationPreflightError(
            'Pi rejected calibration {} before liquid handling: {}.'.format(
                label, acknowledgement['message']
            )
        )
    return acknowledgement


def build_auto_mix_calibration_preflight_record(
        plan, reader_scan_protocol, snapshot, pi_preflight,
        well_binding, resource_preflight):
    '''Build the durable, execution-free evidence record for one assay plan.'''
    plan = validate_auto_mix_calibration_run_plan(plan)
    reader_scan_protocol = _require_reader_scan_protocol(reader_scan_protocol)
    return {
        'record_version': MIX_CALIBRATION_PREFLIGHT_RECORD_VERSION,
        'record_type': 'auto_mix_calibration_preflight',
        'created_at_utc': datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        'calibration_id': plan['calibration_id'],
        'logical_wellname': plan['logical_wellname'],
        'assay_profile': plan['assay_profile'],
        'target': copy.deepcopy(plan['target']),
        'plate_mapping_revision': plan['plate_mapping_revision'],
        'plate_generation': plan['plate_generation'],
        'reader_scan_protocol': reader_scan_protocol,
        'reader_wavelength_coverage_status': (
            'deferred_until_first_raw_scan'
        ),
        'expected_scan_count': plan['expected_scan_count'],
        'pi_snapshot': copy.deepcopy(snapshot),
        'pi_preflight': copy.deepcopy(pi_preflight),
        'well_binding': copy.deepcopy(well_binding),
        'resource_preflight': copy.deepcopy(resource_preflight),
        'physical_execution_started': False,
        'automatic_retry_permitted': False,
    }


def write_auto_mix_calibration_preflight_record(record, record_path):
    '''Atomically save an already-accepted calibration preflight record.'''
    if not isinstance(record, dict):
        raise AutoMixCalibrationPreflightError(
            'Calibration preflight record must be a dictionary.'
        )
    if (record.get('record_version') != MIX_CALIBRATION_PREFLIGHT_RECORD_VERSION
            or record.get('record_type') != 'auto_mix_calibration_preflight'
            or record.get('physical_execution_started') is not False):
        raise AutoMixCalibrationPreflightError(
            'Refusing to write an invalid calibration preflight record.'
        )
    if not isinstance(record_path, str) or not record_path.strip():
        raise AutoMixCalibrationPreflightError(
            'Calibration preflight record path must be nonblank.'
        )
    destination = os.path.abspath(record_path)
    destination_dir = os.path.dirname(destination)
    if not os.path.isdir(destination_dir):
        raise AutoMixCalibrationPreflightError(
            'Calibration preflight record directory does not exist: {}.'
            .format(destination_dir)
        )
    temporary_path = None
    try:
        with NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=destination_dir,
                prefix='.auto_mix_calibration_preflight_', suffix='.tmp',
                delete=False) as temporary_file:
            temporary_path = temporary_file.name
            json.dump(record, temporary_file, indent=2, sort_keys=True)
            temporary_file.write('\n')
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except (IOError, OSError, TypeError, ValueError) as exc:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise AutoMixCalibrationPreflightError(
            'Could not write calibration preflight record: {}.'.format(exc)
        )
    return destination


def run_auto_mix_calibration_pi_preflight(
        plan, transport, packet_registry, reader_scan_protocol,
        record_path=None):
    '''Run the exact no-liquid calibration preflight sequence.

    Ordering is intentional and must remain:

    ``calibration preflight -> well binding -> resource preflight``.

    The calibration preflight requires the logical well to be absent; only its
    successful acknowledgement permits the binding that creates it. The
    resource request then proves source/tip availability for that exact bound
    well. No transfer, mixing, homing, reader macro, or scan exists here.
    '''
    plan = validate_auto_mix_calibration_run_plan(plan)
    reader_scan_protocol = _require_reader_scan_protocol(reader_scan_protocol)
    _require_transport_methods(transport)

    snapshot = transport.get_robot_state_snapshot()
    try:
        # The paired capability validator includes every required preflight,
        # source-transfer, and mix command. A partially deployed Pi must fail
        # here, before the binding changes controller-visible state.
        validate_auto_mix_calibration_resource_capability(
            snapshot, packet_registry
        )
    except (ValueError, TypeError) as exc:
        raise AutoMixCalibrationPreflightError(
            'Calibration Pi capability validation failed: {}.'.format(exc)
        )

    try:
        pi_preflight = validate_mix_calibration_preflight_acknowledgement(
            transport.request_pi_preflight(copy.deepcopy(
                plan['preflight_request']
            )),
            plan['preflight_request']
        )
    except (ValueError, TypeError) as exc:
        raise AutoMixCalibrationPreflightError(
            'Calibration Pi v1 preflight acknowledgement is invalid: {}.'
            .format(exc)
        )
    _require_accepted(pi_preflight, 'v1 preflight')

    binding_request = build_auto_mix_calibration_well_binding_request(plan)
    try:
        well_binding = validate_auto_mix_calibration_well_binding_acknowledgement(
            transport.request_pi_well_binding(copy.deepcopy(binding_request)),
            binding_request
        )
    except (ValueError, TypeError) as exc:
        raise AutoMixCalibrationPreflightError(
            'Calibration Pi well-binding acknowledgement is invalid: {}.'
            .format(exc)
        )
    _require_accepted(well_binding, 'well binding')

    resource_request = build_auto_mix_calibration_resource_preflight_request(
        plan, binding_request['binding_id']
    )
    try:
        resource_preflight = (
            validate_auto_mix_calibration_resource_preflight_acknowledgement(
                transport.request_pi_resource_preflight(
                    copy.deepcopy(resource_request)
                ),
                resource_request
            )
        )
    except (ValueError, TypeError) as exc:
        raise AutoMixCalibrationPreflightError(
            'Calibration Pi resource-preflight acknowledgement is invalid: '
            '{}.'.format(exc)
        )
    _require_accepted(resource_preflight, 'resource preflight')

    record = build_auto_mix_calibration_preflight_record(
        plan, reader_scan_protocol, snapshot, pi_preflight, well_binding,
        resource_preflight
    )
    if record_path is not None:
        record['record_path'] = write_auto_mix_calibration_preflight_record(
            record, record_path
        )
    return record
