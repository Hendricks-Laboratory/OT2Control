'''Hardware-free tests for the Stage 11A remote publication boundary.'''

import ast
import hashlib
import os
import tempfile
import unittest

from auto_live_run_remote import (
    AutoLiveRunRemoteBoundaryError,
    REMOTE_CREATE_OPERATION,
    REMOTE_SNAPSHOT_RECEIPT_TYPE,
    REMOTE_SNAPSHOT_SCHEMA_VERSION,
    build_immutable_snapshot_request,
    publish_immutable_snapshot
)


class _RecordingCreateOnlyPublisher:
    '''Local fake publisher; it never contacts a remote service.'''

    def __init__(self, receipt_mutation=None):
        self.receipt_mutation = receipt_mutation
        self.requests = []

    def create_immutable_snapshot(self, request, workbook_path):
        self.requests.append((request, workbook_path))
        receipt = {
            'schema_version': REMOTE_SNAPSHOT_SCHEMA_VERSION,
            'record_type': REMOTE_SNAPSHOT_RECEIPT_TYPE,
            'operation': request['operation'],
            'run_id': request['run_id'],
            'state_revision': request['state_revision'],
            'content_sha256': request['content_sha256'],
            'remote_relative_path': request['remote_relative_path'],
            'created': True,
            'remote_object_id': 'fake-remote-object-001'
        }
        if self.receipt_mutation is not None:
            self.receipt_mutation(receipt)
        return receipt


class AutoLiveRunRemoteBoundaryTests(unittest.TestCase):
    '''Verifies the offline contract without network or credential access.'''

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workbook_path = os.path.join(
            self.temporary_directory.name,
            'immutable_snapshot.xlsx'
        )
        with open(self.workbook_path, 'wb') as output_file:
            output_file.write(b'synthetic immutable workbook bytes')
        self.digest = hashlib.sha256(
            b'synthetic immutable workbook bytes'
        ).hexdigest()
        self.queue_item = {
            'run_id': 'RTG 018 / night run',
            'state_revision': 12,
            'local_workbook_sha256': self.digest
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_request_is_deterministic_create_only_and_non_overwriting(self):
        request = build_immutable_snapshot_request(self.queue_item)

        self.assertEqual(request['operation'], REMOTE_CREATE_OPERATION)
        self.assertIs(request['overwrite'], False)
        self.assertEqual(request['run_id'], self.queue_item['run_id'])
        self.assertEqual(request['state_revision'], 12)
        self.assertEqual(request['content_sha256'], self.digest)
        self.assertEqual(
            request['remote_relative_path'],
            'Auto_Live_Runs/rtg-018-night-run-af3d67e87dcb/'
            'status/revision_000012_{}.xlsx'.format(self.digest[:12])
        )
        self.assertEqual(request, build_immutable_snapshot_request(
            self.queue_item
        ))

    def test_request_rejects_invalid_revision_and_digest(self):
        bad_revision = dict(self.queue_item, state_revision=True)
        with self.assertRaises(AutoLiveRunRemoteBoundaryError):
            build_immutable_snapshot_request(bad_revision)

        bad_digest = dict(self.queue_item, local_workbook_sha256='A' * 64)
        with self.assertRaises(AutoLiveRunRemoteBoundaryError):
            build_immutable_snapshot_request(bad_digest)

    def test_remote_path_never_uses_untrusted_run_identifier_characters(self):
        request = build_immutable_snapshot_request(dict(
            self.queue_item,
            run_id='../RTG: snowman-☃'
        ))

        self.assertTrue(request['remote_relative_path'].startswith(
            'Auto_Live_Runs/rtg-snowman-'
        ))
        self.assertNotIn('..', request['remote_relative_path'])
        self.assertNotIn('☃', request['remote_relative_path'])

    def test_fake_publisher_receives_verified_immutable_snapshot(self):
        publisher = _RecordingCreateOnlyPublisher()

        receipt = publish_immutable_snapshot(
            publisher,
            self.queue_item,
            self.workbook_path
        )

        self.assertEqual(len(publisher.requests), 1)
        request, received_path = publisher.requests[0]
        self.assertEqual(received_path, os.path.abspath(self.workbook_path))
        self.assertEqual(receipt['remote_relative_path'], request[
            'remote_relative_path'
        ])
        self.assertTrue(receipt['created'])

    def test_publish_rejects_changed_local_snapshot_before_publisher_call(self):
        with open(self.workbook_path, 'wb') as output_file:
            output_file.write(b'changed after local queueing')
        publisher = _RecordingCreateOnlyPublisher()

        with self.assertRaises(AutoLiveRunRemoteBoundaryError):
            publish_immutable_snapshot(
                publisher,
                self.queue_item,
                self.workbook_path
            )

        self.assertEqual(publisher.requests, [])

    def test_publish_rejects_receipt_that_changes_immutable_destination(self):
        def alter_path(receipt):
            receipt['remote_relative_path'] = 'somewhere/else.xlsx'

        with self.assertRaises(AutoLiveRunRemoteBoundaryError):
            publish_immutable_snapshot(
                _RecordingCreateOnlyPublisher(alter_path),
                self.queue_item,
                self.workbook_path
            )

    def test_source_remains_network_and_credentials_free(self):
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'auto_live_run_remote.py'
        )
        with open(source_path, 'r', encoding='utf-8') as input_file:
            source = input_file.read()
        parsed = ast.parse(source, filename=source_path)
        imported_roots = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split('.')[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split('.')[0])
        self.assertEqual(imported_roots, {'copy', 'hashlib', 'os'})


if __name__ == '__main__':
    unittest.main()
