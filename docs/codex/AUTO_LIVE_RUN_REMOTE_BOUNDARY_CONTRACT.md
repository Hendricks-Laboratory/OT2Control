# Auto Live Run Remote Boundary Contract

## Status

Stage 11A is implemented and deliberately offline. It introduces no cloud
SDK, credentials, network call, controller integration, Pi change, or remote
file operation.

## Purpose

When a reviewed remote publisher is added later, it must publish a local Live
workbook revision as one **new immutable snapshot**. The original input
workbook, existing remote documents, and unrelated Drive files are outside
this contract.

The local `AutoLiveRunSyncQueue` remains responsible for FIFO ordering and
immutable per-revision local workbook copies. A remote publisher may be
connected only through `auto_live_run_remote.py` after separate approval.

## Allowed operation

The only valid publication operation is:

```text
create_immutable_snapshot
```

For a queue item, the destination is deterministically derived as:

```text
Auto_Live_Runs/<safe-run-id-and-hash>/status/
revision_<state-revision>_<workbook-sha-prefix>.xlsx
```

The request includes `overwrite: false`; callers cannot provide an arbitrary
remote path. The contract rejects a receipt that changes the operation, run,
revision, checksum, or derived path.

## Publisher interface for a later stage

An injected publisher must expose:

```python
create_immutable_snapshot(request, workbook_path)
```

It must return a receipt confirming that it created exactly the requested
object. Before the publisher is called, the local workbook is read and its
SHA-256 digest is checked against the queued immutable snapshot record.

## Explicit non-goals

Stage 11A does not:

- download, modify, delete, move, or rename a remote file;
- access Google Drive, a network, credentials, or OAuth;
- read an operator action request;
- alter the original input workbook;
- alter the controller, Auto-main/Pi behavior, source allocation, or recovery
  logic.

Stage 11B may add a configured publisher that implements this narrow contract.
Stage 11C may add read-only remote action-request intake with the existing
terminal workflow remaining the offline fallback.
