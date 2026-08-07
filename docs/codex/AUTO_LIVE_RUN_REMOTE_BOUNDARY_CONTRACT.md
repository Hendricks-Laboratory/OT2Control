# Auto Live Run Remote Boundary Contract

## Status

This is a retired, deliberately offline Stage 11A boundary. It introduces no
cloud SDK, credentials, network call, controller integration, Pi change, or
remote file operation. It is not the planned Google Drive publication path.

## Purpose

This file records the earlier, test-only immutable-publication concept. The
original input workbook, existing Drive documents, and unrelated Drive files
remain outside this contract. No reviewed remote publisher is planned under
the current Stage 11 design.

The local `AutoLiveRunSyncQueue` remains responsible for FIFO ordering and
immutable per-revision local workbook copies. The normal Drive-visible output
is the existing Google Drive desktop synchronization of the Lab-PC
`Protocol_Outputs` tree.

## Retired operation

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

## Historical publisher interface

An earlier proposed publisher would have exposed:

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

## Current Stage 11 direction

The former Google Drive API publisher prototype was deliberately removed before
it was committed or connected to the controller. The supported publication path
is the existing Lab-PC Google Drive desktop synchronization of the normal
`Protocol_Outputs` tree. The controller already writes the derived status
workbook beneath that tree at:

```text
<Protocol_Outputs>/<effective-data-dir>/Live_Run/<run>_LIVE.xlsx
```

Stage 11 must not create a second Drive namespace, configure credentials, or
make direct network requests. A later operator-action stage will use a distinct
operator-written workbook in the same `Live_Run` directory; it will never
modify the controller-written status workbook or the original input workbook.
