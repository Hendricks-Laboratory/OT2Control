# SPECTROstar Nano Custom Protocol Migration

## Purpose

Restore the lab's custom SPECTROstar Nano V5.50 measurement protocols, including
`NC_synthesis`, from the preserved old-PC program folder to the replacement PC.

`NC_synthesis` is not an INI setting. OT2Control passes it to BMG's `DDEClient.exe` as a named
measurement protocol located in:

```text
C:\Program Files\SPECTROstar Nano V5.50\User\Definit
```

On the replacement PC, that legacy code-compatible path is a junction to the official install:

```text
C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Definit
```

The files in `Definit` form a related protocol database. Do not copy an individual `.DB`, `.PX`,
or similarly named file by itself. Preserve and restore the complete directory as one set.

## Preconditions

- The preserved old-PC `SPECTROstar Nano V5.50` program-folder archive is available.
- The official SPECTROstar Nano V5.50 software is installed on the replacement PC.
- The legacy junction path has already been created and verified.
- SPECTROstar is not running on either PC while its protocol files are copied.
- PowerShell is opened as Administrator for changes under `Program Files (x86)`.

Do not update reader firmware during this procedure.

## 1. Inspect the old working protocol database

Run on the **old lab PC in normal PowerShell**. This is read-only:

```powershell
Get-ChildItem "C:\Program Files\SPECTROstar Nano V5.50\User\Definit" -File |
Sort-Object Name |
Select-Object Name,Length,LastWriteTime
```

The protocol name may be stored inside BMG database files rather than in a file literally named
`NC_synthesis`. Therefore, the absence of a file named `NC_synthesis` does not prove that the
working database lacks the protocol.

## 2. Extract the preserved old-PC program archive on the replacement PC

Extract it into a neutral staging folder, not into `Program Files`. Example:

```text
C:\Users\science_356_lab\Desktop\SPECTROstar_old_PC_staging\SPECTROstar Nano V5.50
```

Confirm that the staged database exists. Run on the **new PC in PowerShell**:

```powershell
Test-Path "$env:USERPROFILE\Desktop\SPECTROstar_old_PC_staging\SPECTROstar Nano V5.50\User\Definit"
```

Expected result:

```text
True
```

If the archive was extracted somewhere else, substitute its actual path in all following
commands.

## 3. Close SPECTROstar

Close the SPECTROstar Nano application normally. Confirm it is not running. Run on the
**new PC in PowerShell**:

```powershell
Get-Process | Where-Object { $_.ProcessName -match "SPECTROstar|DDEClient" }
```

A blank result means no matching process was found. If a process remains, stop and close the
application normally before continuing.

## 4. Back up the fresh new-PC user database

Open **PowerShell as Administrator** on the new PC:

```powershell
$newRoot = "C:\Program Files (x86)\BMG\SPECTROstar Nano"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$freshUserBackup = "$env:USERPROFILE\Desktop\SPECTROstar_User_fresh_install_$stamp"
Copy-Item "$newRoot\User" $freshUserBackup -Recurse -Force
```

Confirm that the backup exists and contains files:

```powershell
Get-ChildItem $freshUserBackup -Recurse -File |
Measure-Object
```

Do not continue unless `Count` is greater than zero.

## 5. Stage the old protocol database beside the fresh one

Still in **PowerShell as Administrator**, set the old folder path:

```powershell
$oldDefinit = "$env:USERPROFILE\Desktop\SPECTROstar_old_PC_staging\SPECTROstar Nano V5.50\User\Definit"
```

Verify both folders:

```powershell
Test-Path $oldDefinit
Test-Path "$newRoot\User\Definit"
```

Both must return `True`.

Move the fresh directory out of the installed application so it remains immediately recoverable
without leaving a second database beside the active one:

```powershell
$freshDefinitBackup = "$env:USERPROFILE\Desktop\Definit_fresh_install_$stamp"
Move-Item "$newRoot\User\Definit" $freshDefinitBackup
```

Copy the complete old working directory into place:

```powershell
Copy-Item $oldDefinit "$newRoot\User\Definit" -Recurse -Force
```

Confirm the restored directory exists and contains files:

```powershell
Get-ChildItem "$newRoot\User\Definit" -File |
Sort-Object Name |
Select-Object Name,Length,LastWriteTime
```

## 6. Verify through the legacy junction

Run on the **new PC in PowerShell**:

```powershell
Test-Path "C:\Program Files\SPECTROstar Nano V5.50\User\Definit"
```

Expected:

```text
True
```

The controller's existing `PROTOCOL_PATH` should now resolve to the restored database through
the junction.

## 7. Verify `NC_synthesis` in the application

1. Open SPECTROstar Nano V5.50.
2. Log in as `USER` using the established lab login.
3. Inspect the available measurement protocols/definitions.
4. Confirm that `NC_synthesis` appears.
5. Do not update firmware.
6. Close the application before testing OT2Control.

If `NC_synthesis` does not appear, do not run the physical protocol. Preserve screenshots and
contact BMG support to request the V5.50 procedure for restoring a multi-user `User\Definit`
database. The public BMG product page directs users to software support or the manual-request
form for version-specific software instructions.

To restore the fresh database without deleting either copy, close SPECTROstar and run in
**PowerShell as Administrator**:

```powershell
$newRoot = "C:\Program Files (x86)\BMG\SPECTROstar Nano"
$freshDefinitBackup = Get-ChildItem "$env:USERPROFILE\Desktop" -Directory -Filter "Definit_fresh_install_*" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1 -ExpandProperty FullName
$failedOldDatabase = "$env:USERPROFILE\Desktop\Definit_old_PC_restore_failed_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Move-Item "$newRoot\User\Definit" $failedOldDatabase
Move-Item $freshDefinitBackup "$newRoot\User\Definit"
```

## 8. Validate simulation before live use

After installing the plate-reader simulation fix, run on the **new PC in Ubuntu**:

```bash
conda activate ot2control_legacy
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
python controller.py -n DEBUG_PC
```

During simulation:

- No `<<Reader>> executing:` lines should appear.
- The SPECTROstar tray must not move.
- No physical shake or scan should occur.
- Answer `n` when asked whether to run the physical protocol.

Only after this simulation passes and `NC_synthesis` is visible should a separately authorized
live dry run be attempted.
