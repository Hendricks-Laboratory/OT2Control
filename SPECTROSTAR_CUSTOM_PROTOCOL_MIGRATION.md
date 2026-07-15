# SPECTROstar Nano Custom Protocol Migration

## What must be transferred

`NC_synthesis` is not a setting in `SPECTROstar Nano.ini`. It is a named BMG measurement
protocol stored in the related database files inside the complete `User\Definit` folder.

Old working lab-PC folder:

```text
C:\Program Files\SPECTROstar Nano V5.50\User\Definit
```

New-PC real installation folder:

```text
C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Definit
```

The new PC also has this legacy junction path used by OT2Control:

```text
C:\Program Files\SPECTROstar Nano V5.50\User\Definit
```

That junction points to the real new-PC installation. Perform the manual replacement through
the real `Program Files (x86)` location.

Copy the complete `Definit` folder. Do not copy only one `.DB`, `.PX`, or similarly named file;
the files form one related protocol database.

## Option A — Use the preserved old-PC program ZIP

If the migration archive already contains the complete old working folder
`SPECTROstar Nano V5.50`, the old lab PC does not need to be touched again.

1. Download the preserved old-PC SPECTROstar program ZIP onto the new PC.
2. Extract it to a temporary folder on the new PC Desktop.
3. Open the extracted folders until this folder is visible:

   ```text
   SPECTROstar Nano V5.50\User\Definit
   ```

4. Confirm that `Definit` contains the old database files. The protocol name may be stored inside
   those database files, so there may not be a file literally named `NC_synthesis`.
5. Continue to **Install the old protocol database on the new PC** below.

## Option B — Copy the folder manually from the old lab PC

Use this only if the preserved program ZIP does not contain `User\Definit`.

1. On the old lab PC, close SPECTROstar Nano completely.
2. Open Windows File Explorer.
3. Paste this into the address bar:

   ```text
   C:\Program Files\SPECTROstar Nano V5.50\User
   ```

4. Copy the entire folder named:

   ```text
   Definit
   ```

5. Paste it into the private migration folder used to transfer files to the new PC.
6. Zip that copied folder so Google Drive preserves the complete database as one set.
7. Name it clearly, for example:

   ```text
   SPECTROstar_Nano_V5.50_old_lab_PC_Definit.zip
   ```

8. Upload that ZIP to the private migration Google Drive folder.
9. On the new PC, download the ZIP locally and extract it to the Desktop.

Do not commit this folder or ZIP to GitHub.

## Install the old protocol database on the new PC

1. Close SPECTROstar Nano completely on the new PC.
2. Open Task Manager and confirm that neither SPECTROstar Nano nor `DDEClient.exe` is running.
3. Open Windows File Explorer.
4. Paste this real installation path into the address bar:

   ```text
   C:\Program Files (x86)\BMG\SPECTROstar Nano\User
   ```

5. Find the fresh new-PC folder named `Definit`.
6. Move that fresh folder to the Desktop. Rename it:

   ```text
   Definit_NEW_PC_FRESH_INSTALL_BACKUP
   ```

   Do not delete it. This is the rollback copy.

7. Copy the old working `Definit` folder extracted from the old-PC archive.
8. Paste the old folder into:

   ```text
   C:\Program Files (x86)\BMG\SPECTROstar Nano\User
   ```

9. Windows may ask for administrator permission to copy into Program Files. Click **Continue**
   using the approved lab administrator account.
10. Confirm the final path is exactly:

    ```text
    C:\Program Files (x86)\BMG\SPECTROstar Nano\User\Definit
    ```

11. Open `Definit` and confirm the database files are directly inside it.

Correct structure:

```text
...\User\Definit\1.DB
...\User\Definit\1.PX
...\User\Definit\[other database files]
```

Incorrect extra nesting:

```text
...\User\Definit\Definit\1.DB
```

## Verify the imported protocol

1. Open SPECTROstar Nano V5.50.
2. Log in as `USER` using the established lab login.
3. Open the measurement-protocol/definition selection in the application.
4. Confirm that this named protocol appears:

   ```text
   NC_synthesis
   ```

5. Do not update reader firmware.
6. Close SPECTROstar Nano before running `controller.py`.

If `NC_synthesis` still does not appear, do not attempt a physical OT2Control run. Restore the
fresh database using the rollback procedure and contact BMG support for the V5.50 multi-user
protocol-database import procedure.

## Manual rollback

1. Close SPECTROstar Nano completely.
2. Move the imported old-PC folder out of:

   ```text
   C:\Program Files (x86)\BMG\SPECTROstar Nano\User
   ```

3. Preserve it on the Desktop with a name such as:

   ```text
   Definit_OLD_PC_IMPORT_FAILED
   ```

4. Move `Definit_NEW_PC_FRESH_INSTALL_BACKUP` from the Desktop back into:

   ```text
   C:\Program Files (x86)\BMG\SPECTROstar Nano\User
   ```

5. Rename it back to:

   ```text
   Definit
   ```

## Validate the simulation fix before a live run

After the new PC pulls the plate-reader simulation fix, run in **New PC — Ubuntu**:

```bash
conda activate ot2control_legacy
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
python controller.py -n DEBUG_PC
```

During simulation:

- No `<<Reader>> executing:` lines should appear.
- The SPECTROstar tray must not move.
- The reader must not shake or scan.
- The simulation still generates dummy scan data for downstream save and plot checks.
- Enter `n` when asked whether to run the physical protocol.

Only after that simulation passes and `NC_synthesis` appears in SPECTROstar should a separately
authorized live dry run be attempted.
