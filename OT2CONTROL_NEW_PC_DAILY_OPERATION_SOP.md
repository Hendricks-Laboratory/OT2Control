# OT2Control New-PC Daily Startup and Run SOP

## Purpose

This SOP starts the OT-2, temperature module, SPECTROstar Nano, Google Drive paths, robot-side Eve listener, and PC-side OT2Control controller on the replacement lab PC.

This workflow reproduces the legacy working environment. It does not update robot software, firmware, or Python packages.

## Validated baseline

```text
Opentrons App: 6.0.1
Robot server: 6.0.1
Robot serial: OT2CEP20191017B11
Robot firmware: 1.1.0-25e5cea
Supported Protocol API: 2.0 through 2.12
Current wired IP: 169.254.44.249

PC Python environment: ot2control_legacy
Python: 3.9.12
PC Opentrons package: 5.0.1

SPECTROstar software: 5.50
SPECTROstar firmware: 1.20
SPECTROstar serial: 601-0091

Robot-side branch: main
New-PC branch: main-new-pc-opentrons-migration
Eve listener port: 50000
```

## Safety rules

- Do not update the robot, Opentrons App, reader firmware, or robot-side Python packages.
- Do not run `pip install`, `apt upgrade`, or firmware updates on the robot.
- Do not kill the `tmux` session while a protocol is running.
- Do not use `--no-sim` unless a simulation has passed and a live run is intentionally authorized.
- Do not run `changeIP.py` during routine startup when the App shows the expected IP.
- Never put passwords, SSH keys, service-account credentials, or access tokens in GitHub.
- Auto mode is not part of the initial `main` migration validation.

## 1. Turn on the SPECTROstar Nano

1. Press and hold the reader ON/OFF button for approximately three seconds.
2. Confirm the red light flashes.
3. Press IN/OUT to eject the plate holder if required.
4. Confirm the USB connection or USB switch is directed to the new control PC.
5. Open SPECTROstar Nano V5.50.
6. Log in as `USER` with the password field blank.
7. Confirm the software reports `Ready` and recognizes the correct reader.
8. Close the SPECTROstar application before starting `controller.py`. The controller opens and
   configures the application when a live run begins. A simulation uses `DummyReader` and does
   not need the application open.

Do not perform a firmware update. The validated reader firmware is 1.20.

## 2. Turn on the temperature module

Complete this before starting or restarting Eve.

1. Open the Opentrons App.
2. Press the button on the side of the temperature module.
3. Confirm the module appears on USB Port 2.
4. Open the three-dot menu for USB Port 2.
5. Select **Set module temperature**.
6. Enter the temperature required by the protocol. The normal lab setting is 4 °C.
7. Confirm the module is connected and responding.

If the module is turned off and back on while Eve is running, restart the robot-side `run` session before running a protocol. Otherwise, Eve may repeatedly report:

```text
NoResponse: /dev/ot_module_tempdeck0
```

## 3. Confirm Google Drive and output folders

Open File Explorer and confirm Google Drive appears as `G:`.

Confirm this shared-drive folder opens:

```text
G:\Shared drives\Hendricks Lab Drive\Opentrons_Reactions\Plate Reader Data
```

Confirm these local folders exist:

```text
C:\Users\science_356_lab\Robot_Files\Plate Reader Data
C:\Users\science_356_lab\Robot_Files\Protocol_Outputs
```

The data flow is:

```text
SPECTROstar primary export:
C:\Users\science_356_lab\Robot_Files\Plate Reader Data

SPECTROstar backup export:
G:\Shared drives\Hendricks Lab Drive\Opentrons_Reactions\Plate Reader Data

Controller experiment output:
C:\Users\science_356_lab\Robot_Files\Protocol_Outputs
```

The controller does not require `Protocol_Outputs` to be mirrored by Google Drive in order to run. Do not change Drive mirroring or backup settings immediately before a protocol.

If needed, verify the two SPECTROstar destinations in PowerShell:

```powershell
Test-Path "C:\Users\science_356_lab\Robot_Files\Plate Reader Data"
Test-Path "G:\Shared drives\Hendricks Lab Drive\Opentrons_Reactions\Plate Reader Data"
```

Both should return `True`.

## 4. Confirm the robot in the Opentrons App

1. Open the Opentrons App.
2. Confirm the robot appears and is connected.
3. Open **Robot Settings → Networking**.
4. Record the current wired IP.

The validated wired IP is:

```text
169.254.44.249
```

Do not accept any robot software or firmware update prompt.

## 5. Connect to the robot through Ubuntu

Open Ubuntu on the new PC.

Connect using:

```bash
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -i ~/Desktop/ssh_key root@169.254.44.249
```

If the wired IP has changed, replace `169.254.44.249` with the current IP displayed in the Opentrons App.

Enter the SSH-key passphrase privately when prompted.

The following warning is expected because the robot uses older SSH software:

```text
WARNING: connection is not using a post-quantum key exchange algorithm.
```

A successful connection ends at a robot prompt resembling `#`.

The new PC does not need local `sudo` for this command. `root@...` specifies the robot-side account.

## 6. Safely restart the Eve listener

Check for an existing session:

```bash
tmux ls
```

If a `run` session exists, inspect it before killing it:

```bash
tmux capture-pane -pt run | tail -n 30
```

If the output shows an active protocol or robot operations, do not kill it.

If the robot is idle, stop the old listener:

```bash
tmux kill-session -t run
tmux ls
```

A message saying no tmux server is running is acceptable.

Enter the robot-side folder:

```bash
cd /root/OT2Control
```

Confirm the robot-side branch:

```bash
git branch
```

The asterisk must be beside `main`. If another branch is selected, stop and investigate before changing it.

Start a fresh session:

```bash
tmux new -s run
```

Inside the session, start Eve:

```bash
python ot2_robot.py
```

A legacy dependency warning involving `requests`, `urllib3`, or `chardet` may appear. It is currently nonfatal. Do not update robot packages to remove it.

Wait for:

```text
<<eve>> listening on port 50000
```

The correct port is 50000, not 5000. Confirm that temperature-module `NoResponse` messages do not begin repeating.

Detach while leaving Eve running:

1. Press `Ctrl+B`.
2. Release both keys.
3. Press `D`.

Confirm the session and port:

```bash
tmux ls
netstat -an | grep '50000'
```

Expected network output includes `169.254.44.249:50000` and `LISTEN`.

Disconnect from the robot:

```bash
exit
```

Leave this first Ubuntu tab open at the local PC prompt. Its role is robot/Eve administration and troubleshooting.

Do not run `controller.py` from a robot `#` prompt.

## 7. Open a second Ubuntu tab and prepare the PC controller

Open a new Ubuntu terminal tab on the Windows PC. This preserves the original lab workflow and keeps the two roles visually separate:

```text
Tab 1: robot SSH, tmux, and Eve administration
Tab 2: local PC controller and simulations
```

All commands in the remainder of this SOP are run in Tab 2 unless a step explicitly says to reconnect to the robot.

Activate the validated environment:

```bash
conda activate ot2control_legacy
```

Enter the PC-side folder:

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

Confirm the PC-side branch:

```bash
git branch
```

The new PC should use `main-new-pc-opentrons-migration`.

When needed, confirm the environment:

```bash
python --version
python -c "import opentrons; print(opentrons.__version__)"
```

Expected:

```text
Python 3.9.12
5.0.1
```

A `pkg_resources` deprecation warning may appear. It is currently nonfatal. Do not change packages immediately before validation.

## 8. First controlled new-PC simulation

Run:

```bash
python controller.py -n DEBUG_PC
```

This test:

- Runs the workflow in local simulation.
- Redirects the simulated connection to `127.0.0.1`.
- Does not connect to or move the physical robot.
- Automatically uses a dummy plate reader because the controller is simulating.
- Does not command a SPECTROstar scan.
- Must not print any `<<Reader>> executing:` DDE commands.

At the end, the controller asks:

```text
would you like to run the protocol? [yn]
```

For the controlled test, enter `n`. Record the output and any errors. Do not add `--no-sim` to this test.

## 9. Normal protocol simulation

After `DEBUG_PC` passes, simulate an actual reaction:

```bash
python controller.py -n <REACTION_NAME>
```

Example:

```bash
python controller.py -n CNH_077
```

When asked whether to run the physical protocol, enter `n` unless a live run has been explicitly approved and all physical checks have passed.

All controller simulations use `DummyReader`. The SPECTROstar tray must not move, shake, scan,
or load a measurement protocol during the precheck. If any physical reader action occurs during
simulation, stop and verify that the migration branch includes the plate-reader simulation fix.

## 10. Live protocol execution

Only proceed after the corresponding simulation passes.

After simulation, the controller asks:

```text
would you like to run the protocol? [yn]
```

Enter `y` only when the robot is physically ready.

The existing shortcut that skips initial simulation is:

```bash
python controller.py -n <REACTION_NAME> --no-sim
```

Use it only when:

- The same protocol has passed simulation.
- The deck and labware have been checked.
- Pipettes, tips, reagents, and volumes are confirmed.
- The temperature module is connected.
- SPECTROstar reports `Ready`.
- Both plate-reader output paths are available.
- The required named SPECTROstar measurement protocol exists under `User\Definit`.
- No other person is using the robot.

After confirming that SPECTROstar reports `Ready`, close the SPECTROstar application before
starting the controller. The live-run path writes the required settings and initializes the
application through `DDEClient.exe`.

## 11. Auto mode

Auto mode is not part of the initial `main` migration validation.

Existing command:

```bash
python controller.py -m auto -n <REACTION_NAME> --no-sim
```

Do not use it until the normal `main` workflow has been fully validated and Auto-RTG migration work has been separately approved.

## Troubleshooting

### SSH reports `Permission denied (publickey)`

Use the complete compatibility command:

```bash
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -i ~/Desktop/ssh_key root@<CURRENT_ROBOT_IP>
```

The copied key was validated as a 2048-bit RSA key.

### First SSH connection asks whether the host is authentic

Confirm the displayed IP matches the Opentrons App before entering `yes`.

The validated robot ECDSA fingerprint during migration was:

```text
SHA256:kb703/k/uGrpLUWOOIOmgG+jHdeo6+fcihUG+rRyEg4
```

If the fingerprint unexpectedly changes, stop and investigate.

### Temperature module produces repeated `NoResponse` errors

1. Confirm the module is powered on.
2. Confirm the Opentrons App recognizes it.
3. Stop the idle `run` session.
4. Start a new `run` session.
5. Restart `python ot2_robot.py`.

Do not update module firmware.

### Eve appears inactive

Check the process and correct port:

```bash
ps -ef | grep '[o]t2_robot.py'
netstat -an | grep '50000'
```

### Google Drive backup fails

Confirm this path opens:

```text
G:\Shared drives\Hendricks Lab Drive\Opentrons_Reactions\Plate Reader Data
```

The BMG export configuration uses that exact path as `BackupDir`.

### SPECTROstar reports no connection

1. Confirm its USB connection is directed to the new PC.
2. Confirm the official BMG USB drivers 2.12.28 are installed.
3. Restart Windows if drivers were just installed.
4. Reopen SPECTROstar V5.50.
5. Confirm it reports `Ready`.
6. Do not update firmware.

### SPECTROstar reports that `NC_synthesis` does not exist

`NC_synthesis` is a named SPECTROstar measurement protocol, not a setting in
`SPECTROstar Nano.ini`. The fresh V5.50 installation does not include the lab's custom protocol
database. Follow `SPECTROSTAR_CUSTOM_PROTOCOL_MIGRATION.md` to restore the complete working
`User\Definit` directory from the old-PC program backup before attempting a live run.

This error must not occur during simulation because simulations use `DummyReader` and send no
DDE commands to SPECTROstar.

### The robot IP changes

Use the current IP displayed in the Opentrons App for SSH.

`changeIP.py` is a recovery utility, not a routine startup command. Do not run it automatically. Its behavior must be inspected and documented separately before it is added to the finalized IP-recovery procedure.

## End-of-run notes

- Preserve all error output.
- Do not delete failed-run files before documenting them.
- Do not update software in response to warnings without a migration review.
- Keep the old lab PC available as fallback.
- Record successful simulation and live-run milestones in the migration log.
