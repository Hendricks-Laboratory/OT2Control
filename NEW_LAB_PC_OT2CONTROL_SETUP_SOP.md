# New Lab PC OT2Control Setup SOP

**File:** `NEW_LAB_PC_OT2CONTROL_SETUP_SOP.md`  
**Purpose:** Novice-friendly, step-by-step setup guide for preparing a new Windows lab PC to run the Hendricks Laboratory `OT2Control` / Opentrons workflow.  
**Document type:** Living SOP. Update at every major validated milestone.  
**Current checkpoint covered:** Start of migration through Ubuntu/WSL setup, Git/GitHub setup, robot SSH key preservation, Miniconda installation, Git migration branch setup, and the Conda Terms of Service blocker before creating the legacy Python environment.

---

## 1. Scope and goal

This SOP explains how to set up a new Windows lab PC so it can reproduce the old working lab PC workflow for `OT2Control` and the Opentrons OT-2 robot.

The first goal is **legacy reproduction**, not modernization.

That means:

1. Preserve the old lab PC as a fallback.
2. Preserve the robot-side configuration.
3. Set up the new PC so it uses the same folder structure and a similar Ubuntu/Conda workflow.
4. Work on a Git migration branch instead of directly on `main`.
5. Record every step clearly enough for a non-technical user.

Later, after the old workflow is reproduced, a separate environment or branch may be used to test newer Opentrons packages.

---

## 2. Important vocabulary for a novice

### 2.1 Windows path

This is the normal file path in Windows File Explorer:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

### 2.2 Ubuntu / WSL path

WSL means Windows Subsystem for Linux. It lets Windows run an Ubuntu terminal.

The same Windows folder appears inside Ubuntu at:

```text
/mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

These two paths refer to the same folder:

```text
Windows:    C:\Users\science_356_lab\Robot_Files\OT2Control
Ubuntu:     /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

### 2.3 GitHub repository

The GitHub repository is:

```text
https://github.com/Hendricks-Laboratory/OT2Control.git
```

The new PC currently uses HTTPS for GitHub.

The old lab PC appears to use SSH for GitHub and may ask for an SSH identity when fetching or pulling.

Both methods can point to the same GitHub repository.

### 2.4 Robot SSH key

The robot SSH key is separate from GitHub.

The robot SSH key is used to connect to the OT-2 robot, usually with a command like:

```bash
sudo ssh -i ssh_key root@<robot IP>
```

On the new PC Ubuntu environment, the robot SSH key was copied to:

```text
~/Desktop/ssh_key
```

Do not commit this key to GitHub.

### 2.5 Two different Opentrons version layers

There are two separate Opentrons version layers.

**Layer 1: Opentrons App / robot software**

The lab PC / robot-facing software was reported as:

```text
6.0.1
```

**Layer 2: Python package inside Ubuntu / Conda**

The old lab PC Python environment used:

```text
opentrons==5.0.1
opentrons-shared-data==5.0.1
```

These can both be true because they are different software layers.

For the first migration stage, reproduce the old Python package layer:

```text
Python 3.9.12
opentrons==5.0.1
opentrons-shared-data==5.0.1
```

---

## 3. Safety rules

### 3.1 Do not touch the robot side during initial setup

The robot-side files are treated as already working. Do not edit, delete, overwrite, upgrade, or move files on the robot during the first setup stage.

### 3.2 Do not modify the old lab PC

The old lab PC is the fallback. Do not upgrade it or change it during this setup.

### 3.3 Do not run live robot commands yet

Do not run live/no-simulation commands until all earlier checks are complete.

Live robot commands include commands with:

```text
--no-sim
```

### 3.4 Do not commit secrets to GitHub

Never commit these files or folders:

```text
ssh_key
PersonalAccesstoken.txt
client_secret*.json
Credentials/
Google service account JSON files
Cache/
Controller_Out/
Eve_Out/
Armchair_Logs/
__pycache__/
.venv/
large generated output folders
historical plate reader data
```

---

## 4. Old working lab PC baseline

Known old lab PC Windows path:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

Known old lab PC Ubuntu path:

```text
/mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

Old lab PC Ubuntu:

```text
Ubuntu 20.04.6 LTS
```

Old lab PC Python:

```text
Python 3.9.12
```

Old lab PC Python paths:

```text
/home/science_356_lab/anaconda3/bin/python
/home/science_356_lab/anaconda3/bin/python3
```

Old lab PC Python packages included:

```text
opentrons==5.0.1
opentrons-shared-data==5.0.1
numpy==1.22.2
pandas==1.2.5
scipy==1.7.0
scikit-learn==0.24.2
matplotlib==3.4.2
gspread==5.4.0
google-api-python-client==1.6.7
google-auth==1.32.0
oauth2client==4.1.3
customtkinter==5.2.2
```

The old lab PC also had a `.venv` folder in `OT2Control`, but that was not the active Python environment.

---

## 5. Files and folders preserved from the old lab PC

The broader old `Robot_Files` structure included:

```text
Robot_Files/
  OT2Control/
  Protocol_Outputs/
  Plate Reader Data/
  Spectrometer_Interface/
```

For new PC setup, the critical folder is:

```text
OT2Control/
```

Historical output folders such as `Protocol_Outputs` and `Plate Reader Data` do not need to be copied in full unless specifically required.

The desktop migration package included files such as:

```text
ssh_key
requirements.txt
runGUI.bat
controller.sh
open_gui.sh
pi_script.sh
test.sh
directory.txt
test_target_1.csv
pickle.pk
PersonalAccesstoken.txt
client_secret_...
xlaunchconfig.xml
airm2.py
arm.py
cont.py
contrDu.py
controlleraa.py
ml.py
mlD.py
ot.py
otD.py
StartTempControl.py
test.py
opentrons-OT2CEP20191017B11-calibration.json
opentrons-OT2CEP20191017B11-pipette-offset-calibration.json
opentrons-OT2CEP20191017B11-tip-length-calibration.json
Temperature Ramp 1.json
lsc.xml
ltml.xml
```

Sensitive items from this set must not be committed to GitHub.

---

## 6. Windows user setup

### Objective

Use a Windows username that preserves the old lab PC path structure.

### Actual setup

The new PC was set up with Windows user:

```text
science_356_lab
```

This creates the expected Windows user folder:

```text
C:\Users\science_356_lab
```

### Why this matters

The old workflow expects paths under:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

Using the same username reduces broken path problems.

---

## 7. Place the OT2Control folder

### Objective

Place the `OT2Control` folder in the same location as on the old lab PC.

### Expected Windows path

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

### Expected Ubuntu path

```text
/mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

---

## 8. Install Git for Windows

### Objective

Install Git so Windows and GitHub Desktop can manage the repository.

### Recommended Git installer choices

Use these choices during installation:

```text
Git from the command line and also from third-party software
Bundled OpenSSH
Native Windows Secure Channel library
MinTTY terminal
Git Credential Manager
Enable file system caching
Do not enable symbolic links
```

The user selected:

```text
Fast-forward only
```

for the default `git pull` behavior. This is acceptable and conservative.

---

## 9. GitHub Desktop setup

### Objective

Use GitHub Desktop if desired for visual repository management.

### Correct local repository folder

GitHub Desktop should point to:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

Do not point GitHub Desktop at a zip file.

### Branch for migration work

Use:

```text
main-new-pc-opentrons-migration
```

Do not do migration work directly on `main`.

---

## 10. Install Ubuntu / WSL

### Objective

Install Ubuntu through WSL so the new PC can run the same style of Ubuntu commands used by the old lab PC.

### Earlier installation issue

An earlier WSL install attempt showed an error similar to:

```text
A specified logon session does not exist
```

This appeared related to admin/user account context.

### Resolution

The user obtained local admin privileges and installed Ubuntu successfully.

### Ubuntu version installed

The new PC installed:

```text
Ubuntu 26.04 LTS
```

This is newer than the old lab PC's Ubuntu 20.04.6 LTS. That is acceptable for now because Conda will isolate the Python environment.

---

## 11. Create Ubuntu username

### Objective

Use a Linux username matching the old lab PC style.

### Actual Ubuntu username

```text
science_356_lab
```

Example prompt before Conda:

```text
science_356_lab@wks-23-275:~$
```

Example prompt after Conda:

```text
(base) science_356_lab@wks-23-275:~$
```

---

## 12. Validate Ubuntu version

### Command

```bash
cat /etc/os-release
```

### Actual output

```text
PRETTY_NAME="Ubuntu 26.04 LTS"
NAME="Ubuntu"
VERSION_ID="26.04"
VERSION="26.04 (Resolute Raccoon)"
VERSION_CODENAME=resolute
ID=ubuntu
ID_LIKE=debian
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
UBUNTU_CODENAME=resolute
LOGO=ubuntu-logo
```

---

## 13. Validate that Ubuntu can see OT2Control

### Command

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
ls
```

### Actual output

```text
Armchair        Credentials  README.md          deckPositionsGui.py  exceptions.py  pickle.pk
Armchair_Logs   Eve_Out      __pycache__        df_utils.py          ml_models.py   robot_script.sh
Cache           Gui.py       calibrations.json  doc_gen              openGui.py     test_target_1.csv
Controller_Out  LabwareDefs  controller.py      docs                 ot2_robot.py
```

### Interpretation

Success. Ubuntu can access the Windows `OT2Control` folder.

---

## 14. Create an Ubuntu Desktop folder

### Objective

Create a Desktop folder inside Ubuntu so the robot SSH key can be stored there.

### Command

```bash
mkdir -p ~/Desktop
```

`-p` means the command will not fail if the folder already exists.

---

## 15. Copy the robot SSH key into Ubuntu

### Objective

Copy the robot SSH key from the Windows Desktop into Ubuntu.

### Incorrect command attempted

```bash
cp /mnt/Users/science_356_lab/Desktop/ssh_key ~/Desktop/ssh_key
```

### Error received

```text
cp: cannot stat '/mnt/Users/science_356_lab/Desktop/ssh_key': No such file or directory
```

### Cause

The path was missing `/c/`.

In WSL, the Windows C drive is:

```text
/mnt/c
```

not:

```text
/mnt
```

### Correct command

```bash
cp /mnt/c/Users/science_356_lab/Desktop/ssh_key ~/Desktop/ssh_key
```

### Set key permissions

```bash
chmod 600 ~/Desktop/ssh_key
```

### Validate

```bash
ls -la ~/Desktop
```

### Actual output

```text
total 12
drwxr-xr-x 2 science_356_lab science_356_lab 4096 Jul  7 11:49 .
drwxr-x--- 5 science_356_lab science_356_lab 4096 Jul  7 11:48 ..
-rw------- 1 science_356_lab science_356_lab 1896 Jul  7 11:49 ssh_key
```

### Interpretation

Success. The key exists and has restrictive permissions:

```text
-rw-------
```

---

## 16. Install basic Ubuntu tools

### Objective

Install command-line tools needed for setup and troubleshooting.

### Commands

```bash
sudo apt update
sudo apt install -y wget curl git unzip tmux
```

### Password prompt

Ubuntu asked for the sudo password:

```text
[sudo: authenticate] Password:
```

Type the Ubuntu password. It may not show characters while typing.

### `apt update` result

Ubuntu downloaded package lists successfully.

It also reported:

```text
59 packages can be upgraded. Run 'apt list --upgradable' to see them.
```

### Decision

Do not run:

```bash
sudo apt upgrade
```

at this stage unless a specific problem requires it.

### Validation commands

```bash
git --version
tmux -V
wget --version | head -n 1
curl --version | head -n 1
```

### Actual validated versions

```text
git version 2.53.0
tmux 3.6
GNU Wget 1.25.0 built on linux-gnu.
curl 8.18.0 (x86_64-pc-linux-gnu) libcurl/8.18.0 OpenSSL/3.5.5 zlib/1.3.1 brotli/1.2.0 zstd/1.5.7 libidn2/2.3.8 libpsl/0.21.2 libssh2/1.11.1 nghttp2/1.68.0 librtmp/2.3 mit-krb5/1.22.1 OpenLDAP/2.6.10
```

---

## 17. Install Miniconda

### Objective

Install Conda so the old lab PC Anaconda-style Python workflow can be reproduced.

### Why Miniconda is used

The old lab PC used Anaconda Python:

```text
/home/science_356_lab/anaconda3/bin/python
```

Miniconda is lighter than full Anaconda but still provides Conda environments.

This lets us create a dedicated legacy environment rather than using Ubuntu's system Python.

### Commands

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_25.5.1-0-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```

### Installer prompts

During installation:

1. Press `Enter` to view the license.
2. Continue through the license.
3. Type `yes` to accept.
4. Accept the default install location:

```text
/home/science_356_lab/miniconda3
```

5. When asked whether to initialize Conda / update the shell profile, type:

```text
yes
```

### Clarification about the “undo” message

The installer says the shell initialization can be undone later. That is only informational.

Answering `yes` initializes Conda automatically. It does not undo the installation.

### Restart Ubuntu

Close Ubuntu completely and reopen it.

### Validate

```bash
conda --version
python --version
which python
```

### Actual output

```text
conda 25.5.1
Python 3.9.23
/home/science_356_lab/miniconda3/bin/python
```

### Interpretation

Miniconda is installed and active. Base Python is 3.9.23, which is acceptable because the dedicated legacy environment will use Python 3.9.12.

---

## 18. Bookkeeping validation block

### Objective

Record key versions and Git status.

### Safe to paste all at once?

Yes. This block only prints information. It does not commit, push, install, or modify code.

### Command block

```bash
echo "=== WSL / Ubuntu ==="
cat /etc/os-release

echo
echo "=== Kernel ==="
uname -a

echo
echo "=== Conda ==="
conda --version
conda info --envs

echo
echo "=== Base Python ==="
python --version
which python

echo
echo "=== Git/Tmux/Wget/Curl ==="
git --version
tmux -V
wget --version | head -n 1
curl --version | head -n 1

echo
echo "=== OT2Control Git ==="
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
git status
git branch
git remote -v
```

### Actual version outputs included

```text
Ubuntu 26.04 LTS
Linux wks-23-275 6.18.33.2-microsoft-standard-WSL2
conda 25.5.1
Python 3.9.23
/home/science_356_lab/miniconda3/bin/python
git version 2.53.0
tmux 3.6
GNU Wget 1.25.0
curl 8.18.0
```

The Git portion initially produced a safe-directory warning.

---

## 19. Fix Git safe-directory warning

### Error encountered

```text
fatal: detected dubious ownership in repository at '/mnt/c/Users/science_356_lab/Robot_Files/OT2Control'
To add an exception for this directory, call:

        git config --global --add safe.directory /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

### Meaning

Newer Git versions protect users from suspicious repository ownership. This happened because Ubuntu/WSL was accessing a repository stored on the Windows filesystem.

### Fix

```bash
git config --global --add safe.directory /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

### Validate after fix

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
git status
git branch
git remote -v
```

### Successful result before branch switch

```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

Remote:

```text
origin  https://github.com/Hendricks-Laboratory/OT2Control.git (fetch)
origin  https://github.com/Hendricks-Laboratory/OT2Control.git (push)
```

---

## 20. Switch to the migration branch

### Objective

Use a safe branch for migration work.

### Migration branch

```text
main-new-pc-opentrons-migration
```

### Commands used

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
git fetch origin
git checkout main-new-pc-opentrons-migration
```

### Fetch output included

```text
From https://github.com/Hendricks-Laboratory/OT2Control
   5fa5dd7..137dbaf  Auto-RTG   -> origin/Auto-RTG
```

### Warning during checkout

```text
error: chmod on /mnt/c/Users/science_356_lab/Robot_Files/OT2Control/.git/config.lock failed: Operation not permitted
error: chmod on /mnt/c/Users/science_356_lab/Robot_Files/OT2Control/.git/config.lock failed: Operation not permitted
```

### Why this probably happened

The repository is stored under `/mnt/c`, which is the Windows filesystem. Git inside WSL sometimes cannot apply Linux-style permission changes to Windows files.

### Why it was not fatal

Git continued and reported:

```text
branch 'main-new-pc-opentrons-migration' set up to track 'origin/main-new-pc-opentrons-migration'.
Switched to a new branch 'main-new-pc-opentrons-migration'
```

### Validate branch

```bash
git branch
git pull origin main-new-pc-opentrons-migration
git status
```

### Actual validated result

```text
* main-new-pc-opentrons-migration
Already up to date.
On branch main-new-pc-opentrons-migration
nothing to commit, working tree clean
```

---

## 21. Record Git settings

### Objective

Confirm the repository, branch, filemode setting, and safe-directory setting.

### Commands

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control

echo "=== Git remote ==="
git remote -v

echo
echo "=== Git branch/status ==="
git branch
git status

echo
echo "=== Git filemode setting ==="
git config --get core.filemode || echo "core.filemode not set"

echo
echo "=== Safe directories ==="
git config --global --get-all safe.directory
```

### Actual output

```text
=== Git remote ===
origin  https://github.com/Hendricks-Laboratory/OT2Control.git (fetch)
origin  https://github.com/Hendricks-Laboratory/OT2Control.git (push)

=== Git branch/status ===
  Auto
  Auto-RTG
  camera
  emailNotif
  guiTeam
  guiv2
  lab-pc-main-backup-2
  list
  main
* main-new-pc-opentrons-migration
  main2
  mla
  oak_staging_2
  testing

On branch main-new-pc-opentrons-migration
nothing to commit, working tree clean

=== Git filemode setting ===
false

=== Safe directories ===
/mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

### Interpretation

This is the desired state.

`core.filemode=false` is good for a Windows-mounted repository because Git should ignore Linux/Windows file permission bit differences.

---

## 22. GitHub HTTPS versus old lab PC SSH

### Observation

The old lab PC asks for an SSH identity when fetching/pulling from GitHub.

### New PC state

The new PC remote is HTTPS:

```text
https://github.com/Hendricks-Laboratory/OT2Control.git
```

### Decision

Keep HTTPS on the new PC for now.

### Reason

HTTPS is already working and simpler for the new PC. The robot SSH key is separate and still preserved for robot access.

---

## 23. Create documentation files

### Objective

Create living documentation that can be updated throughout the setup.

### Files

```text
NEW_LAB_PC_OT2CONTROL_SETUP_SOP.md
NEW_LAB_PC_OT2CONTROL_MIGRATION_LOG.md
```

### Where created

These were created on the current/editing PC, not directly on the new PC.

### Correct workflow

1. Edit docs on the current PC.
2. Commit to `main-new-pc-opentrons-migration`.
3. Push to GitHub.
4. Pull the branch on the new PC.

### Suggested commit message

```text
Add novice-friendly new lab PC OT2Control setup SOP and migration log
```

---

## 24. Create the legacy Conda environment

### Objective

Create a Conda environment that matches the old lab PC Python version.

### Intended command

```bash
conda create -n ot2control_legacy python=3.9.12 -y
```

Then:

```bash
conda activate ot2control_legacy
python --version
which python
```

Expected:

```text
Python 3.9.12
/home/science_356_lab/miniconda3/envs/ot2control_legacy/bin/python
```

---

## 25. Troubleshooting: Conda Terms of Service not accepted

### Error encountered

When trying to create the legacy Conda environment, Conda reported a Terms of Service not accepted error.

### Meaning

Conda required acceptance of the Anaconda channel Terms of Service before creating an environment from those default channels.

### Fix to try

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Then retry:

```bash
conda create -n ot2control_legacy python=3.9.12 -y
```

### If the command is not recognized

If Conda says `conda tos accept` is not recognized, stop and record the exact error before trying anything else.

---

## 26. Next step after Conda environment exists

After `ot2control_legacy` exists and is active, install the old lab PC Opentrons Python package versions:

```bash
pip install opentrons==5.0.1 opentrons-shared-data==5.0.1
```

Verify:

```bash
python -c "import opentrons; print(opentrons.__version__)"
```

Expected:

```text
5.0.1
```

Do not install the remaining packages until this is validated.

---

## 27. Future steps not yet completed

The following sections must be added later after they are performed and validated:

1. Accept Conda Terms of Service.
2. Create `ot2control_legacy`.
3. Install `opentrons==5.0.1`.
4. Install remaining old lab PC dependencies.
5. Freeze and record package versions.
6. Compile-check Python files.
7. Validate Google Drive appears as `G:` in Windows.
8. Validate Opentrons App version on the new PC.
9. Validate robot software version.
10. Validate robot wired IP.
11. Validate robot SSH connection.
12. Start robot-side listener.
13. Run first simulation.
14. Run first live/no-simulation command only after all prior checks pass.

---

## 28. Current checkpoint status

Completed:

```text
Windows user created as science_356_lab
OT2Control folder placed under C:\Users\science_356_lab\Robot_Files\OT2Control
Ubuntu/WSL installed
Ubuntu version validated as 26.04 LTS
Ubuntu user created as science_356_lab
Ubuntu can see OT2Control folder
Robot SSH key copied to ~/Desktop/ssh_key
Robot SSH key permissions set to -rw-------
Ubuntu tools installed and validated
Miniconda installed and validated
Git safe.directory issue fixed
Migration branch active
Git working tree clean
Git remote uses HTTPS
core.filemode=false
Living documentation files created on current PC
Conda Terms of Service issue encountered before creating legacy environment
```

Pending:

```text
Accept Conda Terms of Service
Create ot2control_legacy
Install opentrons==5.0.1
Install remaining dependencies
Compile-check code
Validate robot SSH
Validate Opentrons App / robot version
Validate Google Drive G:
Run simulation
Run live/no-sim only after prior validations
```

---

## 29. Living SOP update protocol

At every major checkpoint:

1. Add the exact command used.
2. Add the expected result.
3. Add the actual result.
4. Add any errors encountered.
5. Add the fix or workaround.
6. Add final validation.
7. Commit and push the updated documentation to:

```text
main-new-pc-opentrons-migration
```

Recommended commit messages:

```text
Update SOP after Conda legacy environment setup
Update SOP after dependency installation
Update SOP after compile validation
Update SOP after robot SSH validation
Update SOP after first simulation
Update SOP after first live OT2Control test
```
