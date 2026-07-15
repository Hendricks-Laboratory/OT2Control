# New Lab PC OT2Control Migration Log

**File:** `NEW_LAB_PC_OT2CONTROL_MIGRATION_LOG.md`  
**Purpose:** Historical record of what actually happened during the new lab PC setup.  
**Difference from SOP:** The SOP is the clean novice guide. This log preserves the messy real migration path, including errors, warnings, decisions, and validations.

---

## 1. Migration goal

Set up a new Windows lab PC for the existing Hendricks Laboratory `OT2Control` / Opentrons workflow.

Main goal:

```text
Reproduce the old working lab PC behavior first.
```

Secondary goal:

```text
Create a living SOP for future complete-novice setup.
```

Hard constraints:

```text
Do not touch robot-side files during initial setup.
Do not alter the old lab PC.
Do not modernize before legacy reproduction works.
Do not work directly on main for migration changes.
```

---

## 2026-07-15 — Physical plate reader activated during simulation

Command:

```bash
python controller.py -n DEBUG_PC
```

Observed behavior:

```text
The controller entered simulation, but sent real DDE commands to SPECTROstar.
The physical tray moved, the reader shook the plate, and the scan path reached ImportLayout.
SPECTROstar then reported that NC_synthesis did not exist.
```

Root cause in `controller.py`:

```text
Controller._init_pr(simulate=True, no_pr=False) instantiated the real PlateReader.
PlateReader initialization and scan processing issued physical DDE commands.
The code relied on SPECTROstar reloading SimulationMode=1 from its INI file.
That assumption fails when the application is already open or a previous run was interrupted.
```

Migration fix:

```text
Any controller simulation now selects DummyReader unconditionally.
The physical PlateReader is initialized only for a live run where simulate=False and no_pr=False.
```

Separate SPECTROstar migration finding:

```text
NC_synthesis is a named measurement protocol in the BMG User\Definit protocol database.
It is not a SPECTROstar Nano.ini setting.
The fresh V5.50 installation requires restoration of the complete working User\Definit directory
from the preserved old-PC program-folder archive.
```

Validation status:

```text
Automated reader-selection tests passed locally.
New-PC DEBUG_PC simulation validation is still required after pulling the fix.
The NC_synthesis database restore and live dry run remain pending.
```

---

## 2. Early strategy decisions

### Main first, Auto-RTG later

Decision:

```text
Focus first on main.
Defer Auto-RTG-specific work.
```

Reason:

```text
The immediate goal is a safe baseline reproduction.
```

### Legacy first, modern later

Decision:

```text
Use Python 3.9.12 and opentrons==5.0.1 first.
```

Reason:

```text
The old lab PC Python environment used opentrons==5.0.1.
```

### Robot side preserved

Decision:

```text
Do not change robot-side files during setup.
```

Reason:

```text
The robot side is part of the known-working system.
```

---

## 3. Old lab PC facts gathered

Old Windows path:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

Old WSL path:

```text
/mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

Old Ubuntu:

```text
Ubuntu 20.04.6 LTS
```

Old Python:

```text
Python 3.9.12
```

Old Python executable:

```text
/home/science_356_lab/anaconda3/bin/python
```

Old Python packages:

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

Opentrons App / robot software reported separately:

```text
6.0.1
```

Clarification recorded:

```text
Opentrons App / robot software version and Python package version are separate layers.
```

---

## 4. File/archive context

Files previously involved in the migration included:

```text
Robot_Files-20260706T201347Z-3-001.zip
OT2Control-main.zip
OT2Control_labpc_vs_github_main_comparison_2026-07-06.txt
```

Earlier comparison conclusion:

```text
Lab PC OT2Control repo and GitHub main source matched substantively.
Differences were mostly line endings.
```

Lab PC main commit noted earlier:

```text
953d7c708f29e6c06cd8901554c169a1b5bd97c5
```

---

## 5. New PC Windows setup

New Windows username:

```text
science_356_lab
```

Reason:

```text
Preserve expected lab path structure.
```

Expected folder:

```text
C:\Users\science_356_lab\Robot_Files\OT2Control
```

---

## 6. Ubuntu / WSL setup

Earlier WSL issue:

```text
A specified logon session does not exist
```

Interpretation:

```text
Likely due to Windows admin/user context mismatch.
```

Resolution:

```text
User obtained local admin privileges and installed Ubuntu successfully.
```

Ubuntu username:

```text
science_356_lab
```

Ubuntu version validated:

```text
Ubuntu 26.04 LTS
```

Actual `/etc/os-release` output:

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

Kernel bookkeeping:

```text
Linux wks-23-275 6.18.33.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 18 21:54:43 UTC 2026 x86_64 GNU/Linux
```

---

## 7. OT2Control folder visibility

Command:

```bash
cd /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
ls
```

Actual output:

```text
Armchair        Credentials  README.md          deckPositionsGui.py  exceptions.py  pickle.pk
Armchair_Logs   Eve_Out      __pycache__        df_utils.py          ml_models.py   robot_script.sh
Cache           Gui.py       calibrations.json  doc_gen              openGui.py     test_target_1.csv
Controller_Out  LabwareDefs  controller.py      docs                 ot2_robot.py
```

Conclusion:

```text
Ubuntu can see the Windows OT2Control repository.
```

---

## 8. Robot SSH key copy

Created Ubuntu Desktop:

```bash
mkdir -p ~/Desktop
```

Incorrect first copy command:

```bash
cp /mnt/Users/science_356_lab/Desktop/ssh_key ~/Desktop/ssh_key
```

Error:

```text
cp: cannot stat '/mnt/Users/science_356_lab/Desktop/ssh_key': No such file or directory
```

Cause:

```text
Missing /c/ in WSL path.
```

Correct path pattern:

```text
/mnt/c/Users/science_356_lab/Desktop/ssh_key
```

Permissions set:

```bash
chmod 600 ~/Desktop/ssh_key
```

Validation:

```bash
ls -la ~/Desktop
```

Actual output:

```text
total 12
drwxr-xr-x 2 science_356_lab science_356_lab 4096 Jul  7 11:49 .
drwxr-x--- 5 science_356_lab science_356_lab 4096 Jul  7 11:48 ..
-rw------- 1 science_356_lab science_356_lab 1896 Jul  7 11:49 ssh_key
```

Conclusion:

```text
Robot SSH key copied and permissioned correctly.
```

---

## 9. Ubuntu tools installation

Commands:

```bash
sudo apt update
sudo apt install -y wget curl git unzip tmux
```

Observation:

```text
59 packages can be upgraded.
```

Decision:

```text
Do not run apt upgrade during initial setup.
```

Validated versions:

```text
git version 2.53.0
tmux 3.6
GNU Wget 1.25.0 built on linux-gnu.
curl 8.18.0
```

---

## 10. Miniconda installation

Reason:

```text
Old lab PC used Anaconda Python, so Conda is safest for reproducing legacy Python package versions.
```

User asked whether to initialize Conda in shell profile.

Answer:

```text
Yes.
```

Reason:

```text
It makes Conda available automatically and matches old lab PC prompt style.
```

User asked whether the installer's undo message meant saying yes would undo the install.

Clarification:

```text
No. The undo message is informational only.
Saying yes initializes Conda.
```

Validation after install:

```text
conda 25.5.1
Python 3.9.23
/home/science_356_lab/miniconda3/bin/python
```

Conclusion:

```text
Miniconda installed and active.
```

---

## 11. Bookkeeping command block

User asked whether the bookkeeping block could be pasted all at once.

Answer:

```text
Yes. It only prints version/status information.
```

Command block used:

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

Initial Git portion failed with safe-directory warning.

---

## 12. Git safe-directory warning

Error:

```text
fatal: detected dubious ownership in repository at '/mnt/c/Users/science_356_lab/Robot_Files/OT2Control'
To add an exception for this directory, call:

        git config --global --add safe.directory /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

Fix:

```bash
git config --global --add safe.directory /mnt/c/Users/science_356_lab/Robot_Files/OT2Control
```

Validation after fix:

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

## 13. Migration branch switch

Before switch:

```text
main
```

Target branch:

```text
main-new-pc-opentrons-migration
```

Commands:

```bash
git fetch origin
git checkout main-new-pc-opentrons-migration
```

Fetch output included:

```text
From https://github.com/Hendricks-Laboratory/OT2Control
   5fa5dd7..137dbaf  Auto-RTG   -> origin/Auto-RTG
```

Checkout warning:

```text
error: chmod on /mnt/c/Users/science_356_lab/Robot_Files/OT2Control/.git/config.lock failed: Operation not permitted
error: chmod on /mnt/c/Users/science_356_lab/Robot_Files/OT2Control/.git/config.lock failed: Operation not permitted
```

But checkout completed:

```text
branch 'main-new-pc-opentrons-migration' set up to track 'origin/main-new-pc-opentrons-migration'.
Switched to a new branch 'main-new-pc-opentrons-migration'
```

Validation:

```bash
git branch
git pull origin main-new-pc-opentrons-migration
git status
```

Actual result:

```text
* main-new-pc-opentrons-migration
Already up to date.
On branch main-new-pc-opentrons-migration
nothing to commit, working tree clean
```

Conclusion:

```text
The chmod warning was not fatal.
```

---

## 14. GitHub SSH vs HTTPS issue

User noted old lab PC asks for an SSH ID when fetching/pulling.

New PC remote:

```text
https://github.com/Hendricks-Laboratory/OT2Control.git
```

Decision:

```text
Keep new PC on HTTPS for GitHub for now.
```

Reason:

```text
HTTPS is working and avoids extra GitHub SSH key setup.
The robot SSH key remains separate and preserved.
```

---

## 15. Git bookkeeping II

Command:

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

Actual output:

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

Conclusion:

```text
Git repo state is correct.
```

---

## 16. Documentation files created

User clarified the migration is also meant to create a complete living SOP.

Files created on current PC:

```text
NEW_LAB_PC_OT2CONTROL_SETUP_SOP.md
NEW_LAB_PC_OT2CONTROL_MIGRATION_LOG.md
```

Decision:

```text
Generate complete downloadable Markdown files periodically.
```

Workflow:

```text
Current PC edits/pushes docs.
New PC pulls docs from migration branch.
```

---

## 17. Conda Terms of Service issue

Planned command:

```bash
conda create -n ot2control_legacy python=3.9.12 -y
```

Issue:

```text
Terms of service not accepted error
```

Recommended fix:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

Then retry:

```bash
conda create -n ot2control_legacy python=3.9.12 -y
```

Status:

```text
Legacy Conda environment not yet created at this checkpoint.
```

---

## 18. Current checkpoint summary

Completed:

```text
Old lab PC baseline identified
New PC Windows username aligned
OT2Control folder copied and visible from Ubuntu
Ubuntu/WSL installed
Ubuntu user created
Ubuntu version recorded
Robot SSH key copied and chmodded
Ubuntu tools installed
Miniconda installed
Git safe.directory fixed
Migration branch checked out
Git working tree clean
Git remote documented as HTTPS
core.filemode=false documented
Living documentation files created
Conda Terms of Service blocker identified
```

Pending:

```text
Accept Conda Terms of Service
Create ot2control_legacy
Install opentrons==5.0.1
Install remaining dependencies
Compile-check code
Validate Google Drive G:
Validate Opentrons App / robot version
Validate robot SSH
Start robot listener
Run simulation
Run live/no-sim only after validation
```

---

## 19. Future log entry template

Use this template for future milestones:

```text
Date/time:
Step:
Objective:
Commands used:
Expected result:
Actual result:
Errors/warnings:
Fix/workaround:
Validation:
Decision:
Next step:
```
