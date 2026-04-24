# Push Steps for Private GitHub Repo

Repository URL:
https://github.com/pes1ug23cs178/Generalized-Deepfake-Detection-Framework-with-Scalable-Multimodal-Integration.git

Current local status (already prepared):
- branch: main
- remote: origin configured
- working tree: clean
- latest commit: Initial clean project import

## 1) Verify local state
Run in C:/Users/Student3/Downloads/Deepak:

git status
git remote -v
git log --oneline -n 1

## 2) Push

git push -u origin main

If prompted:
- Username: your GitHub username for the private repo account
- Password: use a Personal Access Token (PAT), not GitHub account password

## 3) If credential prompt does not appear
Use Git Credential Manager login first:

git credential-manager-core configure
git push -u origin main

Or use GitHub CLI:

gh auth login
git push -u origin main

## 4) If remote branch already exists and rejects push

# Option A: fetch and rebase, then push
git fetch origin
git pull --rebase origin main
git push -u origin main

# Option B: only if you intentionally want to replace remote history
git push -u origin main --force

## 5) What is intentionally excluded from push
The repo is configured to skip heavy/unnecessary local files via .gitignore:
- datasets (celebdf_v2/celebdf_v2, ff++)
- training outputs/checkpoints
- virtual environments
- installers/archives
- generated logs and runtime artifacts

## 6) Files included as important project content
- README.md
- .gitignore
- celebdf_v2/train_swin_deepfake.py
- celebdf_v2/requirements.txt
- celebdf_v2/PIPELINE_REFERENCE.md
- run_ffpp_overnight_supervisor.ps1
- TRAINING_ACTIVITY_AND_CLEANUP_REPORT_20260424.md
