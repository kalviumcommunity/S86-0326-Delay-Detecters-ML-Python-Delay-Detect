# Lesson 5.10 Assignment: Managing Dependencies Using requirements.txt

## Objective
Implement professional dependency management for this ML repository using strict
version pinning and reproducibility validation.

## Part A: Verify Current Environment

1. Check Python version:

```bash
python --version
```

2. Check active environment package list:

```bash
pip list
```

3. Confirm virtual environment workflow is understood:
- Create environment
- Activate environment
- Install from requirements.txt

## Part B: Create/Validate requirements.txt (Pinned)

This sprint requires strict pinning (`==`) for reproducibility.

Required direct dependencies for this project:

```txt
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.1
matplotlib==3.7.2
seaborn==0.12.2
```

Validation command:

```bash
pip install -r requirements.txt
```

## Part C: Reproducibility Test (Clean Rebuild)

Perform the professional reproducibility check.

1. Remove old environment (if present)
2. Recreate environment
3. Install from requirements.txt
4. Run pipeline and tests

Windows PowerShell:

```powershell
if (Test-Path venv) { Remove-Item -Recurse -Force venv }
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
python -m pytest tests/ -v
```

Expected outcome:
- Pipeline executes successfully
- All tests pass
- No missing import errors

## Part D: README Requirements

README must include:
- Required Python version
- How to create and activate virtual environment
- How to install dependencies from requirements.txt
- How to run pipeline
- How to run tests

## Part E: Common Failure Scenarios and Fixes

1. `ModuleNotFoundError`
- Cause: dependency missing from requirements.txt or environment not activated.
- Fix: activate environment and install with `pip install -r requirements.txt`.

2. Model load/serialization errors
- Cause: scikit-learn version mismatch.
- Fix: align versions with pinned requirements and retrain if needed.

3. Different metrics on two machines
- Cause: dependency drift.
- Fix: recreate both environments from the same pinned requirements.

## Part F: Submission Checklist

- [ ] `requirements.txt` exists in repository root
- [ ] Direct dependencies pinned with `==`
- [ ] README setup instructions are clear and complete
- [ ] Fresh environment recreated successfully
- [ ] `python main.py` works in fresh environment
- [ ] `python -m pytest tests/ -v` passes in fresh environment
- [ ] Changes committed to git with meaningful message

## Recommended Commit Message

```txt
lesson-5.10: Add strict dependency management with pinned requirements
```

## Reflection Questions

1. Why is `requirements.txt` essential even when code is version-controlled?
2. What risk appears when using `>=` instead of `==` for sprint submissions?
3. Why should dependency upgrades be intentional and documented?
4. How does dependency pinning contribute to ML reproducibility?
