# Lesson 5.11 Assignment: Creating an ML Project Folder Structure

## Objective
Design and validate a professional ML repository structure that enforces
separation of concerns and supports reproducible training/prediction workflows.

## Part A: Validate Folder Layout

Required structure:

- data/raw
- data/processed
- data/external
- notebooks
- src
- models
- reports
- logs
- requirements.txt
- README.md
- main.py

Task:
1. Confirm each folder/file exists.
2. If missing, create it.

Windows PowerShell commands:

```powershell
New-Item -ItemType Directory -Force data/raw, data/processed, data/external
New-Item -ItemType Directory -Force notebooks, src, models, reports, logs
```

## Part B: Enforce Folder Responsibilities

Document and apply these rules:

1. data/raw
- immutable source data only
- never manually edited

2. data/processed
- generated outputs only
- can be regenerated from raw + code

3. notebooks
- EDA and experimentation only
- no production business logic

4. src
- reusable production modules only
- no runtime artifacts

5. models
- persisted model and pipeline artifacts

6. reports
- evaluation outputs

7. logs
- execution and experiment logs

## Part C: Training vs Prediction Separation

Describe your project flow using this template:

Training Flow:
raw data -> preprocess -> feature engineering -> fit -> evaluate -> save artifacts

Prediction Flow:
new data -> load artifacts -> transform -> predict

Task:
- Verify your code files map cleanly to these flows.
- Ensure predict path does not refit models/pipelines.

## Part D: Structural Quality Checks

Run these checks:

1. No model files inside src
2. No raw source datasets mixed with processed files
3. No absolute local-only paths in code
4. README contains setup and run instructions
5. requirements.txt exists with pinned dependency versions

## Part E: Reproducibility and Collaboration Readiness

Checklist:

- [ ] Folder structure is complete
- [ ] Responsibilities are clearly separated
- [ ] New teammate can understand where things belong
- [ ] Artifacts are organized (models/reports/logs)
- [ ] Data flow is clear from folder layout

## Part F: Reflection Questions

1. Why should raw data be treated as immutable?
2. Why should notebooks not contain production logic?
3. How does separating models, reports, and logs improve debugging?
4. How does folder structure reduce data leakage risk?
5. Why is this architecture better for team collaboration?

## Suggested Commit Message

lesson-5.11: Implement professional ML folder structure and responsibilities

## Submission Standard

A valid submission should allow a new developer to clone the repository and
understand project flow quickly by structure alone, before reading detailed code.
