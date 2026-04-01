"""
Lesson 5.11: Creating an ML Project Folder Structure

This lesson explains how to design a professional machine learning repository
that is reproducible, maintainable, and collaboration-friendly.

=======================================================================
1) WHY FOLDER STRUCTURE MATTERS
=======================================================================

In ML, structure is system design.
You are not just training a model. You are managing:
- raw and processed data
- feature logic
- training and evaluation
- persisted artifacts
- reports and logs

A poor structure causes:
- hidden dependencies
- accidental data overwrites
- duplicated code
- pipelines that only run on one laptop

A good structure gives:
- clear separation of concerns
- reproducibility
- safer experimentation
- easier onboarding
- scalable project growth

=======================================================================
2) STANDARD ML PROJECT STRUCTURE
=======================================================================

Recommended layout:

ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── persistence.py
├── models/
├── reports/
├── logs/
├── requirements.txt
├── README.md
└── main.py

=======================================================================
3) THE CORE RULE: SEPARATION OF CONCERNS
=======================================================================

Each folder should have one responsibility.
Do not mix artifacts, code, and data randomly.

Examples:
- raw data should never live beside model binaries
- notebook experiments should not contain production logic
- src should not store runtime outputs

When each folder has one purpose, the data flow becomes obvious.

=======================================================================
4) FOLDER-BY-FOLDER RESPONSIBILITIES
=======================================================================

data/raw/
- Original immutable dataset files
- Never manually edited
- Source of truth for reproducibility

data/processed/
- Generated datasets after cleaning/feature transformations
- Re-creatable from raw data + code

data/external/
- Third-party reference datasets or files
- Optional but useful for separation

notebooks/
- EDA, visualization, prototyping
- Not for production pipeline execution logic

src/
- Production-ready, importable modules
- Single responsibility per module

models/
- Persisted model artifacts and preprocessing pipelines
- Example: model_v1.pkl, preprocessing_pipeline_v1.pkl

reports/
- Metrics JSON, plots, confusion matrix images, summaries

logs/
- Experiment execution logs, timestamps, parameter traces

=======================================================================
5) MODULE DESIGN INSIDE src
=======================================================================

config.py
- Central constants: paths, random_state, test_size, target column
- Prevents hardcoded values across files

data_preprocessing.py
- Data loading, null handling, duplicates, split logic
- No model fitting here

feature_engineering.py
- Encoding, scaling, derived features
- Build and apply preprocessing pipeline

train.py
- Model instantiation and fitting
- Save fitted artifacts via persistence layer

evaluate.py
- Metric computation and model comparison
- Separate from train to avoid leakage confusion

predict.py
- Load artifacts
- Validate new input
- Transform and predict
- Never refit pipeline/model

persistence.py
- Save/load model and pipeline artifacts in one place

=======================================================================
6) TRAINING VS PREDICTION FLOW
=======================================================================

Training flow:
raw data -> preprocess -> feature engineering -> fit model -> evaluate -> save artifacts

Prediction flow:
new data -> load artifacts -> transform -> predict

Structure should make this separation explicit and enforceable.

=======================================================================
7) COMMON STRUCTURAL MISTAKES
=======================================================================

Avoid these anti-patterns:
- Everything inside one notebook
- No separation between raw and processed data
- Saving model binaries inside src
- Hardcoded absolute local file paths
- Missing logs folder
- Missing requirements.txt
- No README setup instructions

If your pipeline only works on your machine, structure is failing.

=======================================================================
8) HOW TO CREATE THE STRUCTURE
=======================================================================

From project root:

Windows PowerShell:
    New-Item -ItemType Directory -Force data, data/raw, data/processed, data/external
    New-Item -ItemType Directory -Force notebooks, src, models, reports, logs
    New-Item -ItemType File -Force README.md, requirements.txt, main.py
    New-Item -ItemType File -Force src/config.py

Linux/macOS:
    mkdir -p data/raw data/processed data/external notebooks src models reports logs
    touch README.md requirements.txt main.py src/config.py

Then add modules incrementally and keep responsibilities strict.

=======================================================================
9) WHY THIS MATTERS IN AN ML SPRINT
=======================================================================

During sprint work, you will:
- iterate on models
- tune hyperparameters
- compare runs
- save multiple artifacts
- debug data issues

Without structure, this becomes chaos.
With structure, it becomes predictable engineering.

=======================================================================
10) PRACTICAL CHECKLIST
=======================================================================

Project is structurally healthy if:
- data/raw exists and is treated immutable
- data/processed is generated, not hand-edited source
- notebooks is exploration-only
- src contains reusable importable modules
- models/reports/logs are separate
- requirements and README are present and accurate
- training and prediction paths are clearly separated

=======================================================================
KEY TAKEAWAY
=======================================================================

Folder structure is not cosmetic.
It is the architecture that enables reproducibility, collaboration,
and production readiness in ML.
"""

if __name__ == "__main__":
    print(__doc__)
