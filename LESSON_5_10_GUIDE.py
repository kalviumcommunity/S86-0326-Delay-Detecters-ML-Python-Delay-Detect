"""
Lesson 5.10: Managing Dependencies Using requirements.txt

This guide explains how to manage Python dependencies for machine learning projects
using requirements.txt with strict version pinning for reproducibility.

=======================================================================
1) WHY DEPENDENCY MANAGEMENT MATTERS IN ML
=======================================================================

In ML, reproducibility depends on more than code.
Two developers can run the same script and get different metrics if library
versions are different.

Typical impact of dependency drift:
- Different preprocessing behavior (for example, category handling changes)
- Different model defaults across versions
- Serialization incompatibility (model load/save issues)
- Numeric differences that shift evaluation metrics

Core principle:
A reproducible ML project requires both:
1. Isolated environment (venv)
2. Locked dependencies (requirements.txt)

=======================================================================
2) WHAT IS requirements.txt?
=======================================================================

requirements.txt is a plain text dependency manifest.
Each line contains a package specifier, typically:

    package_name==exact.version

Example:
    pandas==2.1.0
    numpy==1.24.3
    scikit-learn==1.3.0

The operator == pins the exact version.
For this sprint, strict pinning is recommended.

=======================================================================
3) VERSION SPECIFIER OPTIONS
=======================================================================

A) Strict pinning (recommended for this sprint)

    scikit-learn==1.3.0

- Maximum reproducibility
- Best for assignment submission and deterministic results

B) Minimum version

    scikit-learn>=1.3.0

- More flexible
- Lower reproducibility
- Can break behavior silently when major changes appear

C) Compatible release

    scikit-learn~=1.3.0

- Allows patch updates only
- Useful for production balance between stability and patch fixes

For this lesson: use ==.

=======================================================================
4) HOW TO INSTALL FROM requirements.txt
=======================================================================

Always do this inside an activated virtual environment.

Windows PowerShell:
    python -m venv venv
    venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Linux/macOS:
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Verify:
    pip list

Run pipeline and tests:
    python main.py
    python -m pytest tests/ -v

=======================================================================
5) HOW TO CREATE requirements.txt
=======================================================================

Method 1: Manual specification (preferred when stack is known)
- Write only direct dependencies intentionally used by project.
- Keeps file readable and maintainable.

Method 2: Freeze installed packages

    pip freeze > requirements.txt

- Captures full environment (direct + transitive dependencies).
- Strong reproducibility, but often noisy.

Recommended team workflow:
1. Start with manual direct dependencies.
2. Validate in a clean environment.
3. If needed, freeze final lock before release/submission.

=======================================================================
6) COMMON MISTAKES
=======================================================================

Mistake 1: Installing packages globally
- Causes project conflicts and hidden coupling.

Mistake 2: Forgetting to update requirements.txt after adding a new package
- Teammates get import errors.

Mistake 3: Using unpinned requirements for reproducibility-critical work
- Results drift over time.

Mistake 4: Skipping clean-environment validation
- "Works on my machine" failure mode.

Mistake 5: Upgrading random packages without documenting
- Breaks old model artifacts and experiments.

=======================================================================
7) REPRODUCIBILITY CHECKLIST
=======================================================================

Before submission:
- requirements.txt exists at repository root
- Versions are pinned with ==
- Virtual environment setup documented in README
- Project runs from a freshly recreated venv
- Tests pass in fresh environment
- Python version requirement documented in README

=======================================================================
8) WHEN DEPENDENCIES BREAK
=======================================================================

Symptoms:
- Model artifact fails to load
- Deprecated APIs become errors
- Installation conflict messages
- Unexpected metric drift

Correct response:
1. Recreate environment from requirements.txt
2. Confirm Python version alignment
3. Re-run pipeline and tests
4. Upgrade dependencies intentionally, not randomly
5. Document changes in commit and README

=======================================================================
9) APPLYING THIS TO THIS REPOSITORY
=======================================================================

Direct dependencies used by this project:
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn

requirements.txt should pin exact versions for all above.
This repository includes modular code + tests, so deterministic dependencies
complete the reproducibility chain.

=======================================================================
10) KEY TAKEAWAY
=======================================================================

requirements.txt is not a formality.
It is part of the ML system contract.

If code is reproducible but dependencies are not, the project is not reproducible.

Use venv + pinned requirements + clean setup docs every time.
"""

if __name__ == "__main__":
    print(__doc__)
