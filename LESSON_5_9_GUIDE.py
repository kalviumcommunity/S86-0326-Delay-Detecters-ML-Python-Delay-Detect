"""
LESSON 5.9: Creating a Virtual Environment for an ML Project

========================================================================
COMPREHENSIVE GUIDE TO PYTHON VIRTUAL ENVIRONMENTS FOR ML WORKFLOWS
========================================================================

LEARNING OBJECTIVES:
- Understand why virtual environments are essential for ML projects
- Create and manage isolated Python environments with venv
- Install and pin dependencies using pip and requirements.txt
- Share reproducible environments with teammates
- Avoid common environment-related mistakes
- Connect environment management to production ML systems

========================================================================
SECTION 1: WHY VIRTUAL ENVIRONMENTS ARE CRITICAL
========================================================================

THE PROBLEM WITHOUT VIRTUAL ENVIRONMENTS:

Problem 1: Version Conflicts
-----------
GlobalPython/
├── site-packages/
│   ├── scikit-learn==1.0.0  <- Project A needs this
│   ├── pandas==1.3.0        <- Project B needs 2.0.0
│   ├── numpy==1.21.0        <- Project C needs 1.24.0
│   └── ...
└── (unpredictable behavior when versions conflict)

When you run "pip install scikit-learn==1.3.0", it overwrites the global version.
Now Project A breaks. This is the classic dependency nightmare.

Problem 2: Silent Failures
-----------
You finished a project 6 months ago. It ran perfectly.
You upgrade numpy globally for a new project.
A teammate clones your old repo and runs it. It fails silently with cryptic errors.
Why? The numerical precision changed in numpy. Your old experiments cannot reproduce.

Problem 3: System Pollution
-----------
After installing 50 packages for different projects, your global Python is bloated.
You cannot remember which packages belong to which project.
Uninstalling becomes dangerous—you might break something else.

THE SOLUTION: VIRTUAL ENVIRONMENTS

A virtual environment creates an isolated, sandboxed Python installation:

Project A:
├── venv/
│   ├── bin/python        <- Python interpreter
│   ├── lib/
│   │   └── site-packages/
│   │       ├── scikit-learn==1.0.0  (isolated here)
│   │       ├── pandas==1.3.0
│   │       └── numpy==1.21.0
│   └── pyvenv.cfg
├── src/
├── data/
├── requirements.txt
└── main.py

Project B:
├── venv/
│   ├── bin/python        <- Different Python interpreter
│   ├── lib/
│   │   └── site-packages/
│   │       ├── scikit-learn==1.3.0  (different version, isolated)
│   │       ├── pandas==2.0.0
│   │       └── numpy==1.24.0
│   └── pyvenv.cfg
├── src/
├── data/
├── requirements.txt
└── main.py

Each project has its own Python interpreter and packages. No conflicts.

KEY INSIGHT:
When you activate a virtual environment, your terminal switches to using that 
environment's Python and packages. When you deactivate, you return to your 
system's default Python.

========================================================================
SECTION 2: UNDERSTANDING VIRTUAL ENVIRONMENT MECHANICS
========================================================================

WHAT GETS CREATED WHEN YOU RUN: python -m venv venv

venv/
├── bin/                      # (On macOS/Linux) Executable scripts
│   ├── python               # Python interpreter (points to system Python)
│   ├── pip                  # Package installer
│   ├── activate             # Activation script (bash/zsh)
│   └── activate.csh         # Activation script (csh/tcsh)
│
├── Scripts/                 # (On Windows) Executable scripts
│   ├── python.exe           # Python interpreter
│   ├── pip.exe              # Package installer
│   ├── activate.bat         # Activation script (cmd.exe)
│   └── activate.ps1         # Activation script (PowerShell)
│
├── lib/                     # (On macOS/Linux)
│   └── python3.11/
│       └── site-packages/   # Your installed packages go here
│
├── Lib/                     # (On Windows)
│   └── site-packages/       # Your installed packages go here
│
└── pyvenv.cfg             # Configuration file (don't modify)

KEY FILES:
- activate/activate.bat: Script that modifies your PATH and PYTHONHOME
- site-packages: Directory where pip installs your packages
- pyvenv.cfg: Configuration (references original system Python)

WHAT HAPPENS WHEN YOU ACTIVATE:

Before activation (system Python):
$ python --version
Python 3.13.1

$ which python
/usr/bin/python  <- System Python

After activation (isolated Python):
$ source venv/bin/activate
(venv) $ python --version
Python 3.13.1  <- Same version, but from venv

(venv) $ which python
/path/to/project/venv/bin/python  <- Virtual environment Python

(venv) $ pip list
# Shows only packages installed in venv/lib/site-packages

WHY THIS MATTERS:
The activation script temporarily modifies your shell's PATH environment variable.
When you type `python`, the shell finds venv/bin/python first instead of /usr/bin/python.
This is a clever, non-invasive way to "switch" Python environments.

========================================================================
SECTION 3: TOOLS FOR ENVIRONMENT MANAGEMENT
========================================================================

TOOL 1: venv (Built-in, Recommended)
------

PROS:
  ✓ Built into Python 3.3+
  ✓ No additional installation required
  ✓ Simple and lightweight
  ✓ Works with pip and requirements.txt
  ✓ Sufficient for standard ML projects

CONS:
  ✗ No cross-platform compatibility for environment files
  ✗ Cannot pin non-Python dependencies (e.g., CUDA)
  ✗ Slightly slower than conda for large installs

USAGE:
python -m venv venv              # Create
source venv/bin/activate         # Activate (Linux/macOS)
venv\Scripts\activate            # Activate (Windows)
deactivate                        # Deactivate


TOOL 2: virtualenv
------

PROS:
  ✓ More flexible than venv
  ✓ Better error messages
  ✓ Faster environment creation
  ✓ Python 2 support (though Python 2 is deprecated)

CONS:
  ✗ Requires installation: pip install virtualenv
  ✗ Similar concept to venv (not fundamentally different)

USAGE:
pip install virtualenv
virtualenv venv
source venv/bin/activate


TOOL 3: conda (For Complex Scientific Environments)
------

PROS:
  ✓ Can manage non-Python dependencies (CUDA, system libraries)
  ✓ Excellent for data science (comes with common packages pre-installed)
  ✓ Cross-platform reproducibility with environment.yml
  ✓ Easier to manage compiled packages (TensorFlow, PyTorch)

CONS:
  ✗ Heavier (several hundred MB)
  ✗ Slower than pip for simple installs
  ✗ Overkill for lightweight ML projects
  ✗ Requires Miniconda or Anaconda installation

USAGE:
conda create -n ml-env python=3.11
conda activate ml-env
conda install pandas numpy scikit-learn
conda deactivate


TOOL 4: pipenv (For Complex Dependency Graphs)
------

PROS:
  ✓ Uses Pipfile (more readable than requirements.txt)
  ✓ Automatically creates/removes environments
  ✓ Lock file for exact reproducibility (Pipfile.lock)

CONS:
  ✗ Slower than pip (complex dependency resolution)
  ✗ Less adopted than conda or venv
  ✗ Steeper learning curve

USAGE:
pip install pipenv
pipenv install pandas numpy scikit-learn
pipenv shell


TOOL 5: poetry (For Production Python Packages)
------

PROS:
  ✓ Modern Python packaging standard
  ✓ Excellent for publishing to PyPI
  ✓ pyproject.toml (standardized configuration)
  ✓ Version constraint flexibility

CONS:
  ✗ Overkill for simple ML projects
  ✗ Steeper learning curve
  ✗ Slower dependency resolution than pip

USAGE:
pip install poetry
poetry init
poetry add pandas numpy scikit-learn


RECOMMENDATION FOR THIS COURSE:
Use venv + pip + requirements.txt for standard ML projects.
This is the most common in professional ML engineering.

========================================================================
SECTION 4: STEP-BY-STEP VIRTUAL ENVIRONMENT SETUP
========================================================================

STEP 1: Check Python Version
-----

Before creating an environment, verify Python 3.9+ is installed:

    python --version
    Python 3.13.1  ← Good, 3.9+

or

    python3 --version
    Python 3.13.1  ← Good


STEP 2: Create Virtual Environment
-----

Navigate to your project directory:

    cd d:/ML-python

Create the environment named 'venv':

    python -m venv venv

This creates a venv/ folder. DO NOT commit this to Git.


STEP 3: Add venv to .gitignore
-----

Create or update .gitignore:

    venv/
    __pycache__/
    *.pyc
    .pytest_cache/
    *.pkl
    *.json


STEP 4: Activate the Environment
-----

On Linux/macOS:

    source venv/bin/activate
    (venv) $  ← Prompt changes, now inside environment

On Windows (cmd.exe):

    venv\Scripts\activate
    (venv) C:\path\to\project>  ← Prompt changes

On Windows (PowerShell):

    venv\Scripts\Activate.ps1
    (venv) PS C:\path\to\project>


STEP 5: Verify Activation
-----

Check that pip uses the environment:

    (venv) $ which pip
    /path/to/project/venv/bin/pip  ← Points to environment pip

    (venv) $ pip list
    Package    Version
    ------     -------
    pip        24.0
    setuptools 68.0
    wheel       0.41.0
    (nothing else—environment is clean)


STEP 6: Install Dependencies
-----

Inside the activated environment, install required packages:

    (venv) $ pip install pandas numpy scikit-learn matplotlib seaborn joblib

Verify installation:

    (venv) $ pip list
    Package            Version
    ------             -------
    contourpy          1.1.1
    cycler             0.11.0
    joblib             1.3.1
    kiwisolver         1.4.5
    matplotlib         3.7.2
    numpy              1.24.3
    pandas             2.1.0
    pip                24.0
    scikit-learn       1.3.0
    scipy              1.11.2
    seaborn            0.12.2
    setuptools         68.0


STEP 7: Create requirements.txt
-----

Freeze exact versions for reproducibility:

    (venv) $ pip freeze > requirements.txt

This creates requirements.txt:

    certifi==2023.7.22
    charset-normalizer==3.2.0
    contourpy==1.1.1
    cycler==0.11.0
    idna==3.4
    joblib==1.3.1
    kiwisolver==1.4.5
    matplotlib==3.7.2
    numpy==1.24.3
    pandas==2.1.0
    pip==24.0
    requests==2.31.0
    scikit-learn==1.3.0
    scipy==1.11.2
    seaborn==0.12.2
    setuptools==68.0
    urllib3==2.0.4
    wheel==0.41.0

This file is your reproducibility guarantee.


STEP 8: Deactivate the Environment
-----

When finished working:

    (venv) $ deactivate
    $ python --version
    Python 3.13.1  ← Back to system Python

The environment remains saved. Activate it again when needed:

    $ source venv/bin/activate  (or venv\Scripts\activate on Windows)
    (venv) $ 


STEP 9: Reproduce Environment on Another Machine
-----

A teammate clones your repository:

    $ git clone https://github.com/your-project.git
    $ cd your-project
    $ python -m venv venv
    $ source venv/bin/activate  (or venv\Scripts\activate on Windows)
    (venv) $ pip install -r requirements.txt

They now have the EXACT same environment you created.
This is reproducibility.

========================================================================
SECTION 5: BEST PRACTICES FOR ENVIRONMENT MANAGEMENT
========================================================================

BEST PRACTICE 1: Never Commit venv/ to Git
-----

Why:
- venv/ is machine-specific (paths contain absolute file paths on your machine)
- venv/ is OS-specific (Windows and Linux have different structures)
- venv/ is large (thousands of files, hundreds of MB)

How:
Ensure .gitignore contains:
    venv/

Verify:
    $ git status
    (should NOT show venv/ folder)


BEST PRACTICE 2: Always Activate Before Running Scripts
-----

Wrong way:
    $ python main.py  ← Uses system Python (wrong packages)

Right way:
    $ source venv/bin/activate
    (venv) $ python main.py  ← Uses environment Python (correct packages)

Check if activated:
    (venv) $ pip list  ← Shows your packages


BEST PRACTICE 3: Pin Versions in requirements.txt
-----

Wrong way (will break):
    pandas
    numpy
    scikit-learn

Right way (reproducible):
    pandas==2.1.0
    numpy==1.24.3
    scikit-learn==1.3.0

The second approach guarantees EXACT versions.
Six months from now, when you reinstall, you get the same versions.


BEST PRACTICE 4: Document Setup in README.md
-----

Include clear instructions:

    ## Setup

    ### 1. Create Virtual Environment
    python -m venv venv

    ### 2. Activate Environment
    On Linux/macOS:
    source venv/bin/activate

    On Windows:
    venv\Scripts\activate

    ### 3. Install Dependencies
    pip install -r requirements.txt

    ### 4. Run Pipeline
    python main.py


BEST PRACTICE 5: Use .gitignore to Exclude Environment Files
-----

Recommended .gitignore for ML projects:

    # Virtual Environment
    venv/
    env/
    ENV/

    # Python Caches
    __pycache__/
    *.pyc
    *.pyo
    *.pyd
    .Python

    # Model Artifacts (optional)
    models/*.pkl
    *.pkl

    # Data (optional)
    data/raw/
    data/processed/

    # IDE
    .vscode/
    .idea/
    *.swp

    # OS
    .DS_Store
    Thumbs.db


BEST PRACTICE 6: Regularly Clean Up Environments
-----

Old environments waste disk space. Remove them:

    $ deactivate
    $ rm -rf venv  (on Linux/macOS)
    $ rmdir /s venv  (on Windows)

Then create fresh:

    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt


BEST PRACTICE 7: Test Environment Setup on Fresh Machine
-----

Before committing, verify your setup.txt works:

    $ rm -rf venv
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt
    $ python main.py

If this works on a clean machine, it will work for everyone.


========================================================================
SECTION 6: CONNECTING VIRTUAL ENVIRONMENTS TO ML REPRODUCIBILITY
========================================================================

ML REPRODUCIBILITY REQUIRES THREE LAYERS:

Layer 1: Code Reproducibility
    ✓ Source code versioned with Git
    ✓ Random seeds fixed
    ✓ Function behavior documented

Layer 2: Data Reproducibility
    ✓ Training data versioned or documented
    ✓ Data preprocessing steps recorded
    ✓ Data splits fixed and reproducible

Layer 3: Environment Reproducibility  ← Virtual Environments
    ✓ Python version specified
    ✓ Package versions pinned in requirements.txt
    ✓ Environment creation automated in README

Example: ML Experiment Results Vary Because...

Scenario: You trained a model 6 months ago. Results: 95% accuracy.
Today, you retrain on the same data. Results: 91% accuracy.

Who broke your model?

Possible culprits:

1. Code changed (Git helps—check git log)
2. Data changed (Data versioning helps)
3. Environment changed (Virtual environment helps)

Without virtual environments, layer 3 is invisible.
You cannot pinpoint which library version caused the change.

With virtual environments + requirements.txt:

    $ pip install -r requirements.txt
    # Exact same versions as 6 months ago
    $ python main.py
    # Results reproduce exactly

This is professional ML engineering.

========================================================================
SECTION 7: VIRTUAL ENVIRONMENTS AND PRODUCTION ML
========================================================================

The principles extend to production systems:

LOCAL DEVELOPMENT (This Course)
├── venv/
│   └── site-packages/ (isolated packages)
└── requirements.txt (pins versions)

CONTINUOUS INTEGRATION / DEPLOYMENT
├── Docker container (advanced isolation)
├── requirements.txt (specifies dependencies)
└── environment.yml (for conda-based systems)

PRODUCTION CLOUD DEPLOYMENT
├── K8s pod spec (defines container)
├── requirements.txt (locked dependencies)
└── Docker image (reproducible environment)

ML OPERATIONS (MLOps)
├── Experiment tracking (code + environment + metrics)
├── Model registry (versioned models + environment specs)
└── Automated retraining (consistent environment)

KEY INSIGHT:
Environment management at scale (MLOps) builds on the foundational 
practice you are learning today: virtual environments.

If you cannot manage environments locally, you cannot scale to production.

========================================================================
SECTION 8: COMMON MISTAKES TO AVOID
========================================================================

MISTAKE 1: Forgetting to Activate the Environment
-----

Wrong:
    $ pip install pandas
    # Installs to system Python globally

Right:
    $ source venv/bin/activate
    (venv) $ pip install pandas
    # Installs to venv only

How to check:
    $ which python
    /usr/bin/python  ← NOT activated (system Python)

    $ which python
    /path/to/venv/bin/python  ← Activated (environment Python)


MISTAKE 2: Committing venv/ to Git
-----

Wrong:
    $ git add venv/
    $ git commit -m "Add environment"
    # Wastes hundreds of MB

Right:
    $ echo "venv/" >> .gitignore
    $ git add .gitignore
    $ git commit -m "Add venv to gitignore"


MISTAKE 3: Not Pinning Versions
-----

Wrong:
    pip freeze > requirements.txt
    # But then manually edit to remove versions
    pandas
    numpy

Right:
    pip freeze > requirements.txt
    # Keep exact versions
    pandas==2.1.0
    numpy==1.24.3


MISTAKE 4: Installing Packages Outside Environment
-----

Wrong:
    $ python -m pip install --user pandas
    # --user installs to ~/.local, not venv

Right:
    $ pip install pandas
    # Inside activated environment


MISTAKE 5: Repeatedly Creating New Environments
-----

Wrong:
    $ rm -rf venv
    $ python -m venv venv  (repeat 10 times)
    # Wastes time

Right:
    $ source venv/bin/activate
    $ pip install --upgrade pip
    # One environment, maintained


MISTAKE 6: Mixing Global and Environment Installs
-----

Sign you are doing this wrong:
    $ pip list
    # Shows packages you never installed
    # Or shows conflicting versions

Fix:
    $ deactivate
    $ source venv/bin/activate
    (venv) $ pip list
    # Should show only things you explicitly installed


========================================================================
SECTION 9: PRACTICAL WORKFLOW EXAMPLE
========================================================================

YOU ARE STARTING A NEW ML PROJECT:

Step 1: Create Project Directory
    mkdir my-ml-project
    cd my-ml-project

Step 2: Initialize Git
    git init
    echo "venv/" >> .gitignore
    git add .gitignore
    git commit -m "Initial commit: add gitignore"

Step 3: Create Virtual Environment
    python -m venv venv

Step 4: Activate Environment
    source venv/bin/activate  (Linux/macOS)
    OR
    venv\Scripts\activate  (Windows)

Step 5: Create Project Structure
    mkdir src data models reports logs
    touch README.md requirements.txt main.py

Step 6: Install ML Dependencies
    pip install pandas numpy scikit-learn matplotlib seaborn joblib

Step 7: Freeze Requirements
    pip freeze > requirements.txt

Step 8: Commit to Git
    git add -A
    git commit -m "Set up ML project with dependencies"

Step 9: Share with Teammate
    # Teammate clones repository
    git clone <repo>
    cd my-ml-project
    
    # Teammate sets up locally
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Teammate has identical environment
    python main.py  ← Works perfectly

Step 10: Deactivate When Finished
    deactivate
    # Return to system Python


========================================================================
SECTION 10: QUICK REFERENCE CHEAT SHEET
========================================================================

CREATE ENVIRONMENT:
    python -m venv venv

ACTIVATE (Linux/macOS):
    source venv/bin/activate

ACTIVATE (Windows):
    venv\Scripts\activate

DEACTIVATE:
    deactivate

INSTALL PACKAGE:
    pip install package-name

INSTALL MULTIPLE:
    pip install pandas numpy scikit-learn

INSTALL FROM FILE:
    pip install -r requirements.txt

CREATE REQUIREMENTS:
    pip freeze > requirements.txt

LIST INSTALLED:
    pip list

UNINSTALL PACKAGE:
    pip uninstall package-name

UPGRADE PACKAGE:
    pip install --upgrade package-name

CHECK LOCATION:
    which python  (Linux/macOS)
    where python  (Windows)

CHECK WHICH PIP:
    which pip  (Linux/macOS)
    where pip  (Windows)


========================================================================
KEY TAKEAWAY
========================================================================

A virtual environment is not just a nice-to-have feature.
It is foundational infrastructure for ML engineering.

It enables:
✓ Reproducibility (exact environment, exact results)
✓ Collaboration (teammates get identical setup)
✓ Isolation (projects don't interfere)
✓ Scalability (practices extend to Docker, Kubernetes)

If you cannot reproduce locally, you cannot deploy reliably.

Use virtual environments for every project. Always.

========================================================================
END OF LESSON 5.9
========================================================================
"""

if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("LESSON 5.9: Creating a Virtual Environment for an ML Project")
    print("="*80)
    print("\nKey Learning Points:")
    print("1. Why virtual environments prevent dependency conflicts")
    print("2. How virtual environments work at the OS level")
    print("3. Step-by-step setup with venv (recommended tool)")
    print("4. Best practices for environment management")
    print("5. Connection to ML reproducibility and production systems")
    print("\nFor interactive learning, open this file in your editor.")
    print("Also read LESSON_5_9_ASSIGNMENT.md for practical exercises.")
