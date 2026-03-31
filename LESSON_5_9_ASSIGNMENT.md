# LESSON 5.9 ASSIGNMENT: Creating a Virtual Environment for an ML Project

## Assignment Objective

Master virtual environment creation and management by:
1. Understanding why virtual environments are critical for ML reproducibility
2. Creating a venv for this project
3. Installing and pinning dependencies
4. Testing environment reproducibility
5. Setting up proper .gitignore conventions
6. Documenting the setup process

---

## PART 1: Understanding the Current Project Environment

### Current State

Navigate to your project:

```bash
cd d:\ML-python
```

Check current Python and installed packages:

```bash
python --version
# Output: Python 3.13.1 (or your version)

pip list
# Shows packages installed globally (or in your active environment)
```

### Question 1: Environment Analysis

**Task**: Document current state

1. What Python version are you using?
   Answer: ________________________

2. Run `pip list` and count packages installed:
   Count: ________________________

3. Do you know which packages are needed ONLY for this project vs. others?
   Answer: YES / NO (most students say NO—this is the problem virtual environments solve)

---

## PART 2: Creating a Virtual Environment for ML-python Project

### Step 1: Create venv Directory

Inside your project directory, create the virtual environment:

```bash
cd d:\ML-python
python -m venv venv
```

This creates a `venv/` directory. Verify it exists:

```bash
ls venv/          # Linux/macOS
dir venv          # Windows
```

You should see:
- `bin/` (Linux/macOS) or `Scripts/` (Windows) - Activation scripts and Python interpreter
- `lib/` (Linux/macOS) or `Lib/` (Windows) - site-packages directory for your packages
- `pyvenv.cfg` - Configuration file

### Step 2: Add venv to .gitignore

Create or update `.gitignore` in your project root:

```bash
# Check if .gitignore exists
ls -la .gitignore  (Linux/macOS)
dir .gitignore     (Windows)
```

**If .gitignore doesn't exist**, create it:

```bash
touch .gitignore  (Linux/macOS)
# or manually create in editor (Windows)
```

Add the following to `.gitignore`:

```
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

# Model Artifacts
models/*.pkl
*.pkl
*.joblib

# Test Cache
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

Verify it's added:

```bash
git status
# Should NOT show venv/ directory in uncommitted changes
```

### Step 3: Activate the Virtual Environment

**On Linux/macOS:**

```bash
source venv/bin/activate
```

Your prompt should change to show `(venv)`:

```
(venv) user@machine:ML-python$
```

**On Windows (cmd.exe):**

```bash
venv\Scripts\activate
```

Your prompt should change to:

```
(venv) C:\path\to\ML-python>
```

**On Windows (PowerShell):**

```bash
venv\Scripts\Activate.ps1
```

### Step 4: Verify Activation

Inside the activated environment, check that you're using the right Python:

```bash
(venv) $ which python
# Should show: /path/to/ML-python/venv/bin/python

# OR on Windows:
(venv) $ where python
# Should show: C:\path\to\ML-python\venv\Scripts\python.exe
```

Check pip location:

```bash
(venv) $ which pip
# Should show: /path/to/ML-python/venv/bin/pip
```

Check installed packages (should be minimal):

```bash
(venv) $ pip list
Package    Version
---------- -------
pip        24.0
setuptools 68.0
wheel      0.41.0
```

**Important**: Only these three should be installed. The environment is clean.

---

## PART 3: Installing Project Dependencies

### Step 1: Install Requirements

This project already has a `requirements.txt`. Install all dependencies:

```bash
(venv) $ pip install -r requirements.txt
```

Wait for installation to complete. You should see output like:

```
Collecting pandas==2.1.0
  Downloading pandas-2.1.0-...
  Installing collected packages: pandas, numpy, ...
Successfully installed pandas-2.1.0 numpy-1.24.3 scikit-learn-1.3.0 ...
```

### Step 2: Verify Installation

List installed packages to verify:

```bash
(venv) $ pip list
```

You should see:

```
Package            Version
    ------             -------
certifi            2023.7.22
charset-normalizer 3.2.0
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
wheel              0.41.0
```

### Step 3: Test Full Pipeline

Now run the complete pipeline inside the virtual environment:

```bash
(venv) $ python main.py
```

You should see output:

```
2024-XX-XX XX:XX:XX - INFO - Starting ML pipeline...
2024-XX-XX XX:XX:XX - INFO - Loading data from data/raw/delivery_data.csv
...
2024-XX-XX XX:XX:XX - INFO - Pipeline complete!
```

Check that artifacts were created:

```bash
(venv) $ python -c "import json; print(json.load(open('reports/metrics.json')))"
```

This verifies that your complete environment is working correctly.

### Step 4: Run Tests Inside Environment

```bash
(venv) $ python -m pytest tests/ -v
```

All 23 tests should pass:

```
tests/test_preprocessing.py::TestDataPreprocessing::test_handle_missing_values_median PASSED
tests/test_feature_engineering.py::TestFeatureEngineering::test_encode_categorical_features_output_shape PASSED
...
======================== 23 passed in 6.79s =========================
```

This confirms your environment has all dependencies needed for testing.

---

## PART 4: Understanding requirements.txt

### Analysis Task

Open `requirements.txt` and answer:

```bash
(venv) $ cat requirements.txt
# or open in your editor
```

**Question 1**: How many packages are listed?
Answer: ________________________

**Question 2**: Which package is the ML framework?
Answer: ________________________

**Question 3**: What version of numpy is pinned?
Answer: ________________________

**Question 4**: Why are versions pinned (e.g., `==2.1.0` instead of `pandas`)?

Answer (hint: think about reproducibility):
______________________________________________________________________________
______________________________________________________________________________

---

## PART 5: Testing Environment Reproducibility

### Simulation: Clean Installation

This is what a teammate would do to reproduce your environment.

**Step 1: Deactivate Current Environment**

```bash
(venv) $ deactivate
```

Your prompt returns to normal (no `(venv)` prefix).

**Step 2: Delete Virtual Environment**

```bash
$ rm -rf venv  (Linux/macOS)
# OR
$ rmdir /s venv  (Windows, answer 'y' when prompted)
```

**Step 3: Create Fresh Environment**

```bash
$ python -m venv venv
```

**Step 4: Activate Fresh Environment**

```bash
$ source venv/bin/activate  (Linux/macOS)
# OR
$ venv\Scripts\activate  (Windows)
```

**Step 5: Install from requirements.txt**

```bash
(venv) $ pip install -r requirements.txt
```

**Step 6: Run Pipeline Again**

```bash
(venv) $ python main.py
```

**Question**: Did the pipeline run successfully with the fresh environment?

Answer: YES / NO

**Why this matters**: This simulation shows that anyone can recreate your exact environment with just two files:
1. `requirements.txt` (pins exact versions)
2. `.gitignore` (tells Git to ignore venv/)

This is reproducibility.

---

## PART 6: Virtual Environment Best Practices Checklist

For this project, verify you have followed best practices:

### Checklist

- [ ] **venv/ created** in project root with `python -m venv venv`
- [ ] **venv/ is NOT committed** to Git (verified with `git status`)
- [ ] **venv/ added to .gitignore**
- [ ] **requirements.txt has pinned versions** (contains `==` version specifiers, not just package names)
- [ ] **Environment activated before running scripts** (prompt shows `(venv)`)
- [ ] **All dependencies installed from requirements.txt** with `pip install -r requirements.txt`
- [ ] **main.py runs successfully** inside activated environment
- [ ] **All tests pass** with `pytest tests/ -v`
- [ ] **README.md documents setup steps** for teammates
- [ ] **Setup verified on clean installation** (delete venv, reinstall, verify it works)

---

## PART 7: Linking to Production ML Systems

### Connection: Local Environment → Production Systems

The practices you are learning scale up:

**Local Development (Today's Assignment)**
```
venv/
├── bin/python               ← Your isolated Python
├── lib/site-packages/       ← Your isolated packages
└── requirements.txt         ← Pinned versions
```

**Docker Container (Production)**
```
Dockerfile
├── FROM python:3.11         ← Isolated base image
├── RUN pip install -r requirements.txt  ← Same versions
└── CMD ["python", "main.py"]
```

**Kubernetes Pod (Scaled Production)**
```
pod.yaml
├── image: ml-app:v1.0       ← Docker image with same environment
├── env: requirements        ← Mounted requirements.txt
└── python main.py           ← Runs in container
```

**Key Insight**: The virtual environment concepts you learned today—isolation, reproducibility, version pinning—are the foundation for all ML systems at scale.

---

## PART 8: Answering Assignment Questions

### Question 1: Why Are Virtual Environments Critical?

(Hint: Think about the three layers of reproducibility)

Answer:
______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________

### Question 2: What Is the Purpose of requirements.txt?

Answer:
______________________________________________________________________________
______________________________________________________________________________
______________________________________________________________________________

### Question 3: Why Should venv/ Never Be Committed to Git?

Answer:
______________________________________________________________________________
______________________________________________________________________________

### Question 4: How Would You Ensure a Teammate Can Reproduce Your Environment?

Answer (step-by-step):
1. ____________________________________________________________________
2. ____________________________________________________________________
3. ____________________________________________________________________
4. ____________________________________________________________________

### Question 5: What Tool Would You Use for Non-Python ML Dependencies?

Answer: ________________________
(Hint: See LESSON_5_9_GUIDE.py, Section 3)

---

## PART 9: Documentation Update (README.md)

Add a "Setup" section to your README.md if not already present:

```markdown
## Setup

### Prerequisites
- Python 3.9 or higher
- pip (comes with Python)

### 1. Create Virtual Environment
python -m venv venv

### 2. Activate Virtual Environment

**On Linux/macOS:**
source venv/bin/activate

**On Windows:**
venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Verify Installation
python -m pytest tests/ -v

### 5. Run Pipeline
python main.py

### Deactivate Environment
deactivate
```

**Task**: Add this section to your README.md if not present.

---

## PART 10: Commit Changes to Git

Save your work with a meaningful commit:

```bash
(venv) $ git add .gitignore README.md
(venv) $ git commit -m "lesson-5.9: Add virtual environment setup and documentation

- Create venv/ with isolated dependencies
- Add venv/ to .gitignore to prevent committing
- Pin all package versions in requirements.txt
- Document setup process in README.md
- Verify all tests pass in isolated environment
- Demonstrate environment reproducibility"
```

Verify the commit:

```bash
(venv) $ git log --oneline | head -5
```

---

## Submission Checklist

Before claiming this assignment complete:

- [ ] Virtual environment created with `python -m venv venv`
- [ ] Virtual environment activated (prompt shows `(venv)`)
- [ ] All dependencies installed with `pip install -r requirements.txt`
- [ ] main.py runs successfully: `python main.py`
- [ ] All tests pass: `python -m pytest tests/ -v` (23/23 passing)
- [ ] venv/ is in .gitignore (verified with `git status`)
- [ ] requirements.txt has exact versions (contains `==` specifiers)
- [ ] README.md has setup instructions
- [ ] All changes committed to git with meaningful message
- [ ] Assignment questions in PART 8 answered
- [ ] Environment reproducibility verified (clean install works)

---

## Key Takeaways

### What You Learned

1. **Virtual environments solve the dependency problem** by creating isolated Python spaces
2. **venv is the recommended tool** for standard ML projects (built-in, lightweight)
3. **requirements.txt ensures reproducibility** by pinning exact versions
4. **Proper .gitignore setup** prevents committing environment-specific files
5. **Environment management scales** from local development to production ML systems

### Why This Matters

- **Reproducibility**: Same environment, same results, every time
- **Collaboration**: Teammates can recreate your setup exactly
- **Production Ready**: Foundation for Docker, Kubernetes, MLOps
- **Professional Practice**: Industry standard in ML engineering

### Next Steps

After mastering this lesson, you are ready for:
- Lesson 5.10: Containerization with Docker
- Lesson 5.11: CI/CD Pipelines for ML
- Lesson 5.12: MLOps and Experiment Tracking

---

## Additional Resources

- **Official Python venv Documentation**: https://docs.python.org/3/tutorial/venv.html
- **pip freeze Guide**: https://pip.pypa.io/en/user_guide/#requirements-files
- **Poetry Alternative**: https://python-poetry.org/
- **Conda for Data Science**: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

---

## Questions? Common Issues

**Q: I typed `source venv/bin/activate` but the prompt didn't change.**
A: On Windows, try `venv\Scripts\activate` instead. Or add `C:\path\to\venv\Scripts` to your Windows PATH environment variable.

**Q: I ran `pip install` but packages installed globally, not in venv.**
A: You forgot to activate the environment. Check: `which python` (Linux/macOS) or `where python` (Windows) should point to venv.

**Q: My requirements.txt has no version specifiers (just package names).**
A: This will cause reproducibility problems. Run `pip freeze > requirements.txt` to generate exact versions.

**Q: I accidentally committed venv/ to Git. How do I remove it?**
A: ```bash
git rm -r --cached venv/
echo "venv/" >> .gitignore
git commit -m "Remove venv from git history"
```

**Q: Why does my environment keep asking me to upgrade pip?**
A: It's safe to ignore. If you want to upgrade: `pip install --upgrade pip`

---

**END OF LESSON 5.9 ASSIGNMENT**

Good luck! You are learning professional ML engineering practices. 🚀
