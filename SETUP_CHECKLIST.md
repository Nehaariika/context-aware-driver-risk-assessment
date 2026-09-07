# Driver Risk Assessment - Windows PowerShell Setup Checklist

**Your current directory:** `C:\Users\nihar\Desktop\driver-monitoring`

## Problem Diagnosis

❌ **Issue 1:** Missing `main.py` in `src/` folder  
❌ **Issue 2:** Missing `requirements.txt` in project root  
❌ **Issue 3:** Missing `conftest.py` (pytest can't find `src` module)  
❌ **Issue 4:** Wrong venv activation syntax

---

## ✅ Solution: Step-by-Step

### Step 1: Download Files
Download these 3 files from the Claude interface:
- `requirements.txt` 
- `main.py`
- `conftest.py`

### Step 2: Place Files in Correct Locations

**After download, move them to your project:**

```
C:\Users\nihar\Desktop\driver-monitoring\
├── requirements.txt          ← Place here (project root)
├── conftest.py               ← Place here (project root)
├── src\
│   ├── main.py               ← Place here (NEW - copy to src/ folder)
│   ├── preprocessing.py
│   ├── risk_scoring.py
│   ├── temporal_analysis.py
│   ├── warning_system.py
│   ├── detection\
│   │   └── detector.py
│   └── enhancement\
│       └── low_light_enhancement.py
└── tests\
    ├── test_*.py
```

**On Windows File Explorer:**
1. Right-click → Copy each downloaded file
2. Navigate to `C:\Users\nihar\Desktop\driver-monitoring\`
3. Paste `requirements.txt` and `conftest.py` here
4. Go to `src\` subfolder
5. Paste `main.py` here

---

### Step 3: Activate Virtual Environment (Windows PowerShell)

Open PowerShell in your project directory and run:

```powershell
# If you have .venv folder:
.\.venv\Scripts\Activate.ps1

# If you get an error about execution policy, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then try again: .\.venv\Scripts\Activate.ps1

# You should see (.venv) prompt like:
# (.venv) PS C:\Users\nihar\Desktop\driver-monitoring>
```

---

### Step 4: Install Dependencies

Once virtual environment is **activated** (you'll see `(.venv)` at the start of prompt):

```powershell
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed opencv-python-4.8.1.78 numpy-1.24.3 mediapipe-0.10.11 ...
```

---

### Step 5: Verify Installation

Run the quick test (no webcam needed):

```powershell
python test_pipeline.py
```

**Expected output:**
```
============================================================
DRIVER RISK ASSESSMENT PIPELINE - VERIFICATION TEST
============================================================

[TEST] Module 1: Preprocessing
✓ Preprocessing works

[TEST] Module 2: Enhancement
✓ Low-light enhancement works

[TEST] Module 3: Detection
✓ Detection works

[TEST] Module 4: Temporal Analysis
✓ Temporal analysis works

[TEST] Module 5: Risk Scoring
✓ Risk scoring works

[TEST] Module 6: Warning System
✓ Warning system works

============================================================
✅ ALL MODULES VERIFIED SUCCESSFULLY
============================================================
```

---

### Step 6: Run Unit Tests

```powershell
pytest tests/ -v
```

**Should see 129 tests passing** ✅

If you still get import errors, try:
```powershell
pytest tests/ -v --import-mode=importlib
```

---

### Step 7: Run with Webcam

```powershell
python src/main.py --source 0
```

**If webcam fails:**
```powershell
# Try different camera index
python src/main.py --source 1
python src/main.py --source 2
```

---

### Step 8: Run with Video File

```powershell
python src/main.py --source "C:\path\to\your\video.mp4"
```

---

### Step 9: Save Results

```powershell
python src/main.py --source 0 --output results.mp4 --db my_session.db
```

Creates:
- `results.mp4` — video with overlays
- `my_session.db` — SQLite database with all events

---

## Quick Troubleshooting

### ❌ "No module named 'src'"

**Fix:** Make sure `conftest.py` is in project root (not in `src/` or `tests/`)

```powershell
# Verify location:
ls conftest.py
# Should show: Mode Name
#             ---- ----
#             -a-- conftest.py
```

### ❌ "can't open file 'src\main.py'"

**Fix:** Make sure `main.py` is in `src/` folder

```powershell
# Verify:
ls src/main.py
# Should show the file exists
```

### ❌ "ModuleNotFoundError: No module named 'mediapipe'"

**Fix:** Reinstall dependencies

```powershell
pip install mediapipe==0.10.11 -U
```

### ❌ "The term '.\.venv\Scripts\Activate.ps1' is not recognized"

**Fix:** Set execution policy first

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### ❌ Webcam shows black screen

**Fix:** Your webcam may need permissions or is in use

```powershell
# Try with different index
python src/main.py --source 1

# Or test with video file first
python src/main.py --source "C:\path\to\video.mp4"
```

---

## File Structure Check

After placing files, your project should look like:

```
C:\Users\nihar\Desktop\driver-monitoring\
│
├── 📄 requirements.txt             ✅ PROJECT ROOT
├── 📄 conftest.py                  ✅ PROJECT ROOT
├── 📄 test_pipeline.py             (optional)
├── 📄 SETUP_GUIDE.md               (optional)
│
├── 📁 src\
│   ├── 📄 main.py                  ✅ NEW - must be here
│   ├── 📄 preprocessing.py
│   ├── 📄 risk_scoring.py
│   ├── 📄 temporal_analysis.py
│   ├── 📄 warning_system.py
│   ├── 📄 __init__.py
│   ├── 📁 detection\
│   │   ├── 📄 detector.py
│   │   └── 📄 __init__.py
│   └── 📁 enhancement\
│       ├── 📄 low_light_enhancement.py
│       └── 📄 __init__.py
│
├── 📁 tests\
│   ├── 📄 test_detector.py
│   ├── 📄 test_*.py (others)
│   └── 📄 __pycache__
│
├── 📁 .venv\                       (virtual environment)
│   └── Scripts\
│       ├── Activate.ps1
│       └── python.exe
│
└── 📁 __pycache__

```

---

## Complete Quick-Start After Setup

Once all files are in place, activate venv and run:

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Verify
python test_pipeline.py

# 3. Run with webcam
python src/main.py --source 0
```

---

## Commands Reference

| Command | Purpose |
|---------|---------|
| `.\.venv\Scripts\Activate.ps1` | Activate virtual environment |
| `deactivate` | Deactivate virtual environment |
| `pip install -r requirements.txt` | Install all dependencies |
| `python test_pipeline.py` | Quick verification (no webcam) |
| `python src/main.py --source 0` | Run with webcam |
| `python src/main.py --source video.mp4` | Run with video file |
| `python src/main.py --source 0 --output out.mp4` | Save output video |
| `pytest tests/ -v` | Run all unit tests |
| `pytest tests/ -v --cov=src` | Run tests with coverage report |

---

## Still Having Issues?

1. **Check Python version:** `python --version` (need 3.10+)
2. **Check pip:** `pip --version`
3. **Reinstall all:** `pip install -r requirements.txt --force-reinstall`
4. **Check paths:** `Get-ChildItem` to list files in current directory

---

**Created:** September 6, 2026  
**For:** Niharika (23CSEC34) - SIST  
**Status:** ✅ Ready to run
