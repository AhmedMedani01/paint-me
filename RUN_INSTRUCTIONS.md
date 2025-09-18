# 🚀 How to Run Hexa Paint Animation

## Quick Start (Recommended)

### Option 1: Run the Setup Script (Easiest)
```bash
# Navigate to the project directory
cd hexa-paint-animation

# Run the automated setup script
python setup_and_run.py
```

### Option 2: Use Batch File (Windows)
```bash
# Double-click or run in command prompt
setup_and_run.bat
```

### Option 3: Use PowerShell Script (Windows)
```powershell
# Run in PowerShell
.\setup_and_run.ps1
```

## Manual Setup (Step by Step)

### 1. Create Virtual Environment
```bash
# Try Python 3.10 first
py -3.10 -m venv venv

# If Python 3.10 not available, use default
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```

### 5. Open in Browser
Navigate to: `http://localhost:5000`

## What the Setup Script Does

1. ✅ Removes any existing virtual environment
2. ✅ Creates new virtual environment with Python 3.10 (or default)
3. ✅ Upgrades pip to latest version
4. ✅ Installs all required dependencies
5. ✅ Tests that core dependencies work
6. ✅ Starts the web application
7. ✅ Opens browser to http://localhost:5000

## Troubleshooting

### If Python 3.10 is not found:
- Download Python 3.10.18 from: https://www.python.org/downloads/release/python-31018/
- Or the script will automatically use your default Python version

### If packages fail to install:
- Make sure you have internet connection
- Try running: `pip install --upgrade pip` first
- Some packages may take time to download

### If the application doesn't start:
- Check that all dependencies are installed
- Make sure port 5000 is not in use
- Try running: `python main.py` manually

## Features

- 🎨 **Drag & Drop Upload** - Easy image upload
- 🎬 **Real-time Animation** - Watch your image transform
- 📥 **Video Download** - Download your animated video
- 📊 **Usage Statistics** - Track animations created
- 📱 **Responsive Design** - Works on all devices

## API Endpoints

- `POST /api/upload` - Upload image and generate video
- `GET /api/download/<filename>` - Download video
- `GET /api/video/<filename>` - Serve video
- `GET /api/stats` - Get usage statistics

---

**🎉 Enjoy creating beautiful animated paintings!**
