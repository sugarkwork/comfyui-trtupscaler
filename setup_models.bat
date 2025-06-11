@echo off
chcp 65001 > nul 2>&1
setlocal enabledelayedexpansion

REM TensorRT Upscaler Model Setup Script for Windows
REM This script downloads models and converts them to TensorRT engines

echo ========================================
echo TensorRT Upscaler Model Setup
echo ========================================
echo.

REM =======================
REM メイン処理
REM =======================

echo Detecting Python environment...
call :pythonexec "DETECT_PYTHON"

echo.
echo Checking TensorRT installation...
call :pythonexec "CHECK_TENSORRT"

echo.
echo Checking version compatibility...
call :pythonexec "CHECK_VERSIONS"

echo.
echo Setting up models directory...
call :pythonexec "SETUP_DIRS"

echo.
echo Main menu:
echo 1. Download and convert 4x_foolhardy_Remacri model
echo 2. Download and convert 4x-UltraSharp model
echo 3. Convert existing .pth model to TensorRT
echo 4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    call :pythonexec "DOWNLOAD_FOOLHARDY"
    if !errorlevel! equ 0 (
        call :pythonexec "CONVERT_MODEL" "4x_foolhardy_Remacri"
    )
)
if "%choice%"=="2" (
    call :pythonexec "DOWNLOAD_ULTRASHARP"
    if !errorlevel! equ 0 (
        call :pythonexec "CONVERT_MODEL" "4x-UltraSharp"
    )
)
if "%choice%"=="3" (
    call :pythonexec "LIST_PTH_FILES"
    set /p model_choice="Enter model number to convert: "
    call :pythonexec "CONVERT_EXISTING" "!model_choice!"
)
if "%choice%"=="4" goto end

echo.
echo Setup complete!
:end
pause
exit /b

REM =======================
REM Pythonコードセクション
REM =======================

___DETECT_PYTHON_START___
import os
import sys
from pathlib import Path

# Script directory
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

# Try embedded Python first, then venv Python
python_paths = [
    script_dir / "../../../python_embeded/python.exe",
    script_dir / "../../venv/Scripts/python.exe"
]

python_exe = None
for path in python_paths:
    if path.exists():
        python_exe = str(path.resolve())
        break

if python_exe:
    print(f"Found Python: {python_exe}")
    # Save Python path for batch script
    with open(script_dir / "python_path.txt", "w") as f:
        f.write(python_exe)
else:
    print("ERROR: Python not found. Looking for:")
    for path in python_paths:
        print(f"  - {path}")
    sys.exit(1)
___DETECT_PYTHON_END___

___CHECK_TENSORRT_START___
import subprocess
import sys
import os
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

# Read Python path
try:
    with open(script_dir / "python_path.txt", "r") as f:
        python_exe = f.read().strip()
except:
    python_exe = "python"

# Check if trtexec is available
def find_trtexec():
    # Method 1: Check PATH
    try:
        result = subprocess.run(["trtexec", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return "trtexec"
    except:
        pass
    
    # Method 2: Search common paths
    search_paths = [
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/Program Files (x86)/NVIDIA GPU Computing Toolkit", 
        "C:/tools",
        "C:/"
    ]
    
    for base_path in search_paths:
        base = Path(base_path)
        if base.exists():
            for trt_dir in base.glob("TensorRT*"):
                trtexec_path = trt_dir / "bin" / "trtexec.exe"
                if trtexec_path.exists():
                    try:
                        result = subprocess.run([str(trtexec_path), "--help"], 
                                              capture_output=True, text=True, timeout=10)
                        if result.returncode == 0:
                            return str(trtexec_path)
                    except:
                        continue
    
    return None

trtexec_path = find_trtexec()
if trtexec_path:
    print(f"[OK] Found trtexec: {trtexec_path}")
    # Save trtexec path
    with open(script_dir / "trtexec_path.txt", "w") as f:
        f.write(trtexec_path)
else:
    print("ERROR: trtexec not found")
    print("")
    print("Please install TensorRT and ensure trtexec is available.")
    print("Download from: https://developer.nvidia.com/tensorrt")
    print("")
    print("Installation steps:")
    print("1. Download TensorRT for Windows")
    print("2. Extract to a folder")
    print("3. Add the bin directory to your PATH environment variable")
    print("4. Restart this script")
    sys.exit(1)
___CHECK_TENSORRT_END___

___CHECK_VERSIONS_START___
import subprocess
import sys
import re
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

# Read paths
try:
    with open(script_dir / "python_path.txt", "r") as f:
        python_exe = f.read().strip()
    with open(script_dir / "trtexec_path.txt", "r") as f:
        trtexec_path = f.read().strip()
except:
    print("ERROR: Could not read saved paths")
    sys.exit(1)

# Get trtexec version
try:
    result = subprocess.run([trtexec_path, "--help"], 
                          capture_output=True, text=True, timeout=10)
    trtexec_output = result.stdout + result.stderr
    
    print("=== DEBUG: trtexec output (first 1000 chars) ===")
    print(trtexec_output[:1000])
    print("=== END DEBUG ===")
    
    # Convert internal version number to standard format with build number
    def convert_internal_version_with_build(internal_ver, build_num=None):
        """Convert internal version like 101100 with build b33 to 10.11.0.33"""
        if len(internal_ver) == 6 and internal_ver.isdigit():
            major = str(int(internal_ver[:2]))
            minor = str(int(internal_ver[2:4]))
            patch = str(int(internal_ver[4:6]))
            
            if build_num and build_num.isdigit():
                build = str(int(build_num))
                return f"{major}.{minor}.{patch}.{build}"
            else:
                return f"{major}.{minor}.{patch}.0"
        elif len(internal_ver) == 5 and internal_ver.isdigit():
            major = str(int(internal_ver[:1]))
            minor = str(int(internal_ver[1:3]))
            patch = str(int(internal_ver[3:5]))
            
            if build_num and build_num.isdigit():
                build = str(int(build_num))
                return f"{major}.{minor}.{patch}.{build}"
            else:
                return f"{major}.{minor}.{patch}.0"
        return None
    
    # Try to extract version and build number from the specific TensorRT format
    trtexec_version = "unknown"
    
    # First, try to extract the complete version with build number
    full_version_match = re.search(r'RUNNING TensorRT\.trtexec \[TensorRT v([0-9]+)\] \[b([0-9]+)\]', trtexec_output)
    if full_version_match:
        internal_ver = full_version_match.group(1)
        build_num = full_version_match.group(2)
        converted = convert_internal_version_with_build(internal_ver, build_num)
        if converted:
            trtexec_version = converted
            print(f"TensorRT (trtexec): {trtexec_version} [converted from internal v{internal_ver} b{build_num}]")
    
    # If that fails, try other patterns
    if trtexec_version == "unknown":
        version_patterns = [
            r'TensorRT-([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Path-based standard format
            r'TensorRT v([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Version prefix standard
            r'RUNNING TensorRT\.trtexec \[TensorRT v([0-9]+)\]',  # Internal version in brackets (fallback)
            r'# .*TensorRT-([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Comment line with path
            r'# (.+\\TensorRT-([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\\)',  # Full path in comment
            r'tensorrt ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # Lowercase
            r'version ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)',  # General version
        ]
        
        for i, pattern in enumerate(version_patterns):
            version_match = re.search(pattern, trtexec_output, re.IGNORECASE)
            if version_match:
                if len(version_match.groups()) == 2:  # Full path pattern
                    raw_version = version_match.group(2)
                else:
                    raw_version = version_match.group(1)
                
                # Special handling for internal version pattern (pattern index 2) and full path patterns
                if i == 2:  # RUNNING TensorRT.trtexec pattern - internal version (fallback)
                    converted = convert_internal_version_with_build(raw_version)
                    if converted:
                        trtexec_version = converted
                        print(f"TensorRT (trtexec): {trtexec_version} [converted from internal version: {raw_version} (no build number)]")
                        break
                elif i == 4:  # Full path pattern with 2 groups
                    trtexec_version = version_match.group(2)
                    print(f"TensorRT (trtexec): {trtexec_version} [detected with pattern: {pattern}]")
                    break
                else:
                    trtexec_version = raw_version
                    print(f"TensorRT (trtexec): {trtexec_version} [detected with pattern: {pattern}]")
                    break
    
    if trtexec_version == "unknown":
        print("WARNING: Could not extract TensorRT version from trtexec")
        print("Please manually check your TensorRT installation.")
        
        # Ask user to provide version manually
        manual_version = input("Enter TensorRT version manually (e.g., 10.11.0.33) or press Enter to continue with 'unknown': ").strip()
        if manual_version and re.match(r'^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$', manual_version):
            trtexec_version = manual_version
            print(f"Using manually entered version: {trtexec_version}")
        
except Exception as e:
    print(f"ERROR: Failed to get trtexec version: {e}")
    sys.exit(1)

# Check Python tensorrt version
try:
    result = subprocess.run([python_exe, "-c", "import tensorrt as trt; print(trt.__version__)"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        python_version = result.stdout.strip()
        print(f"TensorRT (Python): {python_version}")
    else:
        python_version = "Not installed"
        print(f"TensorRT (Python): Not installed")
except Exception as e:
    python_version = "Not installed"
    print(f"TensorRT (Python): Not installed")

# Save version info
version_info = f"""TensorRT Version Information:

trtexec version: {trtexec_version}
Python tensorrt version: {python_version}

Status: {"Match" if trtexec_version == python_version and python_version != "Not installed" else "Version mismatch" if python_version != "Not installed" else "Python TensorRT not installed"}
"""

with open(script_dir / "tensorrt_version_info.txt", "w") as f:
    f.write(version_info)

print(f"Version information saved to: {script_dir / 'tensorrt_version_info.txt'}")

# Handle version mismatch
if python_version == "Not installed":
    if trtexec_version == "unknown":
        print("")
        print("TensorRT version could not be determined from trtexec.")
        print("Available options:")
        print("1. Install latest TensorRT (recommended)")
        print("2. Install specific version")
        print("3. Skip installation (may cause issues)")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("Installing latest TensorRT and pycuda...")
            result = subprocess.run([python_exe, "-m", "pip", "install", "-U", "tensorrt", "pycuda"])
        elif choice == "2":
            version = input("Enter TensorRT version (e.g., 10.11.0.33): ").strip()
            if re.match(r'^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$', version):
                print(f"Installing TensorRT {version} and pycuda...")
                result = subprocess.run([python_exe, "-m", "pip", "install", "-U", f"tensorrt=={version}", "pycuda"])
            else:
                print("Invalid version format. Skipping installation.")
                result = subprocess.CompletedProcess([], 1)
        else:
            print("Skipping TensorRT installation.")
            result = subprocess.CompletedProcess([], 1)
        
        if result.returncode == 0:
            print("[OK] TensorRT installation completed successfully.")
        else:
            print("WARNING: TensorRT installation failed or skipped.")
            print("You may encounter errors during model conversion.")
    else:
        choice = input(f"Install TensorRT {trtexec_version} for Python? (y/n): ")
        if choice.lower() == 'y':
            print(f"Installing TensorRT {trtexec_version} and pycuda...")
            result = subprocess.run([python_exe, "-m", "pip", "install", "-U", 
                                   f"tensorrt=={trtexec_version}", "pycuda"])
            if result.returncode == 0:
                print("[OK] TensorRT installation completed successfully.")
            else:
                print("ERROR: Failed to install TensorRT")
                sys.exit(1)
        else:
            print("Installation cancelled.")
            sys.exit(1)
elif trtexec_version != python_version and trtexec_version != "unknown":
    print("")
    print("[WARNING] TensorRT version mismatch detected!")
    print("This may cause compatibility issues during model conversion.")
    choice = input(f"Update Python TensorRT to match trtexec version {trtexec_version}? (y/n): ")
    if choice.lower() == 'y':
        print(f"Installing TensorRT {trtexec_version} and pycuda...")
        result = subprocess.run([python_exe, "-m", "pip", "install", "-U", 
                               f"tensorrt=={trtexec_version}", "pycuda"])
        if result.returncode == 0:
            print("[OK] TensorRT installation completed successfully.")
        else:
            print("ERROR: Failed to install TensorRT")
            sys.exit(1)
    else:
        print("Continuing with version mismatch...")
        print("Note: You may encounter errors during model conversion.")
elif trtexec_version == "unknown":
    print("")
    print("[WARNING] TensorRT version could not be determined.")
    print("Continuing with unknown version compatibility.")
    print("Note: You may encounter errors during model conversion.")
else:
    print("[OK] TensorRT versions match. Continuing...")
___CHECK_VERSIONS_END___

___SETUP_DIRS_START___
import os
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
models_dir = script_dir / "../../models/upscale_models"
temp_dir = script_dir / "temp"

# Create directories
models_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

print(f"Models directory: {models_dir.resolve()}")
print(f"Temp directory: {temp_dir.resolve()}")

# Save paths
with open(script_dir / "models_dir.txt", "w") as f:
    f.write(str(models_dir.resolve()))
with open(script_dir / "temp_dir.txt", "w") as f:
    f.write(str(temp_dir.resolve()))
___SETUP_DIRS_END___

___DOWNLOAD_FOOLHARDY_START___
import requests
from pathlib import Path
from tqdm import tqdm
import sys

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
with open(script_dir / "models_dir.txt", "r") as f:
    models_dir = Path(f.read().strip())

model_name = "4x_foolhardy_Remacri"
model_url = "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth?download=true"
pth_path = models_dir / f"{model_name}.pth"

if pth_path.exists():
    choice = input(f"Model already exists: {pth_path}\nOverwrite? (y/n): ")
    if choice.lower() != 'y':
        print("Download skipped.")
        print(f"Using existing model: {pth_path}")
        sys.exit(0)  # Exit successfully so conversion can proceed

print(f"Downloading {model_name}...")
try:
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(pth_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Model downloaded successfully: {pth_path}")
except Exception as e:
    print(f"ERROR: Failed to download model: {e}")
    sys.exit(1)
___DOWNLOAD_FOOLHARDY_END___

___DOWNLOAD_ULTRASHARP_START___
import requests
from pathlib import Path
from tqdm import tqdm
import sys

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
with open(script_dir / "models_dir.txt", "r") as f:
    models_dir = Path(f.read().strip())

model_name = "4x-UltraSharp"
model_url = "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth?download=true"
pth_path = models_dir / f"{model_name}.pth"

if pth_path.exists():
    choice = input(f"Model already exists: {pth_path}\nOverwrite? (y/n): ")
    if choice.lower() != 'y':
        print("Download skipped.")
        print(f"Using existing model: {pth_path}")
        sys.exit(0)  # Exit successfully so conversion can proceed

print(f"Downloading {model_name}...")
try:
    response = requests.get(model_url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(pth_path, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Model downloaded successfully: {pth_path}")
except Exception as e:
    print(f"ERROR: Failed to download model: {e}")
    sys.exit(1)
___DOWNLOAD_ULTRASHARP_END___

___CONVERT_MODEL_START___
import sys
import os
from pathlib import Path

# Add current directory to Python path for imports
script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.insert(0, str(script_dir))

try:
    from convert_model import convert_pth_to_trt
    
    # Debug: Print all arguments
    print(f"DEBUG: sys.argv = {sys.argv}")
    print(f"DEBUG: Number of arguments = {len(sys.argv)}")
    
    # Get model name from arguments
    if len(sys.argv) >= 3:
        model_name = sys.argv[2]  # Second argument after script name and block name
    elif len(sys.argv) >= 2:
        # Try to extract from the second argument if it's not the block name
        arg = sys.argv[1]
        if arg != "CONVERT_MODEL":
            model_name = arg
        else:
            model_name = "4x_foolhardy_Remacri"  # Default
    else:
        model_name = "4x_foolhardy_Remacri"  # Default
    
    print(f"Using model name: {model_name}")
    
    with open(script_dir / "models_dir.txt", "r") as f:
        models_dir = f.read().strip()
    
    print(f"Converting {model_name} to TensorRT...")
    
    # Convert with FP16
    print("Converting with FP16 precision...")
    convert_pth_to_trt(model_name, models_dir, use_fp16=True)
    
    # Convert with FP32  
    print("Converting with FP32 precision...")
    convert_pth_to_trt(model_name, models_dir, use_fp16=False)
    
    print(f"Conversion completed successfully!")
    print(f"Generated files:")
    print(f"- {models_dir}/{model_name}_fp16.trt")
    print(f"- {models_dir}/{model_name}_fp32.trt")
    
except Exception as e:
    print(f"ERROR: Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
___CONVERT_MODEL_END___

___LIST_PTH_FILES_START___
import os
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
with open(script_dir / "models_dir.txt", "r") as f:
    models_dir = Path(f.read().strip())

print(f"Available .pth files in {models_dir}:")
print()

pth_files = list(models_dir.glob("*.pth"))
if not pth_files:
    print("No .pth files found in models directory")
    exit()

for i, pth_file in enumerate(pth_files, 1):
    print(f"{i}. {pth_file.stem}")

# Save file list for batch script
with open(script_dir / "pth_files.txt", "w") as f:
    for pth_file in pth_files:
        f.write(f"{pth_file.stem}\n")
___LIST_PTH_FILES_END___

___CONVERT_EXISTING_START___
import sys
from pathlib import Path

script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

try:
    # Debug: Print all arguments
    print(f"DEBUG: sys.argv = {sys.argv}")
    print(f"DEBUG: Number of arguments = {len(sys.argv)}")
    
    # Get file number from arguments
    if len(sys.argv) >= 3:
        file_num_str = sys.argv[2]  # Second argument after script name and block name
    elif len(sys.argv) >= 2:
        # Try to extract from the second argument if it's not the block name
        arg = sys.argv[1]
        if arg != "CONVERT_EXISTING":
            file_num_str = arg
        else:
            file_num_str = "1"  # Default
    else:
        file_num_str = "1"  # Default
    
    print(f"Using file number string: {file_num_str}")
    
    try:
        file_num = int(file_num_str)
    except ValueError:
        print(f"ERROR: Invalid file number '{file_num_str}'. Must be a number.")
        sys.exit(1)
    
    # Read file list
    with open(script_dir / "pth_files.txt", "r") as f:
        pth_files = [line.strip() for line in f.readlines()]
    
    print(f"Available files: {pth_files}")
    print(f"Selected file number: {file_num}")
    
    if file_num < 1 or file_num > len(pth_files):
        print(f"Invalid selection. Must be between 1 and {len(pth_files)}")
        sys.exit(1)
    
    model_name = pth_files[file_num - 1]
    print(f"Selected model: {model_name}")
    
    # Import and run conversion
    sys.path.insert(0, str(script_dir))
    from convert_model import convert_pth_to_trt
    
    with open(script_dir / "models_dir.txt", "r") as f:
        models_dir = f.read().strip()
    
    print(f"Converting {model_name} to TensorRT...")
    
    # Convert with FP16
    print("Converting with FP16 precision...")
    convert_pth_to_trt(model_name, models_dir, use_fp16=True)
    
    # Convert with FP32
    print("Converting with FP32 precision...")
    convert_pth_to_trt(model_name, models_dir, use_fp16=False)
    
    print(f"Conversion completed successfully!")
    
except Exception as e:
    print(f"ERROR: Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
___CONVERT_EXISTING_END___

REM =======================
REM サブルーチン
REM =======================

:pythonexec
setlocal
set "blockname=%~1"
set "arg1=%~2"

REM Python path detection for first call
if not exist "%~dp0python_path.txt" (
    python -c "import sys; f=open(r'%~f0', encoding='utf-8'); content=f.read(); f.close(); start_marker='___'+sys.argv[1]+'_START___'; end_marker='___'+sys.argv[1]+'_END___'; code=content.split(start_marker)[1].split(end_marker)[0].strip(); exec(code)" "%blockname%" "%arg1%"
) else (
    REM Use detected Python path
    for /f "delims=" %%i in (%~dp0python_path.txt) do set "PYTHON_EXE=%%i"
    "!PYTHON_EXE!" -c "import sys; f=open(r'%~f0', encoding='utf-8'); content=f.read(); f.close(); start_marker='___'+sys.argv[1]+'_START___'; end_marker='___'+sys.argv[1]+'_END___'; code=content.split(start_marker)[1].split(end_marker)[0].strip(); exec(code)" "%blockname%" "%arg1%"
)

endlocal
exit /b