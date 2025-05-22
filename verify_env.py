# filepath: /home/charles/_github/charles-cai/mli-test1/verify_env.py
"""
Environment Verification Script

This script checks which Python environment is being used
and verifies it's the UV-created .venv environment.
"""
import sys
import os
import subprocess

def main():
    """Check and verify the current Python environment"""
    print("\n=== PYTHON ENVIRONMENT VERIFICATION ===\n")
    
    # Print Python executable path
    print(f"Python executable: {sys.executable}")
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    print(f"In virtual environment: {in_venv}")
    
    # Check if it's the expected .venv folder
    expected_venv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    expected_python = os.path.join(expected_venv, "bin", "python")
    
    if sys.executable == expected_python:
        print(f"✅ Using the expected Python in {expected_venv}")
    else:
        print(f"❌ NOT using the expected Python in {expected_venv}")
        print(f"   Current: {sys.executable}")
        print(f"   Expected: {expected_python}")
    
    # Check if UV was used to create the environment
    print("\n-- Checking for UV installation --")
    try:
        # Try to run UV command
        result = subprocess.run(["uv", "--version"], 
                                capture_output=True, 
                                text=True, 
                                check=False)
        if result.returncode == 0:
            print(f"✅ UV is installed: {result.stdout.strip()}")
        else:
            print("❌ UV command returned an error")
            print(result.stderr)
    except FileNotFoundError:
        print("❌ UV command not found in PATH")
    
    # Check installed packages
    print("\n-- Installed Packages --")
    try:
        import pkg_resources
        for package in pkg_resources.working_set:
            print(f"{package.key}=={package.version}")
    except ImportError:
        print("Could not import pkg_resources to list packages")
    
    print("\n=== VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main()