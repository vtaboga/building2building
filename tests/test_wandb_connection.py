#!/usr/bin/env python3
"""
Test script to verify W&B connection works in the container environment.
Run this before running the full sweep to check configuration.
"""

import subprocess
import sys
import os

def debug_environment():
    """Debug the container environment to understand paths."""
    print("🔍 Debug Info:")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   Contents of current directory:")
    try:
        for item in sorted(os.listdir('.')):
            print(f"     - {item}")
    except:
        print("     (Could not list directory)")
    
    # Check if we're in a container
    container_paths = ['/opt/repository', '/opt/repository-host']
    for path in container_paths:
        if os.path.exists(path):
            print(f"   Found container path: {path}")
            try:
                contents = os.listdir(path)
                print(f"     Contents: {sorted(contents)[:5]}...")  # Show first 5 items
            except:
                print(f"     (Could not list {path})")
    print()

def test_wandb_available():
    """Test if wandb command is available."""
    try:
        result = subprocess.run(['wandb', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ W&B available: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ W&B command not available")
        return False

def test_wandb_login():
    """Test if wandb is logged in."""
    try:
        result = subprocess.run(['wandb', 'status'], capture_output=True, text=True, check=True)
        if "Logged in" in result.stdout:
            print("✅ W&B logged in successfully")
            return True
        else:
            print("❌ W&B not logged in")
            print(f"Status: {result.stdout.strip()}")
            return False
    except subprocess.CalledProcessError:
        print("❌ W&B login check failed")
        return False

def test_sweep_config():
    """Test if sweep configuration is valid."""
    import yaml
    import os
    
    # Try multiple possible paths for the config file
    possible_paths = [
        'configs/sweep_dqn_optimization.yaml',
        '../configs/sweep_dqn_optimization.yaml',
        '/opt/repository/configs/sweep_dqn_optimization.yaml',
        '/opt/repository-host/configs/sweep_dqn_optimization.yaml'
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    # Test 1: Check if file exists
    if not config_path:
        print(f"❌ Sweep config file not found. Tried paths:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"   Current working directory: {os.getcwd()}")
        return False
    
    # Test 2: Check if YAML is valid
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Sweep configuration YAML is valid")
    except yaml.YAMLError as e:
        print(f"❌ Sweep configuration YAML error: {e}")
        return False
    
    # Test 3: Check required fields
    required_fields = ['program', 'method', 'metric', 'parameters']
    missing_fields = []
    
    for field in required_fields:
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        print(f"❌ Sweep configuration missing required fields: {missing_fields}")
        return False
    
    # Test 4: Check if program file exists
    program_path = config.get('program', '')
    if program_path:
        # Try multiple possible paths for the program file
        program_possible_paths = [
            program_path,
            f"../{program_path}",
            f"/opt/repository/{program_path}",
            f"/opt/repository-host/{program_path}"
        ]
        
        program_found = False
        for path in program_possible_paths:
            if os.path.exists(path):
                program_found = True
                break
        
        if not program_found:
            print(f"❌ Program file not found. Tried paths:")
            for path in program_possible_paths:
                print(f"   - {path}")
            return False
    
    print("✅ Sweep configuration is valid")
    return True

def main():
    """Run all tests."""
    print("Testing W&B environment setup...")
    print("=" * 50)
    
    # Debug environment first
    debug_environment()
    
    all_passed = True
    login_required = False
    
    # Test 1: W&B availability
    if not test_wandb_available():
        all_passed = False
    
    # Test 2: W&B login
    if not test_wandb_login():
        login_required = True
        print("💡 To login: wandb login")
    
    # Test 3: Sweep config
    if not test_sweep_config():
        all_passed = False
    
    print("=" * 50)
    if all_passed and not login_required:
        print("🎉 All tests passed! Ready to run sweep.")
    elif not all_passed:
        print("⚠️  Configuration issues found. Please fix them before running sweep.")
        sys.exit(1)
    elif login_required:
        print("⚠️  W&B login required. Please login first:")
        print("   wandb login")
        print("   Then run the sweep.")
        sys.exit(1)

if __name__ == "__main__":
    main() 