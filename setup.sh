#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting setup process..."

# Store the repository root directory
REPO_ROOT="$PWD"

# Parse command line arguments
ENERGYPLUS_PATH=""
SKIP_CONTAINER=false
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --energyplus_path=*)
        ENERGYPLUS_PATH="${key#*=}"
        shift
        ;;
        --skip-container)
        SKIP_CONTAINER=true
        shift
        ;;
        *)
        echo "Unknown option: $key"
        echo "Usage: ./setup.sh [--energyplus_path=/path/to/energyplus] [--skip-container]"
        exit 1
        ;;
    esac
done

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 is not installed. Please install Python 3.10 first."
    exit 1
fi

# Check if pip or pip3 is installed
if ! (command -v pip &> /dev/null || command -v pip3 &> /dev/null); then
    echo "❌ Neither pip nor pip3 is installed. Please install pip first."
    exit 1
fi

# Check if Singularity/Apptainer is installed (only if not skipping containers)
CONTAINER_AVAILABLE=true
if [ "$SKIP_CONTAINER" = true ]; then
    echo "⏭️ Skipping container setup as requested."
    CONTAINER_AVAILABLE=false
elif ! (command -v singularity &> /dev/null || command -v apptainer &> /dev/null); then
    echo "⚠️ Neither Singularity nor Apptainer is installed. Will skip container-related steps."
    CONTAINER_AVAILABLE=false
fi

# Create and activate virtual environment for local development
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
    echo "✨ Virtual environment created"
else
    echo "ℹ️ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Create .env file
echo "📝 Creating .env file..."
if [ ! -f ".env" ]; then
    # First create the basic .env file
    cat > .env << EOF
# Environment Variables
PYTHONPATH=${PWD}
VIRTUAL_ENV=${PWD}/.venv
PATH=${PWD}/.venv/bin:${PATH}
EOF

    # Only try to add EnergyPlus path if container is available
    if [ "$CONTAINER_AVAILABLE" = true ]; then
        # Now try to add EnergyPlus path using the new config_manager
        echo "🔍 Looking for EnergyPlus installation..."
        EPLUS_ENV=$(python3 -c "
import sys
sys.path.append('${PWD}')
from building2building.simulator.config_manager import find_energyplus_path, update_energyplus_path

# Use manually specified path if provided
path = find_energyplus_path('$ENERGYPLUS_PATH')
if path:
    # Save to config file for future use
    update_energyplus_path(path)
    print(f'ENERGYPLUS_PATH={path}')
")
        
        if [ ! -z "$EPLUS_ENV" ]; then
            EPLUS_PATH=$(echo "$EPLUS_ENV" | cut -d'=' -f2)
            
            # Only add project path to PYTHONPATH, not EnergyPlus
            cat > .env << EOF
# Environment Variables
PYTHONPATH=${PWD}
VIRTUAL_ENV=${PWD}/.venv
PATH=${PWD}/.venv/bin:${PATH}
${EPLUS_ENV}
EOF
            
            echo "✨ Added EnergyPlus environment variable to .env"
        else
            echo "⚠️ EnergyPlus installation not found locally. You can rely on the container version or specify the path with --energyplus_path."
        fi
    else
        echo "⚠️ Skipping EnergyPlus path configuration as container is not available."
    fi
    
    # --- Add Building2Building path configuration ---
    echo "# Path configuration for Building2Building" >> .env
    REPO_ROOT="$PWD"
    # Try to use $SCRATCH if set, otherwise $HOME/scratch
    if [ -n "$SCRATCH" ]; then
        SCRATCH_ROOT="$SCRATCH/Building2Building"
    else
        SCRATCH_ROOT="$HOME/scratch/Building2Building"
    fi
    echo "CONTAINER_PATH=${REPO_ROOT}/container/container.sif" >> .env
    echo "RESULTS_DIR=${REPO_ROOT}/results" >> .env
    echo "WANDB_DIR=${SCRATCH_ROOT}/wandb" >> .env
    echo "DATA_ROOT=${SCRATCH_ROOT}/data" >> .env

    echo "✨ Created .env file"
else
    echo "ℹ️ .env file already exists, skipping..."
fi

# Install dependencies in virtual environment
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
echo "📦 Installing package in development mode..."
pip install -e .

# Only build containers if not skipped and Singularity/Apptainer is available
if [ "$CONTAINER_AVAILABLE" = true ]; then
    echo "🏗️ Building Singularity containers..."
    mkdir -p container

    # Download EnergyPlus installer if it doesn't exist
    EPLUS_INSTALLER="container/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh"
    if [ ! -f "$EPLUS_INSTALLER" ]; then
        echo "📥 Downloading EnergyPlus installer..."
        wget -O "$EPLUS_INSTALLER" https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu20.04-x86_64.sh
        chmod +x "$EPLUS_INSTALLER"
    fi

    # Build the containers
    SINGULARITY_CMD="singularity"
    if ! command -v singularity &> /dev/null; then
        SINGULARITY_CMD="apptainer"
    fi

    # Check if debootstrap is installed
    if ! command -v debootstrap &> /dev/null; then
        echo "❌ debootstrap is not installed. Installing..."
        sudo apt-get update && sudo apt-get install -y debootstrap
    fi

    # Build container
    if [ ! -f "container/container.sif" ]; then
        echo "🏗️ Building main container..."
        sudo $SINGULARITY_CMD build container/container.sif container/container.def
    else
        echo "ℹ️ Main container already exists, skipping build..."
    fi

    # Ask if user wants to run tests
    read -p "Do you want to run validation tests? (y/n): " RUN_TESTS
    if [[ $RUN_TESTS =~ ^[Yy]$ ]]; then
        echo "🧪 Running validation tests..."
        pytest tests/test_container.py -v
        pytest tests/test_gym_wrapper.py -v
    else
        echo "⏭️ Skipping tests as requested."
    fi
else
    echo "⚠️ Skipping container build and container-related tests."
    
    # Ask if user wants to run non-container tests
    read -p "Do you want to run non-container tests? (y/n): " RUN_TESTS
    if [[ $RUN_TESTS =~ ^[Yy]$ ]]; then
        echo "🧪 Running non-container tests..."
        pytest tests/test_gym_wrapper.py -v
    else
        echo "⏭️ Skipping tests as requested."
    fi
fi

echo "✅ Setup completed successfully!"