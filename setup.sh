#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting setup process..."

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3.10 is not installed. Please install Python 3.10 first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Check if Singularity/Apptainer is installed
if ! (command -v singularity &> /dev/null || command -v apptainer &> /dev/null); then
    echo "❌ Neither Singularity nor Apptainer is installed. Please install one of them first."
    exit 1
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
    cat > .env << EOF
# Environment Variables
PYTHONPATH=${PWD}
VIRTUAL_ENV=${PWD}/.venv
PATH=${PWD}/.venv/bin:${PATH}
# Add other environment variables here
EOF
    echo "✨ Created .env file"
else
    echo "ℹ️ .env file already exists, skipping..."
fi

# Install dependencies in virtual environment
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

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

echo "🧪 Running validation tests..."
pytest tests/test_container.py -v

echo "✅ Setup completed successfully!" 
