#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting setup process..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
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

# Create and activate virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
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

echo "🏗️ Building Singularity container..."
CONTAINER_PATH="energyplus/singularity_eplus.sif"
DEF_FILE="energyplus/singularity_eplus.def"

if [ ! -f "$CONTAINER_PATH" ]; then
    if command -v singularity &> /dev/null; then
        sudo singularity build "$CONTAINER_PATH" "$DEF_FILE"
    else
        sudo apptainer build "$CONTAINER_PATH" "$DEF_FILE"
    fi
else
    echo "ℹ️ Container already exists, skipping build..."
fi

echo "🧪 Running validation tests..."
pytest tests/test_singularity_eplus.py -v

echo "✅ Setup completed successfully!" 
