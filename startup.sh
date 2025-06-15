#!/bin/bash

# Loan Approval Predictor - Startup Script
# This script sets up and runs the Streamlit application

echo "🏦 Loan Approval Predictor - Starting Application"
echo "================================================="

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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "📁 Creating data directory..."
    mkdir data
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir models
fi

# Check if dataset exists
if [ ! -f "data/loan_data.csv" ]; then
    echo "⚠️  Dataset not found at 'data/loan_data.csv'"
    echo "📋 Please place your dataset file in the 'data/' directory with the name 'loan_data.csv'"
    echo ""
    echo "Your CSV should have these columns:"
    echo "  - loan_id"
    echo "  - no_of_dependents"
    echo "  - education"
    echo "  - self_employed"
    echo "  - income_annum"
    echo "  - loan_amount"
    echo "  - loan_term"
    echo "  - cibil_score"
    echo "  - residential_assets_value"
    echo "  - commercial_assets_value"
    echo "  - luxury_assets_value"
    echo "  - bank_asset_value"
    echo "  - loan_status"
    echo ""
    read -p "Press Enter once you've placed the dataset file..."
fi

# Check if model files exist
MODEL_FILES=("models/model.pkl" "models/scaler.pkl")
MODELS_EXIST=true

for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MODELS_EXIST=false
        break
    fi
done

# Train model if it doesn't exist
if [ "$MODELS_EXIST" = false ]; then
    echo "🤖 Model files not found. Training model..."
    echo "⏳ This may take a few minutes..."
    python src/train_model.py
    
    if [ $? -eq 