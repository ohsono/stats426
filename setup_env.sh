g#!/usr/bin/env bash
# ============================================================
# setup_env.sh — Create and activate a Python 3.12 environment
#                for the Traffic Sign Recognition project.
#
# Supports both conda and pyenv. Auto-detects which is available.
#
# Usage:
#   chmod +x setup_env.sh
#   source setup_env.sh
# ============================================================

ENV_NAME="stats426"
PYTHON_VERSION="3.12"

set -e

# -----------------------------------------------------------
# Detect environment manager
# -----------------------------------------------------------
if command -v conda &>/dev/null; then
    MANAGER="conda"
elif command -v pyenv &>/dev/null; then
    MANAGER="pyenv"
else
    echo "❌ Neither conda nor pyenv found. Please install one first."
    echo "   conda: https://docs.conda.io/en/latest/miniconda.html"
    echo "   pyenv: https://github.com/pyenv/pyenv#installation"
    return 1 2>/dev/null || exit 1
fi

echo "🔍 Detected environment manager: $MANAGER"

# -----------------------------------------------------------
# Conda path
# -----------------------------------------------------------
if [ "$MANAGER" = "conda" ]; then
    # Check if env already exists
    if conda env list | grep -qw "$ENV_NAME"; then
        echo "✅ Conda env '$ENV_NAME' already exists — activating..."
    else
        echo "📦 Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    fi

    echo "🔄 Activating conda env '$ENV_NAME'..."
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    echo "📥 Installing requirements..."
    pip install -r requirements.txt

    echo ""
    echo "✅ Done! Environment '$ENV_NAME' is active."
    echo "   Python: $(python --version)"
    echo "   pip:    $(pip --version)"
fi

# -----------------------------------------------------------
# pyenv path
# -----------------------------------------------------------
if [ "$MANAGER" = "pyenv" ]; then
    # Ensure pyenv-virtualenv plugin is available
    if ! command -v pyenv-virtualenv-init &>/dev/null && ! pyenv commands | grep -q virtualenv; then
        echo "⚠️  pyenv-virtualenv plugin not found. Installing Python only..."
    fi

    # Install Python version if not already installed
    if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}"; then
        echo "📦 Installing Python $PYTHON_VERSION via pyenv..."
        pyenv install "$PYTHON_VERSION"
    else
        echo "✅ Python $PYTHON_VERSION already installed in pyenv."
    fi

    # Create virtualenv if pyenv-virtualenv is available
    if pyenv commands | grep -q virtualenv; then
        if ! pyenv virtualenvs --bare | grep -q "^${ENV_NAME}$"; then
            echo "📦 Creating pyenv virtualenv '$ENV_NAME'..."
            pyenv virtualenv "$PYTHON_VERSION" "$ENV_NAME"
        else
            echo "✅ pyenv virtualenv '$ENV_NAME' already exists."
        fi

        echo "🔄 Activating pyenv virtualenv '$ENV_NAME'..."
        pyenv activate "$ENV_NAME"
    else
        echo "🔄 Setting local Python to $PYTHON_VERSION..."
        pyenv local "$PYTHON_VERSION"
    fi

    echo "📥 Installing requirements..."
    pip install -r requirements.txt

    echo ""
    echo "✅ Done! Environment '$ENV_NAME' is active."
    echo "   Python: $(python --version)"
    echo "   pip:    $(pip --version)"
fi

echo ""
echo "🚀 Quick start:"
echo "   python main.py info          # Check device"
echo "   python -m pytest tests/ -v   # Run all tests"
echo "   python main.py train --help  # Train a model"
