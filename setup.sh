#!/bin/bash
set -e

# Force Python 3.11 compatibility mode
export PYTHON_VERSION=3.11

# Install requirements with legacy build support
pip install --upgrade pip
pip install --no-binary :all: scikit-learn==1.3.2
pip install -r requirements.txt
