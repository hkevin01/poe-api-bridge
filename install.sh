#!/bin/bash

# Poe Code Chat Backend Installation Script

echo "🚀 Installing Poe Code Chat Backend..."
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements-prod.txt

# Check if poe-api-wrapper was installed successfully
if python -c "import poe_api_wrapper" 2>/dev/null; then
    echo "✅ poe-api-wrapper installed successfully"
else
    echo "❌ Failed to install poe-api-wrapper"
    exit 1
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run: python setup_tokens.py"
echo "2. Follow the prompts to enter your Poe tokens"
echo "3. Run: python start.py"
echo "4. Test with: python test_api.py"
echo ""
echo "For more information, see: README.md" 