#!/bin/bash

# Script to manually download the U^2-Net model for rembg
# This is useful when automatic download fails due to network issues

echo "🔽 Downloading U^2-Net model for rembg..."
echo "Model size: ~176 MB"
echo ""

# Create directory
mkdir -p ~/.u2net

# Check if model already exists
if [ -f ~/.u2net/u2net.onnx ]; then
    echo "✅ Model already exists at ~/.u2net/u2net.onnx"
    echo "File size: $(du -h ~/.u2net/u2net.onnx | cut -f1)"
    read -p "Do you want to re-download? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping download."
        exit 0
    fi
fi

# Download the model
echo "Downloading from GitHub..."
wget --progress=bar:force:noscroll \
     https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
     -O ~/.u2net/u2net.onnx

# Check if download was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Download successful!"
    echo "Model saved at: ~/.u2net/u2net.onnx"
    echo "File size: $(du -h ~/.u2net/u2net.onnx | cut -f1)"
    echo ""
    echo "You can now use the background removal feature in the app."
else
    echo ""
    echo "❌ Download failed!"
    echo ""
    echo "Alternative methods:"
    echo "1. Try using curl instead:"
    echo "   curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o ~/.u2net/u2net.onnx"
    echo ""
    echo "2. Download manually from browser:"
    echo "   URL: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
    echo "   Save to: ~/.u2net/u2net.onnx"
    echo ""
    echo "3. Check your internet connection and try again"
    exit 1
fi
