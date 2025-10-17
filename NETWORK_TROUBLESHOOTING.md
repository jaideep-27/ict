# 🔧 Network Issues & Model Download Troubleshooting

## Problem: Cannot Download AI Models

### Symptoms
- ❌ Background removal fails with "HTTPSConnectionPool" error
- ❌ Face analysis fails with "Failed to resolve" error
- ❌ "NameResolutionError" or "Temporary failure in name resolution"

### Root Cause
The AI features require downloading pre-trained models on first use:
- **U^2-Net model** (~176 MB) for background removal (rembg)
- **DeepFace models** (~50-100 MB) for face analysis

These are downloaded from GitHub automatically, but network issues can prevent this.

---

## 🔍 Diagnostic Steps

### 1. Check Internet Connection
```bash
# Test basic connectivity
ping -c 3 google.com

# Test GitHub connectivity
ping -c 3 github.com

# Test DNS resolution
nslookup github.com
```

### 2. Check Model Status
```bash
# Check if U^2-Net model exists
ls -lh ~/.u2net/u2net.onnx

# Check if DeepFace models exist
ls -lh ~/.deepface/weights/
```

---

## ✅ Solutions

### Solution 1: Wait and Retry
Sometimes GitHub servers are temporarily unavailable. Wait 5-10 minutes and try again.

### Solution 2: Manual Download (U^2-Net for rembg)

#### Option A: Using the provided script
```bash
# Navigate to project directory
cd /home/nnrg/ICT/ict

# Run the download script
./download_rembg_model.sh
```

#### Option B: Manual wget
```bash
# Create directory
mkdir -p ~/.u2net

# Download model
wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
     -O ~/.u2net/u2net.onnx

# Verify download
ls -lh ~/.u2net/u2net.onnx
```

#### Option C: Using curl
```bash
# Create directory
mkdir -p ~/.u2net

# Download model
curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx \
     -o ~/.u2net/u2net.onnx
```

#### Option D: Browser download
1. Open in browser: https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx
2. Save file as `u2net.onnx`
3. Move to: `~/.u2net/u2net.onnx`
   ```bash
   mkdir -p ~/.u2net
   mv ~/Downloads/u2net.onnx ~/.u2net/
   ```

### Solution 3: Manual Download (DeepFace Models)

#### Download required models manually:
```bash
# Create directory
mkdir -p ~/.deepface/weights

# Age model (~65 MB)
wget https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5 \
     -O ~/.deepface/weights/age_model_weights.h5

# Gender model (~28 MB)  
wget https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5 \
     -O ~/.deepface/weights/gender_model_weights.h5

# Race model (~11 MB)
wget https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_weights.h5 \
     -O ~/.deepface/weights/race_model_weights.h5

# Emotion model (~63 MB)
wget https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5 \
     -O ~/.deepface/weights/facial_expression_model_weights.h5

# VGG-Face model (~509 MB) - for face detection
wget https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5 \
     -O ~/.deepface/weights/vgg_face_weights.h5
```

### Solution 4: Use VPN or Proxy
If GitHub is blocked in your network:
```bash
# Set proxy (if available)
export http_proxy="http://proxy.example.com:8080"
export https_proxy="http://proxy.example.com:8080"

# Then retry download
./download_rembg_model.sh
```

### Solution 5: Configure DNS
If DNS resolution fails:
```bash
# Edit resolv.conf
sudo nano /etc/resolv.conf

# Add Google DNS
nameserver 8.8.8.8
nameserver 8.8.4.4

# Or Cloudflare DNS
nameserver 1.1.1.1
nameserver 1.0.0.1
```

### Solution 6: Disable Features Temporarily
If you can't download models, you can still use other features:
- ✅ Color Analysis - Works without models
- ✅ Edge Detection - Works without models
- ✅ Filters & Effects - Works without models
- ✅ Advanced Analysis - Works without models
- ❌ Background Removal - Requires U^2-Net model
- ❌ Face Analysis - Requires DeepFace models

---

## 🧪 Verify Installation

### Check U^2-Net Model
```bash
# Should show ~176 MB file
ls -lh ~/.u2net/u2net.onnx

# Example output:
# -rw-r--r-- 1 user user 176M Oct 17 12:00 /home/user/.u2net/u2net.onnx
```

### Check DeepFace Models
```bash
# List all models
ls -lh ~/.deepface/weights/

# Should show multiple .h5 files
# age_model_weights.h5 (~65 MB)
# gender_model_weights.h5 (~28 MB)
# race_model_weights.h5 (~11 MB)
# facial_expression_model_weights.h5 (~63 MB)
# vgg_face_weights.h5 (~509 MB)
```

---

## 🔄 Restart Application
After downloading models manually, restart Streamlit:
```bash
# Stop current app (Ctrl+C in terminal)

# Restart
cd /home/nnrg/ICT/ict
source streamlit_env/bin/activate
streamlit run run2.py
```

---

## 📊 Model Size Reference

| Model | Purpose | Size | Location |
|-------|---------|------|----------|
| U^2-Net | Background removal | ~176 MB | `~/.u2net/u2net.onnx` |
| Age Model | Age estimation | ~65 MB | `~/.deepface/weights/age_model_weights.h5` |
| Gender Model | Gender detection | ~28 MB | `~/.deepface/weights/gender_model_weights.h5` |
| Race Model | Ethnicity | ~11 MB | `~/.deepface/weights/race_model_weights.h5` |
| Emotion Model | Emotion detection | ~63 MB | `~/.deepface/weights/facial_expression_model_weights.h5` |
| VGG-Face | Face detection | ~509 MB | `~/.deepface/weights/vgg_face_weights.h5` |

**Total**: ~850 MB (all models)

---

## 🌐 Network Requirements

### Required Domains
- `github.com`
- `release-assets.githubusercontent.com`
- `raw.githubusercontent.com`

### Firewall Rules
If behind corporate firewall, ensure these are whitelisted:
```
*.github.com
*.githubusercontent.com
```

### Ports
- HTTPS: Port 443

---

## 🆘 Still Having Issues?

### Check App Logs
```bash
# The Streamlit terminal will show detailed error messages
# Look for lines containing:
# - "HTTPSConnectionPool"
# - "NameResolutionError"
# - "Failed to resolve"
# - "Connection refused"
```

### Alternative: Use Different Network
- Try mobile hotspot
- Try different WiFi network
- Try VPN service

### Contact Support
If none of the above works:
1. Check GitHub Status: https://www.githubstatus.com/
2. Report issue with error logs
3. Use features that don't require model downloads

---

## ✅ Success Indicators

After successful model download, you should see:
- ✅ "U^2-Net model is installed (~176 MB)" in the app
- ✅ Background removal works without errors
- ✅ Face analysis completes successfully
- ✅ No network errors in console

---

## 📝 Prevention

### Pre-download Models
Before deploying to environments with limited internet:
```bash
# Run the download script on a machine with good internet
./download_rembg_model.sh

# Copy the ~/.u2net and ~/.deepface directories to target machine
scp -r ~/.u2net user@target:/home/user/
scp -r ~/.deepface user@target:/home/user/
```

### Cache Models
Models are cached after first download and won't be re-downloaded.

---

## 🔗 Official Links

- **rembg GitHub**: https://github.com/danielgatis/rembg
- **DeepFace GitHub**: https://github.com/serengil/deepface
- **U^2-Net Paper**: https://arxiv.org/abs/2005.09007
- **Model Releases**: https://github.com/danielgatis/rembg/releases

---

**Last Updated**: October 17, 2025
