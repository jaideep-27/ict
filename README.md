# 🔮 NexusAI Analytics Studio

A comprehensive multi-modal intelligence platform built with **Streamlit** featuring AI-powered Video, Text, Audio, Image, and Story analysis with cutting-edge deep learning capabilities.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ About

**NexusAI Analytics Studio** is a state-of-the-art multi-modal data analysis platform with a unique cyberpunk-inspired gradient UI. Powered by advanced AI/ML libraries including TensorFlow, DeepFace, and NLTK, it provides comprehensive analysis across five data types.

## 🚀 Features Overview

### 🎥 Video Intelligence
- Video playback and frame extraction
- Brightness/contrast analysis over time
- Metadata extraction (resolution, FPS, duration)
- Frame-by-frame processing
- Advanced video analytics

### 📝 Text Analytics
- Word frequency analysis with visualizations
- Character distribution analysis
- Sentiment analysis (polarity, subjectivity)
- Keyword extraction with treemap visualization
- **Text-to-Speech** (10+ languages with adjustable speed)
- NLP-powered text processing

### 🎵 Audio Processing
- Audio playback
- Waveform visualization
- Spectrum analysis (spectrogram)
- Spectral features (centroid, rolloff)
- **Speech-to-Text** with multi-language support
- Automatic MP3 to WAV conversion

### 🖼️ Image Analysis ⭐ NEW
- **🎨 Color Analysis**: RGB histograms, dominant colors, brightness/contrast
- **✂️ AI Background Removal**: Using rembg with U^2-Net model
- **👤 Face Analysis**: Age, gender, emotion detection using DeepFace
- **🔍 Edge Detection**: 5 algorithms (Canny, Sobel, Prewitt, Roberts, Scharr)
- **🎭 Filters & Effects**: 8 professional filters
- **📊 Advanced Metrics**: Sharpness, entropy, SNR

## 🛠️ Technology Stack

### Core Framework
- **Streamlit** (1.40.1) - Web application framework
- **Python** (3.8+) - Programming language

### Video Processing
- **OpenCV** (cv2) - Computer vision
- **Pillow** - Image handling

### Text Processing
- **TextBlob** - Sentiment analysis
- **gTTS** - Google Text-to-Speech
- **Plotly** - Interactive visualizations

### Audio Processing
- **Librosa** - Audio analysis
- **SpeechRecognition** - Speech-to-text
- **pydub** - Audio format conversion
- **FFmpeg** - Audio codecs

### Image Processing (AI-Powered) 🤖
- **rembg** (2.0.61) - AI background removal
- **DeepFace** (0.0.95) - Facial analysis
- **ONNX Runtime** (1.16.3) - Deep learning inference
- **scikit-image** - Image processing algorithms
- **scikit-learn** - Machine learning (K-means clustering)
- **matplotlib** - Plotting and visualization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/jaideep-27/ict.git
cd ict
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv streamlit_env
source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip

# Core dependencies
pip install streamlit pandas plotly numpy

# Video & Image processing
pip install opencv-python pillow scikit-image matplotlib

# AI/ML libraries
pip install rembg deepface onnxruntime scikit-learn

# Text processing
pip install textblob gtts

# Audio processing
pip install librosa soundfile pydub SpeechRecognition
```

### Step 4: Install FFmpeg (for audio)
**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## 🚀 Usage

### Run the Application
```bash
# Activate virtual environment
source streamlit_env/bin/activate

# Run Streamlit app
streamlit run run2.py
```

The app will open in your browser at `http://localhost:8501`

### Run CSV Data Analysis (Alternative)
```bash
streamlit run run.py
```

## 📖 User Guide

### Video Analysis
1. Select **🎥 Video Analysis** from sidebar
2. Upload video file (MP4, AVI, MOV, MKV, WebM)
3. View video playback and metadata
4. Explore analysis tabs:
   - Frame Extraction
   - Brightness Analysis
   - Metadata Details

### Text Analysis
1. Select **📝 Text Analysis** from sidebar
2. Enter or upload text file
3. Navigate through tabs:
   - Word Frequency
   - Character Analysis
   - Sentiment Analysis
   - Keyword Extraction
   - **Text-to-Speech** (convert to audio)

### Audio Analysis
1. Select **🎵 Audio Analysis** from sidebar
2. Upload audio file (MP3, WAV, OGG, FLAC)
3. Explore features:
   - Audio playback
   - Waveform visualization
   - Spectrum analysis
   - **Speech-to-Text** (transcribe audio)

### Image Analysis 🆕
1. Select **🖼️ Image Analysis** from sidebar
2. Upload image (JPG, PNG, GIF, BMP, WebP, TIFF)
3. View image properties and metadata
4. Use AI-powered features:
   - **Color Analysis**: Extract dominant colors
   - **Background Removal**: Remove background with AI
   - **Face Analysis**: Detect emotions and demographics
   - **Edge Detection**: Apply algorithms
   - **Filters**: Apply professional effects
   - **Advanced Analysis**: View quality metrics

## 🤖 AI Features

### Background Removal (rembg)
- **Model**: U^2-Net architecture
- **Accuracy**: State-of-the-art segmentation
- **Output**: PNG with transparency
- **Use Cases**: Product photos, portraits, social media

### Face Analysis (DeepFace)
- **Age Estimation**: Predicts age from face
- **Gender Recognition**: Male/Female classification
- **Emotion Detection**: 7 emotions (happy, sad, angry, surprise, fear, disgust, neutral)
- **Ethnicity**: Race prediction
- **Confidence Scores**: Percentage for each attribute

### Speech Recognition
- **Google API**: High accuracy, multi-language
- **Sphinx**: Offline fallback
- **Auto-conversion**: MP3 → WAV for compatibility
- **Output**: Full transcription + keywords

### Text-to-Speech
- **Google TTS**: Natural voices
- **Languages**: English, Spanish, French, German, Hindi, Arabic, Chinese, Japanese, Korean, Russian
- **Speed Control**: 0.5x to 2.0x
- **Format**: MP3 download

## 📊 Example Use Cases

### 1. E-commerce Product Photos
```
Image Analysis → Background Removal → Download PNG
Perfect for online stores with clean product images
```

### 2. Meeting Transcription
```
Audio Analysis → Upload MP3 → Speech-to-Text → Download Transcript
Convert meetings to searchable text
```

### 3. Brand Color Extraction
```
Image Analysis → Color Analysis → Dominant Colors → RGB values
Extract brand colors from logos
```

### 4. Content Localization
```
Text Analysis → Enter English text → Text-to-Speech → Select Spanish → Download
Create multilingual audio content
```

### 5. UX Research
```
Image Analysis → Upload user photo → Face Analysis → Emotion Detection
Measure user sentiment and emotions
```

## 🗂️ Project Structure

```
ict/
├── run.py                          # CSV data analysis app
├── run2.py                         # Unstructured data analysis (main)
├── data.csv                        # Sample CSV data
├── .gitignore                      # Git ignore rules
├── streamlit_env/                  # Virtual environment
│   ├── bin/
│   ├── lib/
│   └── ...
├── README.md                       # This file
├── FEATURES_SUMMARY.md            # Feature documentation
├── IMAGE_ANALYSIS_FEATURES.md     # Detailed image features
├── README_TTS_STT.md              # TTS/STT documentation
├── TESTING_GUIDE.md               # Testing guide
├── AUDIO_FIX_MP3_SUPPORT.md       # Audio fix details
└── WARNING_FIXES.md               # Warning resolution
```

## 🎨 UI Features

### Dark Theme
- Professional dark mode interface
- GitHub-inspired design
- Responsive layout
- Clean typography

### Interactive Components
- File uploaders with format validation
- Real-time progress indicators
- Collapsible sections
- Download buttons for results

### Visualizations
- Interactive Plotly charts
- Matplotlib plots
- Image galleries
- Color palettes

## ⚙️ Configuration

### Environment Variables
No environment variables required. All settings are configurable through the UI.

### Model Downloads
AI models are automatically downloaded on first use:
- **DeepFace models**: ~/.deepface/weights/ (~50-100MB)
- **rembg U^2-Net**: ~/.u2net/ (~176MB)

Internet connection required for first run.

## 🐛 Troubleshooting

### Import Errors
```bash
# If you see "No module named 'xyz'"
pip install xyz
```

### Audio Processing Issues
```bash
# Install FFmpeg
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS
```

### Face Detection Fails
- Use frontal, well-lit faces
- Ensure image resolution > 200px
- Try different image

### Background Removal Slow
- First run downloads model (~176MB)
- Processing time depends on image size
- High-resolution images take longer

### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf streamlit_env
python3 -m venv streamlit_env
source streamlit_env/bin/activate
pip install -r requirements.txt  # Create this file
```

## 📝 Documentation

- **[FEATURES_SUMMARY.md](FEATURES_SUMMARY.md)**: Complete feature list
- **[IMAGE_ANALYSIS_FEATURES.md](IMAGE_ANALYSIS_FEATURES.md)**: Detailed image analysis guide
- **[README_TTS_STT.md](README_TTS_STT.md)**: TTS/STT documentation
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)**: Testing procedures
- **[AUDIO_FIX_MP3_SUPPORT.md](AUDIO_FIX_MP3_SUPPORT.md)**: MP3 support details
- **[WARNING_FIXES.md](WARNING_FIXES.md)**: Warning resolutions

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

### Libraries & Frameworks
- [Streamlit](https://streamlit.io/) - Web framework
- [rembg](https://github.com/danielgatis/rembg) - Background removal
- [DeepFace](https://github.com/serengil/deepface) - Face analysis
- [OpenCV](https://opencv.org/) - Computer vision
- [Librosa](https://librosa.org/) - Audio analysis
- [scikit-image](https://scikit-image.org/) - Image processing

### AI Models
- **U^2-Net** - Salient object detection
- **VGG-Face** - Facial recognition
- **FER** - Emotion recognition

## 📞 Contact

- **GitHub**: [jaideep-27/ict](https://github.com/jaideep-27/ict)
- **Issues**: [Report bugs](https://github.com/jaideep-27/ict/issues)

## 🔄 Version History

### v2.0.0 (2025-10-17) - Latest
- ✨ Added comprehensive Image Analysis module
- 🤖 Integrated AI-powered background removal (rembg)
- 👤 Added facial analysis with emotion detection (DeepFace)
- 🎨 Implemented color analysis and dominant color extraction
- 🔍 Added 5 edge detection algorithms
- 🎭 Included 8 professional image filters
- 📊 Added advanced quality metrics

### v1.5.0 (Previous)
- ✅ Fixed MP3 audio transcription with auto-conversion
- 🔇 Suppressed audio library warnings
- 📚 Added comprehensive documentation

### v1.0.0 (Initial)
- 🎥 Video analysis
- 📝 Text analysis with TTS
- 🎵 Audio analysis with STT

---

**Made with ❤️ using Streamlit, AI/ML, and Python**

⭐ Star this repo if you find it useful!
