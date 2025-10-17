# Text-to-Speech (TTS) and Speech-to-Text (STT) Features

## 🎉 New Features Added to run2.py

### 🔊 Text-to-Speech (TTS)
Located in: **Text Analysis → Text-to-Speech Tab**

**Features:**
- Convert any text to natural-sounding speech
- Support for 10+ languages (English, Spanish, French, German, Hindi, Japanese, etc.)
- Adjustable speech speed (normal/slow)
- Download generated audio as MP3
- View statistics (characters, words, file size)

**Powered by:** Google Text-to-Speech (gTTS)

**How to Use:**
1. Go to "📝 Text Analysis" section
2. Upload or enter your text
3. Click on "🔊 Text-to-Speech" tab
4. Select language and speed
5. Click "🎤 Generate Speech"
6. Listen and download the audio

**Supported Languages:**
- English (US/UK)
- Spanish
- French
- German
- Italian
- Portuguese
- Hindi
- Japanese
- Korean
- Chinese (Mandarin)

---

### 🎤 Speech-to-Text (STT)
Located in: **Audio Analysis → Speech-to-Text Transcription Tab**

**Features:**
- Convert audio speech to written text
- Multiple recognition engines (Google, Sphinx)
- Support for 8+ languages
- Automatic keyword extraction
- Download transcript as text file
- Word count and analysis

**Powered by:** Google Speech Recognition & Sphinx

**How to Use:**
1. Go to "🎵 Audio Analysis" section
2. Upload an audio file (MP3, WAV, etc.)
3. Click on "🎤 Speech-to-Text Transcription" tab
4. Select recognition engine and language
5. Click "🎯 Transcribe Audio"
6. View and download the transcript

**Recognition Engines:**
- **Google Speech Recognition**: High accuracy, requires internet
- **Sphinx**: Works offline, lower accuracy

**Best Practices:**
- Use clear, high-quality audio recordings
- Minimize background noise
- WAV format works best for compatibility
- Select correct language for better accuracy
- For MP3 files, ensure they contain speech (not just music)

---

## 📦 Installed Dependencies

```bash
# Text-to-Speech
pip install gtts

# Speech-to-Text
pip install SpeechRecognition

# Audio processing support
sudo apt-get install ffmpeg portaudio19-dev
```

---

## 🚀 Usage Examples

### Example 1: Convert Text to Speech
```
Input Text: "Hello, welcome to the Data Analysis platform!"
Language: English (US)
Speed: Normal
Output: audio file (MP3)
```

### Example 2: Transcribe Audio
```
Input: audio_recording.mp3 (contains speech)
Engine: Google Speech Recognition
Language: English (US)
Output: "This is the transcribed text from your audio file"
```

---

## 🔧 Troubleshooting

### TTS Issues:
- **No audio generated**: Check internet connection (gTTS requires internet)
- **Language not working**: Ensure language code is correct
- **Poor quality**: Text might be too long, try shorter segments

### STT Issues:
- **"Could not understand audio"**: Audio might be unclear or contain no speech
- **Recognition error**: Check internet connection for Google engine
- **Format not supported**: Convert audio to WAV format
- **Offline transcription**: Use Sphinx engine (lower accuracy)

---

## 🎯 Future Enhancements (Planned)

- [ ] Whisper AI integration for better STT accuracy
- [ ] Multiple voice options for TTS
- [ ] Real-time audio transcription
- [ ] Speaker diarization (identify multiple speakers)
- [ ] Timestamp generation for transcripts
- [ ] Subtitle file generation (SRT, VTT)
- [ ] Emotion detection in speech
- [ ] Translation of transcribed text

---

## 📝 Notes

1. **Internet Required**: Both Google TTS and Google STT require internet connection
2. **Audio Format**: For best results with STT, use WAV format
3. **File Size Limits**: Keep audio files under 10 minutes for optimal performance
4. **Privacy**: Audio is processed by Google's servers for transcription (online mode)
5. **Offline Mode**: Use Sphinx for offline STT (lower accuracy)

---

## 🎬 Access Your App

- **Local URL**: http://localhost:8501
- **Network URL**: http://10.100.201.114:8501

Enjoy your enhanced unstructured data analysis platform! 🚀
