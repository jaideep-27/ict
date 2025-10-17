# Quick Test Examples for TTS and STT

## Testing Text-to-Speech

### Test 1: Simple English TTS
```
Text: "Hello! This is a test of the text to speech system."
Language: English (US)
Speed: Normal
Expected: Clear English voice
```

### Test 2: Multilingual TTS
```
Text: "Bonjour! Comment allez-vous?"
Language: French
Speed: Normal
Expected: French pronunciation
```

### Test 3: Slow Speed
```
Text: "This is spoken slowly for better understanding."
Language: English (US)
Speed: Slow
Expected: Slower speech rate
```

---

## Testing Speech-to-Text

### Test Audio Files You Can Use:

1. **Record a simple message:**
   - Open your phone's voice recorder
   - Say: "Hello, this is a test recording for speech to text conversion"
   - Save as MP3 or WAV
   - Upload to the app

2. **Use system sounds:**
   ```bash
   # Linux: Record your voice
   arecord -d 5 -f cd test_audio.wav
   
   # Or use any audio file with speech
   ```

3. **Expected Results:**
   - Clear speech should transcribe with 90%+ accuracy
   - Background music may reduce accuracy
   - Accents are supported but may need language selection

---

## Sample Use Cases

### Use Case 1: Content Creation
**Scenario:** You write blog posts and want audio versions

1. Go to Text Analysis
2. Paste your blog post
3. Use Text-to-Speech tab
4. Generate MP3
5. Embed audio in your blog

### Use Case 2: Meeting Transcription
**Scenario:** You recorded a meeting and need notes

1. Go to Audio Analysis
2. Upload meeting recording (MP3/WAV)
3. Use Speech-to-Text tab
4. Get full transcript
5. Download as TXT file

### Use Case 3: Language Learning
**Scenario:** Learn pronunciation in different languages

1. Write phrases in target language
2. Use Text-to-Speech
3. Listen to correct pronunciation
4. Practice speaking
5. Record yourself
6. Use Speech-to-Text to verify

### Use Case 4: Accessibility
**Scenario:** Make content accessible to all users

**For visually impaired:**
- Convert written content to audio
- Generate audio descriptions

**For hearing impaired:**
- Transcribe audio content to text
- Create subtitles from speech

---

## Python Code Examples (if you want to use libraries directly)

### Example 1: Using gTTS in Python
```python
from gtts import gTTS

# Create speech
text = "Hello, this is a test!"
tts = gTTS(text=text, lang='en', slow=False)

# Save to file
tts.save("output.mp3")

# Multiple languages
languages = [
    ('en', 'Hello'),
    ('es', 'Hola'),
    ('fr', 'Bonjour'),
    ('de', 'Hallo'),
    ('hi', 'नमस्ते')
]

for lang, text in languages:
    tts = gTTS(text=text, lang=lang)
    tts.save(f"greeting_{lang}.mp3")
```

### Example 2: Using SpeechRecognition in Python
```python
import speech_recognition as sr

# Initialize recognizer
r = sr.Recognizer()

# From audio file
with sr.AudioFile('audio.wav') as source:
    audio = r.record(source)
    
# Transcribe
try:
    text = r.recognize_google(audio)
    print(f"Transcript: {text}")
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print(f"Error: {e}")
```

### Example 3: Audio Format Conversion (if needed)
```python
from pydub import AudioFile

# Convert MP3 to WAV
audio = AudioFile.from_mp3("input.mp3")
audio.export("output.wav", format="wav")
```

---

## CLI Commands for Quick Testing

### Generate speech from text (using gTTS CLI):
```bash
# Activate virtual environment
source streamlit_env/bin/activate

# Generate speech
gtts-cli "Hello, this is a test" --output test.mp3 --lang en

# Play the audio
ffplay test.mp3
```

### Check installed packages:
```bash
pip list | grep -E "gtts|SpeechRecognition|librosa"
```

### Test audio file compatibility:
```bash
ffprobe your_audio_file.mp3
```

---

## Supported Audio Formats

### For Speech-to-Text (STT):
- ✅ WAV (best compatibility)
- ✅ MP3 (requires FFmpeg)
- ✅ FLAC
- ✅ OGG
- ⚠️ M4A (may need conversion)

### For Text-to-Speech (TTS) Output:
- ✅ MP3 (default, best compatibility)

---

## Performance Tips

1. **For large texts (TTS):**
   - Split into chunks of 500 characters
   - Generate separately
   - Combine using audio editing tools

2. **For long audio (STT):**
   - Break into 5-minute segments
   - Transcribe each segment
   - Combine transcripts

3. **Improve STT accuracy:**
   - Use WAV format (44.1kHz, 16-bit)
   - Remove background noise
   - Normalize audio levels
   - Use noise cancellation

4. **Language detection:**
   - If unsure of language, try multiple
   - Check language codes in dropdown
   - Some languages work better than others

---

## Troubleshooting Guide

### Problem: TTS generates no audio
**Solution:**
1. Check internet connection
2. Verify text is not empty
3. Try different language
4. Check console for errors

### Problem: STT shows "Unknown value error"
**Solution:**
1. Check if audio contains speech
2. Try different language setting
3. Convert to WAV format
4. Reduce background noise

### Problem: Audio file not uploading
**Solution:**
1. Check file size (< 200MB)
2. Convert to WAV format
3. Verify file is not corrupted
4. Try different browser

### Problem: Poor transcription quality
**Solution:**
1. Use clearer audio
2. Select correct language
3. Use Google engine (not Sphinx)
4. Pre-process audio (noise reduction)

---

Happy analyzing! 🎉
