import streamlit as st
import pandas as pd
from datetime import datetime
import os
import warnings
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Suppress specific audio library warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*PySoundFile failed.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*audioread_load.*')
warnings.filterwarnings('ignore', category=FutureWarning, message='.*deprecated.*')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="DataSense — Unstructured Data Explorer",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------------------
# Custom Clean Dark Theme CSS
# -----------------------------------------
st.markdown("""
<style>
/* General layout */
.main {
    background-color: #0e1117;
    color: #e6edf3;
    font-family: "Inter", sans-serif;
    padding: 1.5rem 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(18, 21, 27, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid #30363d;
}

/* Headings */
h1, h2, h3 {
    color: #f0f6fc;
    font-weight: 600;
}
h1 {
    margin-bottom: 1rem;
}
h2, h3 {
    color: #c9d1d9;
}

/* Metrics */
div[data-testid="stMetricValue"] {
    color: #58a6ff;
    font-size: 1.7rem;
    font-weight: 700;
}
div[data-testid="stMetricLabel"] {
    color: #9da7b2;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: rgba(22, 27, 34, 0.8);
    border-radius: 8px;
    padding: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    border-radius: 6px;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
}
.stTabs [aria-selected="true"] {
    background-color: #161b22;
    color: #ffffff;
    font-weight: 600;
    border-bottom: none;
}

/* Buttons */
.stButton button, .stDownloadButton button {
    background: #1f6feb;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #388bfd;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: rgba(22,27,34,0.6);
    border-radius: 8px;
    padding: 1rem;
}

/* Text area */
textarea {
    background-color: #161b22 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* Info boxes */
.stAlert {
    background-color: rgba(56, 139, 253, 0.1);
    border: 1px solid #388bfd;
    border-radius: 6px;
    color: #e6edf3;
}

/* Cards */
.analysis-card {
    background: rgba(22,27,34,0.8);
    border-radius: 10px;
    padding: 1.5rem;
    border: 1px solid #30363d;
    margin-bottom: 1rem;
}

p, span, div, li {
    color: #e6edf3;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Header
# -----------------------------------------
st.title("🎬 DataSense — Unstructured Data Explorer")
st.caption("Analyze video, text, and audio data with advanced processing and insights.")

# -----------------------------------------
# Sidebar
# -----------------------------------------
with st.sidebar:
    st.header("📂 Data Type")
    data_type = st.radio(
        "Select analysis type:",
        ["🎥 Video Analysis", "📝 Text Analysis", "🎵 Audio Analysis", "🖼️ Image Analysis", "📖 Story Analysis"]
    )
    st.markdown("---")
    st.caption("Upload your unstructured data for comprehensive analysis.")

# -----------------------------------------
# Video Analysis Section
# -----------------------------------------
if data_type == "🎥 Video Analysis":
    st.subheader("🎥 Video Analysis")
    
    # File uploader for video
    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Supported formats: MP4, AVI, MOV, MKV, WebM"
    )
    
    if video_file:
        # Save uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        # Display video
        st.markdown("#### 🎬 Video Preview")
        st.video("temp_video.mp4")
        
        # Video metadata
        col1, col2, col3, col4 = st.columns(4)
        file_size = os.path.getsize("temp_video.mp4") / (1024 * 1024)  # MB
        col1.metric("File Size", f"{file_size:.2f} MB")
        col2.metric("Format", video_file.type.split('/')[-1].upper())
        col3.metric("File Name", video_file.name[:20] + "..." if len(video_file.name) > 20 else video_file.name)
        col4.metric("Upload Time", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("#### ⚙️ Analysis Options")
        analysis_tabs = st.tabs(["📊 Basic Info", "🎞️ Frame Analysis", "🔍 Object Detection", "📈 Advanced Metrics"])
        
        with analysis_tabs[0]:
            st.markdown("##### Basic Video Information")
            
            try:
                import cv2
                
                cap = cv2.VideoCapture("temp_video.mp4")
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                
                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.metric("Resolution", f"{width}x{height}")
                info_col2.metric("FPS", f"{fps:.2f}")
                info_col3.metric("Duration", f"{duration:.2f}s")
                
                info_col4, info_col5, info_col6 = st.columns(3)
                info_col4.metric("Total Frames", f"{frame_count:,}")
                info_col5.metric("Aspect Ratio", f"{width/height:.2f}:1")
                info_col6.metric("Bitrate", "Calculating...")
                
                # Video properties table
                st.markdown("##### 📋 Detailed Properties")
                properties_df = pd.DataFrame({
                    'Property': ['File Name', 'Format', 'Resolution', 'FPS', 'Duration', 'Total Frames', 'File Size'],
                    'Value': [
                        video_file.name,
                        video_file.type,
                        f"{width}x{height}",
                        f"{fps:.2f}",
                        f"{duration:.2f}s",
                        f"{frame_count:,}",
                        f"{file_size:.2f} MB"
                    ]
                })
                st.dataframe(properties_df, use_container_width=True, hide_index=True)
                
            except ImportError:
                st.warning("⚠️ OpenCV not installed. Install it with: `pip install opencv-python`")
                st.info("Basic file information only available.")
        
        with analysis_tabs[1]:
            st.markdown("##### 🎞️ Frame Extraction & Analysis")
            
            try:
                import cv2
                import numpy as np
                from PIL import Image
                
                cap = cv2.VideoCapture("temp_video.mp4")
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                num_frames = st.slider("Number of frames to extract", 1, min(20, frame_count), 5)
                
                if st.button("🎬 Extract Frames", key="extract_frames"):
                    with st.spinner("Extracting frames..."):
                        frames = []
                        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
                        
                        for idx in frame_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frames.append(frame_rgb)
                        
                        cap.release()
                        
                        # Display extracted frames
                        st.markdown("##### Extracted Frames")
                        cols = st.columns(min(3, num_frames))
                        for i, frame in enumerate(frames):
                            with cols[i % 3]:
                                st.image(frame, caption=f"Frame {frame_indices[i]}", use_container_width=True)
                                
                                # Frame statistics
                                avg_color = frame.mean(axis=(0, 1))
                                st.caption(f"Avg RGB: ({avg_color[0]:.0f}, {avg_color[1]:.0f}, {avg_color[2]:.0f})")
                        
                        st.success(f"✅ Successfully extracted {len(frames)} frames!")
                
            except ImportError:
                st.warning("⚠️ OpenCV and PIL not installed. Install with: `pip install opencv-python pillow`")
        
        with analysis_tabs[2]:
            st.markdown("##### 🔍 Object Detection (Placeholder)")
            st.info("🚧 This feature requires advanced ML models (YOLO, etc.). Coming soon!")
            
            st.markdown("**Planned Features:**")
            st.markdown("""
            - 🎯 Object detection and counting
            - 👤 Face detection and recognition
            - 🚗 Vehicle detection
            - 🏷️ Scene classification
            - 📊 Confidence scores and bounding boxes
            """)
            
            # Simulated object detection
            if st.button("🔮 Simulate Detection", key="simulate_detection"):
                st.markdown("##### Detected Objects (Simulated)")
                simulated_objects = pd.DataFrame({
                    'Object': ['Person', 'Car', 'Tree', 'Building', 'Person'],
                    'Confidence': ['95%', '87%', '92%', '78%', '89%'],
                    'Frame': [45, 120, 230, 340, 450],
                    'Bounding Box': ['(120,80,200,300)', '(300,150,450,280)', '(50,30,150,280)', '(400,50,600,400)', '(180,90,250,320)']
                })
                st.dataframe(simulated_objects, use_container_width=True, hide_index=True)
        
        with analysis_tabs[3]:
            st.markdown("##### 📈 Advanced Video Metrics")
            
            try:
                import cv2
                import numpy as np
                import plotly.graph_objects as go
                
                cap = cv2.VideoCapture("temp_video.mp4")
                
                sample_frames = st.slider("Sample frames for analysis", 10, 100, 30)
                
                if st.button("📊 Analyze Video Metrics", key="analyze_metrics"):
                    with st.spinner("Analyzing video..."):
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)
                        
                        brightness_values = []
                        contrast_values = []
                        
                        for idx in frame_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                            ret, frame = cap.read()
                            if ret:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                brightness = np.mean(gray)
                                contrast = np.std(gray)
                                brightness_values.append(brightness)
                                contrast_values.append(contrast)
                        
                        cap.release()
                        
                        # Plot brightness over time
                        fig_brightness = go.Figure()
                        fig_brightness.add_trace(go.Scatter(
                            x=list(range(len(brightness_values))),
                            y=brightness_values,
                            mode='lines+markers',
                            name='Brightness',
                            line=dict(color='#58a6ff', width=2)
                        ))
                        fig_brightness.update_layout(
                            title='Brightness Variation Over Time',
                            xaxis_title='Sample Frame',
                            yaxis_title='Brightness (0-255)',
                            template='plotly_dark',
                            height=350
                        )
                        st.plotly_chart(fig_brightness, use_container_width=True)
                        
                        # Plot contrast over time
                        fig_contrast = go.Figure()
                        fig_contrast.add_trace(go.Scatter(
                            x=list(range(len(contrast_values))),
                            y=contrast_values,
                            mode='lines+markers',
                            name='Contrast',
                            line=dict(color='#39d353', width=2)
                        ))
                        fig_contrast.update_layout(
                            title='Contrast Variation Over Time',
                            xaxis_title='Sample Frame',
                            yaxis_title='Contrast (Std Dev)',
                            template='plotly_dark',
                            height=350
                        )
                        st.plotly_chart(fig_contrast, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Avg Brightness", f"{np.mean(brightness_values):.2f}")
                        col2.metric("Avg Contrast", f"{np.mean(contrast_values):.2f}")
                        col3.metric("Stability Score", f"{100 - np.std(brightness_values):.1f}%")
                
            except ImportError:
                st.warning("⚠️ OpenCV not installed. Install it with: `pip install opencv-python`")
        
        # Cleanup
        st.markdown("---")
        if st.button("🗑️ Clear Video", key="clear_video"):
            if os.path.exists("temp_video.mp4"):
                os.remove("temp_video.mp4")
            st.rerun()
    
    else:
        st.info("👆 Upload a video file to begin analysis.")
        st.markdown("---")
        st.markdown("""
        **Video Analysis Features:**
        - 🎬 Video preview and playback
        - 📊 Metadata extraction (resolution, FPS, duration)
        - 🎞️ Frame extraction and analysis
        - 🔍 Object detection capabilities
        - 📈 Brightness and contrast analysis
        - 🎯 Scene detection and classification
        """)

# -----------------------------------------
# Text Analysis Section
# -----------------------------------------
elif data_type == "📝 Text Analysis":
    st.subheader("📝 Text Analysis")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["📄 Upload Text File", "✍️ Enter Text Manually"])
    
    text_content = None
    
    if input_method == "📄 Upload Text File":
        text_file = st.file_uploader(
            "Upload a text file",
            type=["txt", "md", "csv", "json", "log"],
            help="Supported formats: TXT, MD, CSV, JSON, LOG"
        )
        
        if text_file:
            text_content = text_file.read().decode('utf-8', errors='ignore')
            st.success(f"✅ Loaded {len(text_content)} characters from {text_file.name}")
    
    else:
        text_content = st.text_area(
            "Enter or paste your text here:",
            height=200,
            placeholder="Type or paste your text for analysis..."
        )
    
    if text_content:
        # Display text
        with st.expander("📖 View Full Text", expanded=False):
            st.text(text_content[:5000] + "..." if len(text_content) > 5000 else text_content)
        
        st.markdown("---")
        
        # Basic metrics
        st.markdown("#### 📊 Text Statistics")
        
        words = text_content.split()
        sentences = text_content.split('.')
        lines = text_content.split('\n')
        
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Characters", f"{len(text_content):,}")
        col2.metric("Words", f"{len(words):,}")
        col3.metric("Sentences", f"{len(sentences):,}")
        col4.metric("Lines", f"{len(lines):,}")
        col5.metric("Avg Word Length", f"{sum(len(w) for w in words) / len(words):.1f}" if words else "0")
        
        st.markdown("---")
        
        # Analysis tabs
        analysis_tabs = st.tabs(["📈 Word Analysis", "🔤 Character Analysis", "💭 Sentiment", "🏷️ Keywords", "� Text-to-Speech", "�📚 Advanced NLP"])
        
        with analysis_tabs[0]:
            st.markdown("##### 📈 Word Frequency Analysis")
            
            # Word frequency
            from collections import Counter
            import plotly.express as px
            
            # Clean and count words
            clean_words = [w.lower().strip('.,!?;:"()[]{}') for w in words if len(w) > 2]
            word_freq = Counter(clean_words)
            
            # Top words
            top_n = st.slider("Show top N words", 5, 50, 20, key="top_words")
            top_words = word_freq.most_common(top_n)
            
            if top_words:
                # Bar chart
                words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
                fig = px.bar(words_df, x='Word', y='Frequency',
                           title=f'Top {top_n} Most Frequent Words',
                           color='Frequency',
                           color_continuous_scale='Blues')
                fig.update_layout(template='plotly_dark', height=400)
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(words_df, use_container_width=True, hide_index=True)
            
            # Word length distribution
            st.markdown("##### 📏 Word Length Distribution")
            word_lengths = [len(w) for w in words]
            length_dist = Counter(word_lengths)
            
            length_df = pd.DataFrame(sorted(length_dist.items()), columns=['Length', 'Count'])
            fig_length = px.bar(length_df, x='Length', y='Count',
                              title='Distribution of Word Lengths',
                              color='Count',
                              color_continuous_scale='Teal')
            fig_length.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig_length, use_container_width=True)
        
        with analysis_tabs[1]:
            st.markdown("##### 🔤 Character Analysis")
            
            # Character frequency
            char_freq = Counter(text_content.lower())
            
            # Filter to letters only
            letters_only = {k: v for k, v in char_freq.items() if k.isalpha()}
            top_chars = sorted(letters_only.items(), key=lambda x: x[1], reverse=True)[:26]
            
            char_df = pd.DataFrame(top_chars, columns=['Character', 'Frequency'])
            
            fig_chars = px.bar(char_df, x='Character', y='Frequency',
                             title='Character Frequency Distribution',
                             color='Frequency',
                             color_continuous_scale='Viridis')
            fig_chars.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_chars, use_container_width=True)
            
            # Special characters
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Letter Statistics")
                uppercase_count = sum(1 for c in text_content if c.isupper())
                lowercase_count = sum(1 for c in text_content if c.islower())
                
                stats_df = pd.DataFrame({
                    'Type': ['Uppercase', 'Lowercase', 'Digits', 'Spaces', 'Special Chars'],
                    'Count': [
                        uppercase_count,
                        lowercase_count,
                        sum(1 for c in text_content if c.isdigit()),
                        sum(1 for c in text_content if c.isspace()),
                        sum(1 for c in text_content if not c.isalnum() and not c.isspace())
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("##### Punctuation Frequency")
                punctuation = {k: v for k, v in char_freq.items() if k in '.,!?;:\'"()-'}
                if punctuation:
                    punct_df = pd.DataFrame(sorted(punctuation.items(), key=lambda x: x[1], reverse=True),
                                          columns=['Symbol', 'Count'])
                    st.dataframe(punct_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No punctuation found")
        
        with analysis_tabs[2]:
            st.markdown("##### 💭 Sentiment Analysis")
            
            try:
                from textblob import TextBlob
                
                blob = TextBlob(text_content)
                sentiment = blob.sentiment
                
                # Sentiment metrics
                col1, col2, col3 = st.columns(3)
                
                polarity = sentiment.polarity
                subjectivity = sentiment.subjectivity
                
                # Determine sentiment
                if polarity > 0.1:
                    sentiment_label = "😊 Positive"
                    sentiment_color = "#39d353"
                elif polarity < -0.1:
                    sentiment_label = "😞 Negative"
                    sentiment_color = "#f85149"
                else:
                    sentiment_label = "😐 Neutral"
                    sentiment_color = "#58a6ff"
                
                col1.metric("Sentiment", sentiment_label)
                col2.metric("Polarity", f"{polarity:.3f}")
                col3.metric("Subjectivity", f"{subjectivity:.3f}")
                
                # Visualization
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=polarity,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Sentiment Polarity"},
                    delta={'reference': 0},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [-1, -0.3], 'color': "rgba(248, 81, 73, 0.3)"},
                            {'range': [-0.3, 0.3], 'color': "rgba(88, 166, 255, 0.3)"},
                            {'range': [0.3, 1], 'color': "rgba(57, 211, 83, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': polarity
                        }
                    }
                ))
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation
                st.markdown("##### 📖 Understanding the Metrics")
                st.markdown("""
                - **Polarity**: Ranges from -1 (most negative) to +1 (most positive)
                - **Subjectivity**: Ranges from 0 (very objective) to 1 (very subjective)
                """)
                
            except ImportError:
                st.warning("⚠️ TextBlob not installed. Install it with: `pip install textblob`")
                st.info("After installation, run: `python -m textblob.download_corpora`")
        
        with analysis_tabs[3]:
            st.markdown("##### 🏷️ Keyword Extraction")
            
            try:
                from collections import Counter
                import re
                
                # Simple keyword extraction (remove common stop words)
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                            'could', 'should', 'may', 'might', 'this', 'that', 'these', 'those',
                            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
                            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                            'own', 'same', 'so', 'than', 'too', 'very', 'can', 'just', 'now'}
                
                # Clean and filter words
                clean_words = [w.lower().strip('.,!?;:"()[]{}') for w in words 
                              if len(w) > 3 and w.lower() not in stop_words]
                
                keyword_freq = Counter(clean_words)
                top_keywords = keyword_freq.most_common(30)
                
                if top_keywords:
                    keywords_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
                    
                    # Treemap visualization
                    import plotly.express as px
                    fig = px.treemap(keywords_df.head(20), path=['Keyword'], values='Frequency',
                                   title='Top Keywords (Treemap)',
                                   color='Frequency',
                                   color_continuous_scale='Blues')
                    fig.update_layout(template='plotly_dark', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Table
                    st.dataframe(keywords_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No keywords found")
                
            except Exception as e:
                st.error(f"Error in keyword extraction: {str(e)}")
        
        with analysis_tabs[4]:
            st.markdown("##### � Text-to-Speech Conversion")
            
            st.markdown("Convert your text to speech using Google Text-to-Speech (gTTS)")
            
            # TTS options
            col1, col2 = st.columns(2)
            
            with col1:
                tts_text = st.text_area(
                    "Text to convert to speech:",
                    value=text_content[:500] if len(text_content) > 500 else text_content,
                    height=150,
                    help="Maximum 500 characters recommended for optimal performance"
                )
            
            with col2:
                st.markdown("##### Settings")
                language = st.selectbox(
                    "Language",
                    options=[
                        ("English", "en"),
                        ("Spanish", "es"),
                        ("French", "fr"),
                        ("German", "de"),
                        ("Italian", "it"),
                        ("Portuguese", "pt"),
                        ("Hindi", "hi"),
                        ("Japanese", "ja"),
                        ("Korean", "ko"),
                        ("Chinese", "zh-CN")
                    ],
                    format_func=lambda x: x[0],
                    index=0
                )
                
                slow_speed = st.checkbox("Slow speech", value=False)
                
                st.caption(f"Selected: {language[0]} ({language[1]})")
            
            if st.button("🎤 Generate Speech", key="generate_tts"):
                if tts_text.strip():
                    try:
                        from gtts import gTTS
                        import os
                        
                        with st.spinner("Generating speech..."):
                            # Create TTS
                            tts = gTTS(text=tts_text, lang=language[1], slow=slow_speed)
                            
                            # Save to file
                            tts_file = "temp_tts_output.mp3"
                            tts.save(tts_file)
                            
                            # Display audio player
                            st.success("✅ Speech generated successfully!")
                            st.audio(tts_file)
                            
                            # Download button
                            with open(tts_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download Audio",
                                    data=f,
                                    file_name="text_to_speech.mp3",
                                    mime="audio/mp3"
                                )
                            
                            # Show stats
                            file_size = os.path.getsize(tts_file) / 1024  # KB
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric("Characters", len(tts_text))
                            col_b.metric("Words", len(tts_text.split()))
                            col_c.metric("Audio Size", f"{file_size:.1f} KB")
                            
                    except ImportError:
                        st.error("⚠️ gTTS not installed. Install it with: `pip install gtts`")
                    except Exception as e:
                        st.error(f"Error generating speech: {str(e)}")
                else:
                    st.warning("⚠️ Please enter some text to convert to speech.")
            
            st.markdown("---")
            st.markdown("##### 📖 About Text-to-Speech")
            st.info("""
            **Google Text-to-Speech (gTTS)** converts text into natural-sounding speech.
            - Supports multiple languages
            - Adjustable speech speed
            - High-quality voice synthesis
            - Ideal for accessibility and content creation
            """)
        
        with analysis_tabs[5]:
            st.markdown("##### �📚 Advanced NLP Features")
            
            st.info("🚧 Advanced NLP features require additional libraries (spaCy, NLTK, transformers)")
            
            st.markdown("**Planned Features:**")
            st.markdown("""
            - 🏷️ Named Entity Recognition (NER)
            - 🌳 Part-of-Speech Tagging
            - 🔗 Dependency Parsing
            - 📊 Text Classification
            - 🌐 Language Detection
            - 📝 Text Summarization
            - 🔄 Text Translation
            """)
            
            # Simulated NER
            if st.button("🔮 Simulate NER", key="simulate_ner"):
                st.markdown("##### Named Entities (Simulated)")
                simulated_entities = pd.DataFrame({
                    'Entity': ['OpenAI', 'San Francisco', 'GPT-4', 'Python', 'USA'],
                    'Type': ['ORGANIZATION', 'LOCATION', 'PRODUCT', 'TECHNOLOGY', 'LOCATION'],
                    'Count': [3, 1, 2, 5, 1]
                })
                st.dataframe(simulated_entities, use_container_width=True, hide_index=True)
        
        # Download analysis
        st.markdown("---")
        
        analysis_report = f"""
TEXT ANALYSIS REPORT
====================

File Statistics:
- Characters: {len(text_content):,}
- Words: {len(words):,}
- Sentences: {len(sentences):,}
- Lines: {len(lines):,}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        st.download_button(
            "📥 Download Analysis Report",
            analysis_report.encode('utf-8'),
            "text_analysis_report.txt",
            "text/plain"
        )
    
    else:
        st.info("👆 Upload a file or enter text to begin analysis.")
        st.markdown("---")
        st.markdown("""
        **Text Analysis Features:**
        - 📊 Comprehensive text statistics
        - 📈 Word frequency analysis
        - 🔤 Character distribution
        - 💭 Sentiment analysis
        - 🏷️ Keyword extraction
        - 📚 Advanced NLP capabilities
        """)

# -----------------------------------------
# Audio Analysis Section
# -----------------------------------------
elif data_type == "🎵 Audio Analysis":
    st.subheader("🎵 Audio Analysis")
    
    audio_file = st.file_uploader(
        "Upload an audio file",
        type=["mp3", "wav", "ogg", "m4a", "flac"],
        help="Supported formats: MP3, WAV, OGG, M4A, FLAC"
    )
    
    if audio_file:
        # Save audio file
        with open("temp_audio.mp3", "wb") as f:
            f.write(audio_file.read())
        
        # Display audio player
        st.markdown("#### 🎧 Audio Player")
        st.audio("temp_audio.mp3")
        
        # Basic metrics
        file_size = os.path.getsize("temp_audio.mp3") / (1024 * 1024)  # MB
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("File Size", f"{file_size:.2f} MB")
        col2.metric("Format", audio_file.type.split('/')[-1].upper())
        col3.metric("File Name", audio_file.name[:20] + "..." if len(audio_file.name) > 20 else audio_file.name)
        col4.metric("Upload Time", datetime.now().strftime("%H:%M:%S"))
        
        st.markdown("---")
        
        # Analysis tabs
        analysis_tabs = st.tabs(["📊 Audio Info", "🌊 Waveform", "📈 Spectrum", "🎤 Transcription"])
        
        with analysis_tabs[0]:
            st.markdown("##### 📊 Audio Properties")
            
            try:
                import librosa
                import numpy as np
                
                # Load audio
                y, sr = librosa.load("temp_audio.mp3", sr=None)
                duration = len(y) / sr
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Duration", f"{duration:.2f}s")
                col2.metric("Sample Rate", f"{sr:,} Hz")
                col3.metric("Channels", "Mono")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Total Samples", f"{len(y):,}")
                col5.metric("Bit Depth", "16-bit (estimated)")
                col6.metric("Avg Amplitude", f"{np.mean(np.abs(y)):.4f}")
                
                # Detailed properties
                st.markdown("##### 📋 Detailed Audio Properties")
                props_df = pd.DataFrame({
                    'Property': ['File Name', 'Format', 'Duration', 'Sample Rate', 'Total Samples', 'File Size'],
                    'Value': [
                        audio_file.name,
                        audio_file.type,
                        f"{duration:.2f}s",
                        f"{sr:,} Hz",
                        f"{len(y):,}",
                        f"{file_size:.2f} MB"
                    ]
                })
                st.dataframe(props_df, use_container_width=True, hide_index=True)
                
            except ImportError:
                st.warning("⚠️ Librosa not installed. Install it with: `pip install librosa`")
                st.info("Basic file information only available.")
        
        with analysis_tabs[1]:
            st.markdown("##### 🌊 Waveform Visualization")
            
            try:
                import librosa
                import librosa.display
                import numpy as np
                import plotly.graph_objects as go
                
                # Load audio
                y, sr = librosa.load("temp_audio.mp3", sr=None, duration=60)  # Limit to 60s
                
                # Create time array
                time = np.linspace(0, len(y) / sr, len(y))
                
                # Downsample for plotting
                downsample_factor = max(1, len(y) // 5000)
                time_plot = time[::downsample_factor]
                y_plot = y[::downsample_factor]
                
                # Plot waveform
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_plot,
                    y=y_plot,
                    mode='lines',
                    name='Amplitude',
                    line=dict(color='#58a6ff', width=1)
                ))
                fig.update_layout(
                    title='Audio Waveform',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Amplitude',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Max Amplitude", f"{np.max(np.abs(y)):.4f}")
                col2.metric("Min Amplitude", f"{np.min(y):.4f}")
                col3.metric("RMS Energy", f"{np.sqrt(np.mean(y**2)):.4f}")
                col4.metric("Zero Crossings", f"{len(librosa.zero_crossings(y)):,}")
                
            except ImportError:
                st.warning("⚠️ Librosa not installed. Install it with: `pip install librosa`")
        
        with analysis_tabs[2]:
            st.markdown("##### 📈 Frequency Spectrum Analysis")
            
            try:
                import librosa
                import numpy as np
                import plotly.graph_objects as go
                
                # Load audio
                y, sr = librosa.load("temp_audio.mp3", sr=None, duration=30)
                
                # Compute spectrogram
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                
                # Plot spectrogram
                fig = go.Figure(data=go.Heatmap(
                    z=D,
                    colorscale='Viridis',
                    colorbar=dict(title='dB')
                ))
                fig.update_layout(
                    title='Spectrogram',
                    xaxis_title='Time Frame',
                    yaxis_title='Frequency Bin',
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Compute spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Avg Spectral Centroid", f"{np.mean(spectral_centroids):.0f} Hz")
                col2.metric("Avg Spectral Rolloff", f"{np.mean(spectral_rolloff):.0f} Hz")
                col3.metric("Bandwidth", f"{np.std(spectral_centroids):.0f} Hz")
                
                # Plot spectral features over time
                time_frames = np.linspace(0, len(y) / sr, len(spectral_centroids))
                
                fig_features = go.Figure()
                fig_features.add_trace(go.Scatter(
                    x=time_frames,
                    y=spectral_centroids,
                    mode='lines',
                    name='Spectral Centroid',
                    line=dict(color='#58a6ff')
                ))
                fig_features.add_trace(go.Scatter(
                    x=time_frames,
                    y=spectral_rolloff,
                    mode='lines',
                    name='Spectral Rolloff',
                    line=dict(color='#39d353')
                ))
                fig_features.update_layout(
                    title='Spectral Features Over Time',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Frequency (Hz)',
                    template='plotly_dark',
                    height=350
                )
                st.plotly_chart(fig_features, use_container_width=True)
                
            except ImportError:
                st.warning("⚠️ Librosa not installed. Install it with: `pip install librosa`")
        
        with analysis_tabs[3]:
            st.markdown("##### 🎤 Speech-to-Text Transcription")
            
            st.markdown("Convert audio speech to text using Google Speech Recognition")
            
            st.info("ℹ️ Audio files are automatically converted to WAV format for optimal recognition. Supports MP3, WAV, OGG, FLAC, and more!")
            
            # Transcription options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Transcription Method**")
                transcription_method = st.radio(
                    "Choose recognition engine:",
                    ["Google Speech Recognition (Free)", "Sphinx (Offline)"],
                    help="Google requires internet connection but provides better accuracy"
                )
            
            with col2:
                st.markdown("**Language**")
                lang_code = st.selectbox(
                    "Audio Language",
                    options=[
                        ("English (US)", "en-US"),
                        ("English (UK)", "en-GB"),
                        ("Spanish", "es-ES"),
                        ("French", "fr-FR"),
                        ("German", "de-DE"),
                        ("Hindi", "hi-IN"),
                        ("Japanese", "ja-JP"),
                        ("Chinese", "zh-CN")
                    ],
                    format_func=lambda x: x[0],
                    index=0
                )
            
            if st.button("� Transcribe Audio", key="transcribe_audio"):
                try:
                    import speech_recognition as sr
                    
                    with st.spinner("Transcribing audio... This may take a moment..."):
                        # Initialize recognizer
                        recognizer = sr.Recognizer()
                        
                        # Convert audio to WAV format if needed
                        audio_file_path = "temp_audio.mp3"
                        wav_file_path = "temp_audio_converted.wav"
                        
                        # Load audio file
                        try:
                            from pydub import AudioSegment
                            
                            st.info("🔄 Converting audio to WAV format...")
                            # Load audio file and convert to WAV
                            audio = AudioSegment.from_file(audio_file_path)
                            
                            # Export as WAV with proper settings for speech recognition
                            audio.export(
                                wav_file_path,
                                format="wav",
                                parameters=["-ar", "16000", "-ac", "1"]  # 16kHz, mono
                            )
                            
                            st.info("📖 Reading audio file...")
                            
                            with sr.AudioFile(wav_file_path) as source:
                                # Adjust for ambient noise
                                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                                audio_data = recognizer.record(source)
                                
                                st.info("🔍 Performing speech recognition...")
                                
                                # Perform recognition
                                if "Google" in transcription_method:
                                    transcript = recognizer.recognize_google(
                                        audio_data,
                                        language=lang_code[1]
                                    )
                                else:
                                    transcript = recognizer.recognize_sphinx(audio_data)
                                
                                # Display results
                                st.success("✅ Transcription completed successfully!")
                                
                                st.markdown("##### 📝 Transcription Results")
                                st.text_area(
                                    "Transcribed Text:",
                                    transcript,
                                    height=200,
                                    help="Copy this text for further use"
                                )
                                
                                # Analysis
                                words = transcript.split()
                                col_a, col_b, col_c, col_d = st.columns(4)
                                col_a.metric("Words", len(words))
                                col_b.metric("Characters", len(transcript))
                                col_c.metric("Sentences", transcript.count('.') + transcript.count('!') + transcript.count('?'))
                                col_d.metric("Language", lang_code[0])
                                
                                # Word frequency
                                st.markdown("##### 🏷️ Top Keywords")
                                from collections import Counter
                                word_freq = Counter([w.lower() for w in words if len(w) > 3])
                                top_words = word_freq.most_common(10)
                                
                                if top_words:
                                    keywords_df = pd.DataFrame(top_words, columns=['Keyword', 'Frequency'])
                                    st.dataframe(keywords_df, use_container_width=True, hide_index=True)
                                
                                # Download transcript
                                st.download_button(
                                    label="📥 Download Transcript",
                                    data=transcript,
                                    file_name="audio_transcript.txt",
                                    mime="text/plain"
                                )
                                
                                # Cleanup converted WAV file
                                if os.path.exists(wav_file_path):
                                    os.remove(wav_file_path)
                                
                        except sr.UnknownValueError:
                            st.error("❌ Could not understand the audio. Please ensure the audio contains clear speech.")
                            st.info("💡 Tips: Use clear speech, reduce background noise, and ensure audio quality is good.")
                        except sr.RequestError as e:
                            st.error(f"❌ Could not request results from recognition service: {e}")
                            st.info("💡 Check your internet connection (required for Google Speech Recognition).")
                        except Exception as e:
                            st.error(f"❌ Error processing audio file: {str(e)}")
                            st.info("💡 The audio has been automatically converted to WAV format for processing.")
                        finally:
                            # Cleanup converted file if it exists
                            if os.path.exists(wav_file_path):
                                try:
                                    os.remove(wav_file_path)
                                except:
                                    pass
                
                except ImportError:
                    st.warning("⚠️ SpeechRecognition not installed. Install it with: `pip install SpeechRecognition`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            st.markdown("---")
            st.markdown("##### 📖 About Speech Recognition")
            st.info("""
            **Speech-to-Text** converts spoken words in audio to written text.
            
            **Available Engines:**
            - **Google Speech Recognition**: High accuracy, requires internet
            - **Sphinx**: Works offline, lower accuracy
            
            **Best Practices:**
            - Use clear, high-quality audio recordings
            - Minimize background noise
            - Use WAV format for best compatibility
            - Select the correct language for better accuracy
            """)
        
        # Cleanup
        st.markdown("---")
        if st.button("🗑️ Clear Audio", key="clear_audio"):
            if os.path.exists("temp_audio.mp3"):
                os.remove("temp_audio.mp3")
            st.rerun()
    
    else:
        st.info("👆 Upload an audio file to begin analysis.")
        st.markdown("---")
        st.markdown("""
        **Audio Analysis Features:**
        - 🎧 Audio playback
        - 📊 Metadata extraction
        - 🌊 Waveform visualization
        - 📈 Spectrum analysis
        - 🎤 Speech-to-text transcription
        - 🎵 Music feature extraction
        - 🔊 Noise reduction
        """)

# -----------------------------------------
# Image Analysis Section
# -----------------------------------------
elif data_type == "🖼️ Image Analysis":
    st.subheader("🖼️ Image Analysis")
    
    # File uploader for images
    image_file = st.file_uploader(
        "Upload an image file",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"],
        help="Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF"
    )
    
    if image_file is not None:
        # Save uploaded image temporarily
        image_path = f"temp_image_{image_file.name}"
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Load image
        from PIL import Image
        import cv2
        from rembg import remove
        from skimage import filters, feature, exposure
        from skimage.color import rgb2gray
        import matplotlib.pyplot as plt
        
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # Display original image
        st.image(image, caption="Original Image", use_container_width=True)
        
        # Image metadata
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Width", f"{image.width}px")
        with col2:
            st.metric("Height", f"{image.height}px")
        with col3:
            st.metric("Format", image.format)
        with col4:
            file_size_kb = len(image_file.getvalue()) / 1024
            st.metric("Size", f"{file_size_kb:.2f} KB")
        
        st.markdown("---")
        
        # Analysis tabs
        tabs = st.tabs([
            "🎨 Color Analysis",
            "✂️ Background Removal",
            "👤 Face Analysis",
            "🔍 Edge Detection",
            "🎭 Filters & Effects",
            "📊 Advanced Analysis"
        ])
        
        # Tab 1: Color Analysis
        with tabs[0]:
            st.subheader("🎨 Color Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Color Space Analysis**")
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                else:
                    rgb_image = image
                
                # Calculate color histograms
                fig, axes = plt.subplots(3, 1, figsize=(10, 8))
                colors = ['red', 'green', 'blue']
                for i, color in enumerate(colors):
                    axes[i].hist(np.array(rgb_image)[:,:,i].ravel(), bins=256, color=color, alpha=0.7)
                    axes[i].set_title(f'{color.upper()} Channel Histogram')
                    axes[i].set_xlim([0, 256])
                    axes[i].set_facecolor('#161b22')
                    axes[i].tick_params(colors='white')
                    for spine in axes[i].spines.values():
                        spine.set_edgecolor('white')
                
                fig.patch.set_facecolor('#0d1117')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**Dominant Colors**")
                # Get dominant colors using k-means
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if len(img_array.shape) == 3 else img_array
                pixels = img_rgb.reshape(-1, 3)
                
                # Sample pixels for faster processing
                sample_size = min(10000, len(pixels))
                sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]
                
                from sklearn.cluster import KMeans
                n_colors = 5
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(sampled_pixels)
                
                # Get dominant colors
                colors = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_
                counts = np.bincount(labels)
                
                # Sort by frequency
                indices = np.argsort(-counts)
                colors = colors[indices]
                counts = counts[indices]
                
                # Display color palette
                for i, (color, count) in enumerate(zip(colors, counts)):
                    percentage = (count / len(labels)) * 100
                    st.markdown(f"**Color {i+1}:** {percentage:.1f}%")
                    color_box = np.zeros((50, 200, 3), dtype=np.uint8)
                    color_box[:] = color
                    st.image(color_box)
                    st.caption(f"RGB({color[0]}, {color[1]}, {color[2]})")
            
            # Brightness & Contrast Analysis
            st.markdown("---")
            st.markdown("**Brightness & Contrast Metrics**")
            
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
            brightness = np.mean(gray_image)
            contrast = np.std(gray_image)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Brightness", f"{brightness:.2f}")
            with col2:
                st.metric("Contrast (Std)", f"{contrast:.2f}")
            with col3:
                st.metric("Min Intensity", f"{np.min(gray_image)}")
            with col4:
                st.metric("Max Intensity", f"{np.max(gray_image)}")
        
        # Tab 2: Background Removal (using rembg)
        with tabs[1]:
            st.subheader("✂️ AI-Powered Background Removal")
            st.markdown("Using **rembg** with deep learning model for accurate background removal")
            
            if st.button("🚀 Remove Background", key="remove_bg"):
                with st.spinner("Processing with AI model..."):
                    try:
                        # Remove background using rembg
                        with open(image_path, 'rb') as input_file:
                            input_data = input_file.read()
                        
                        output_data = remove(input_data)
                        
                        # Save result
                        output_path = f"temp_nobg_{image_file.name.split('.')[0]}.png"
                        with open(output_path, 'wb') as output_file:
                            output_file.write(output_data)
                        
                        # Display result
                        st.success("✅ Background removed successfully!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original**")
                            st.image(image, use_container_width=True)
                        with col2:
                            st.markdown("**Background Removed**")
                            result_image = Image.open(output_path)
                            st.image(result_image, use_container_width=True)
                        
                        # Download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Result",
                                data=file,
                                file_name=f"nobg_{image_file.name.split('.')[0]}.png",
                                mime="image/png"
                            )
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"❌ Error during background removal: {error_msg}")
                        
                        # Check if it's a network/download error
                        if "HTTPSConnectionPool" in error_msg or "Failed to resolve" in error_msg or "NameResolutionError" in error_msg:
                            st.warning("⚠️ **Network Error**: Cannot download the U^2-Net model.")
                            st.info("""
                            **Possible Solutions:**
                            
                            1. **Check Internet Connection**: Ensure you're connected to the internet
                            2. **Wait and Retry**: GitHub servers might be temporarily unavailable
                            3. **Manual Download** (if network issues persist):
                               ```bash
                               # Create model directory
                               mkdir -p ~/.u2net
                               
                               # Download model manually (176 MB)
                               wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -O ~/.u2net/u2net.onnx
                               ```
                            4. **Use VPN/Proxy**: If GitHub is blocked in your network
                            5. **Try Later**: The model downloads automatically on first use
                            """)
                        else:
                            st.info("💡 Try with a different image or check the error details above.")
            
            st.markdown("---")
            st.info("💡 **Tip:** The AI model works best with clear subjects and good lighting.")
            
            # Model status check
            import os.path
            model_path = os.path.expanduser("~/.u2net/u2net.onnx")
            if os.path.exists(model_path):
                st.success(f"✅ U^2-Net model is installed (~176 MB)")
            else:
                st.warning("⚠️ U^2-Net model not yet downloaded. Will download on first use (~176 MB).")
        
        # Tab 3: Face Analysis (using DeepFace)
        with tabs[2]:
            st.subheader("👤 AI Face Analysis")
            st.markdown("Using **DeepFace** for advanced facial recognition and analysis")
            
            if st.button("🔍 Analyze Faces", key="analyze_faces"):
                with st.spinner("Detecting and analyzing faces..."):
                    try:
                        # Import DeepFace only when needed (lazy loading to avoid startup conflicts)
                        try:
                            from deepface import DeepFace
                        except Exception as import_error:
                            st.error(f"❌ DeepFace library error: {str(import_error)}")
                            st.info("This may be a TensorFlow/Keras version conflict. Face analysis requires compatible versions.")
                            raise
                        
                        # Analyze faces
                        analysis = DeepFace.analyze(
                            img_path=image_path,
                            actions=['age', 'gender', 'race', 'emotion'],
                            enforce_detection=False
                        )
                        
                        if isinstance(analysis, list):
                            analysis = analysis[0]
                        
                        st.success(f"✅ Detected {1 if analysis else 0} face(s)")
                        
                        if analysis:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Demographics**")
                                st.metric("Estimated Age", f"{analysis['age']} years")
                                st.metric("Gender", analysis['dominant_gender'].capitalize())
                                st.metric("Dominant Race", analysis['dominant_race'].capitalize())
                            
                            with col2:
                                st.markdown("**Emotion Analysis**")
                                emotions = analysis['emotion']
                                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                                
                                for emotion, score in sorted_emotions[:3]:
                                    st.metric(emotion.capitalize(), f"{score:.1f}%")
                            
                            # Emotion chart
                            st.markdown("---")
                            st.markdown("**Emotion Distribution**")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            emotions_sorted = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))
                            ax.barh(list(emotions_sorted.keys()), list(emotions_sorted.values()), color='#388bfd')
                            ax.set_xlabel('Confidence (%)')
                            ax.set_facecolor('#161b22')
                            fig.patch.set_facecolor('#0d1117')
                            ax.tick_params(colors='white')
                            for spine in ax.spines.values():
                                spine.set_edgecolor('white')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        
                    except Exception as e:
                        st.warning(f"⚠️ No faces detected or error: {str(e)}")
                        st.info("Try uploading an image with a clear, frontal face.")
        
        # Tab 4: Edge Detection
        with tabs[3]:
            st.subheader("🔍 Edge Detection")
            
            edge_method = st.selectbox(
                "Select edge detection method:",
                ["Canny", "Sobel", "Prewitt", "Roberts", "Scharr"]
            )
            
            if st.button("🎯 Detect Edges", key="detect_edges"):
                with st.spinner(f"Applying {edge_method} edge detection..."):
                    # Ensure grayscale is a float image in [0,1]
                    if len(img_array.shape) == 3:
                        gray = rgb2gray(img_array)
                    else:
                        # If already single channel, convert to float in [0,1]
                        gray = img_array.astype('float32')
                        if gray.max() > 1.0:
                            gray = gray / 255.0

                    # Apply chosen edge detector
                    if edge_method == "Canny":
                        # skimage.feature.canny expects a float image in [0,1]
                        edges_bool = feature.canny(gray, sigma=2)
                        # If skimage produced no edges (sometimes due to scaling), try OpenCV Canny as a fallback
                        if edges_bool.sum() == 0:
                            try:
                                # Convert float gray [0,1] to uint8 0-255
                                gray_u8 = (np.clip(gray, 0, 1) * 255).astype('uint8')
                                # Use median to choose thresholds
                                v = np.median(gray_u8)
                                lower = int(max(0, 0.66 * v))
                                upper = int(min(255, 1.33 * v))
                                edges_cv = cv2.Canny(gray_u8, lower, upper)
                                edges = edges_cv
                                # Inform user that OpenCV fallback was used
                                st.info(f"⚠️ skimage Canny returned no edges; using OpenCV Canny fallback (lower={lower}, upper={upper})")
                            except Exception:
                                # Fallback to skimage boolean->uint8 even if empty
                                edges = (edges_bool.astype('uint8') * 255)
                        else:
                            # Convert boolean array to uint8 image for display
                            edges = (edges_bool.astype('uint8') * 255)
                    elif edge_method == "Sobel":
                        edges_f = filters.sobel(gray)
                        edges = (np.clip(edges_f, 0, 1) * 255).astype('uint8')
                    elif edge_method == "Prewitt":
                        edges_f = filters.prewitt(gray)
                        edges = (np.clip(edges_f, 0, 1) * 255).astype('uint8')
                    elif edge_method == "Roberts":
                        edges_f = filters.roberts(gray)
                        edges = (np.clip(edges_f, 0, 1) * 255).astype('uint8')
                    elif edge_method == "Scharr":
                        edges_f = filters.scharr(gray)
                        edges = (np.clip(edges_f, 0, 1) * 255).astype('uint8')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original (Grayscale)**")
                        # Show normalized grayscale (convert back to 0-255 for display)
                        disp_gray = (np.clip(gray, 0, 1) * 255).astype('uint8')
                        st.image(disp_gray, use_container_width=True, clamp=True)
                    with col2:
                        st.markdown(f"**{edge_method} Edges**")
                        st.image(edges, use_container_width=True, clamp=True)
                    
                    # Edge statistics
                    st.markdown("---")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Edge Pixels", f"{np.sum(edges > 0.1)}")
                    with col2:
                        edge_percentage = (np.sum(edges > 0.1) / edges.size) * 100
                        st.metric("Edge Coverage", f"{edge_percentage:.2f}%")
                    with col3:
                        st.metric("Max Edge Strength", f"{np.max(edges):.3f}")
        
        # Tab 5: Filters & Effects
        with tabs[4]:
            st.subheader("🎭 Filters & Effects")
            
            effect = st.selectbox(
                "Select effect:",
                ["Blur", "Sharpen", "Emboss", "Negative", "Sepia", "Grayscale", "Equalize Histogram", "Contrast Stretch"]
            )
            
            if st.button("✨ Apply Effect", key="apply_effect"):
                with st.spinner(f"Applying {effect}..."):
                    result = None
                    
                    if effect == "Blur":
                        from scipy.ndimage import gaussian_filter
                        result = gaussian_filter(img_array, sigma=3)
                    
                    elif effect == "Sharpen":
                        kernel = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
                        result = cv2.filter2D(img_array, -1, kernel)
                    
                    elif effect == "Emboss":
                        kernel = np.array([[-2,-1, 0],
                                         [-1, 1, 1],
                                         [ 0, 1, 2]])
                        result = cv2.filter2D(img_array, -1, kernel)
                    
                    elif effect == "Negative":
                        result = 255 - img_array
                    
                    elif effect == "Sepia":
                        sepia_filter = np.array([[0.393, 0.769, 0.189],
                                               [0.349, 0.686, 0.168],
                                               [0.272, 0.534, 0.131]])
                        result = cv2.transform(img_array, sepia_filter)
                        result = np.clip(result, 0, 255).astype(np.uint8)
                    
                    elif effect == "Grayscale":
                        result = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                    
                    elif effect == "Equalize Histogram":
                        if len(img_array.shape) == 3:
                            result = exposure.equalize_hist(img_array)
                        else:
                            result = exposure.equalize_hist(img_array)
                    
                    elif effect == "Contrast Stretch":
                        p2, p98 = np.percentile(img_array, (2, 98))
                        result = exposure.rescale_intensity(img_array, in_range=(p2, p98))
                    
                    if result is not None:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Original**")
                            st.image(image, use_container_width=True)
                        with col2:
                            st.markdown(f"**{effect} Applied**")
                            st.image(result, use_container_width=True, clamp=True)
                        
                        # Save and download
                        result_path = f"temp_filtered_{image_file.name}"
                        if len(result.shape) == 2:
                            result_img = Image.fromarray(result.astype(np.uint8))
                        else:
                            result_img = Image.fromarray(result.astype(np.uint8))
                        result_img.save(result_path)
                        
                        with open(result_path, "rb") as file:
                            st.download_button(
                                label="📥 Download Filtered Image",
                                data=file,
                                file_name=f"filtered_{image_file.name}",
                                mime=f"image/{image_file.name.split('.')[-1]}"
                            )
        
        # Tab 6: Advanced Analysis
        with tabs[5]:
            st.subheader("📊 Advanced Image Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Image Quality Metrics**")
                
                # Sharpness (Laplacian variance)
                gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                st.metric("Sharpness Score", f"{laplacian_var:.2f}")
                
                # Entropy (information content)
                from skimage.measure import shannon_entropy
                entropy = shannon_entropy(gray)
                st.metric("Entropy", f"{entropy:.3f}")
                
                # Signal-to-Noise Ratio
                signal = np.mean(gray)
                noise = np.std(gray)
                snr = signal / noise if noise > 0 else 0
                st.metric("SNR", f"{snr:.2f}")
            
            with col2:
                st.markdown("**Image Statistics**")
                
                # Color mode
                st.text(f"Color Mode: {image.mode}")
                
                # Aspect ratio
                aspect_ratio = image.width / image.height
                st.text(f"Aspect Ratio: {aspect_ratio:.3f}")
                
                # Total pixels
                total_pixels = image.width * image.height
                st.text(f"Total Pixels: {total_pixels:,}")
                
                # Unique colors
                if image.mode == 'RGB':
                    unique_colors = len(set(tuple(pixel) for pixel in pixels[:10000]))
                    st.text(f"Unique Colors (sample): {unique_colors}")
            
            st.markdown("---")
            st.markdown("**Intensity Distribution**")
            
            # Plot intensity histogram
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(gray.ravel(), bins=256, color='#388bfd', alpha=0.7)
            ax.set_xlabel('Pixel Intensity')
            ax.set_ylabel('Frequency')
            ax.set_title('Intensity Distribution')
            ax.set_facecolor('#161b22')
            fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        if st.button("🗑️ Clear Image", key="clear_image"):
            if os.path.exists(image_path):
                os.remove(image_path)
            # Clean up temp files
            for file in os.listdir('.'):
                if file.startswith('temp_nobg_') or file.startswith('temp_filtered_'):
                    os.remove(file)
            st.rerun()
    
    else:
        st.info("👆 Upload an image file to begin analysis.")
        st.markdown("---")
        st.markdown("""
        **Image Analysis Features:**
        - 🎨 Color analysis and dominant colors
        - ✂️ AI-powered background removal (rembg)
        - 👤 Face detection and emotion analysis (DeepFace)
        - 🔍 Edge detection (multiple algorithms)
        - 🎭 Image filters and effects
        - 📊 Advanced quality metrics
        - 📥 Download processed images
        """)

# -----------------------------------------
# Story Analysis Section
# -----------------------------------------
elif data_type == "📖 Story Analysis":
    st.subheader("📖 Story Analysis with NLP")
    
    # Hardcoded sample stories
    SAMPLE_STORIES = {
        "The Lost Key": """
        Emma woke up to the sound of rain pattering against her window. She had an important meeting at 9 AM, 
        but something felt off. Reaching for her bag, she realized with horror that her office key was missing. 
        Panic set in as she frantically searched every pocket, every drawer, every corner of her apartment. 
        
        After thirty minutes of desperate searching, she decided to retrace her steps from yesterday. 
        The coffee shop! She had stopped there after work. With renewed hope, Emma grabbed her coat and 
        rushed out into the rain. The barista smiled when she walked in, holding up a small silver key. 
        "Looking for this?" he asked. Emma's relief was overwhelming. Sometimes, the things we lose find 
        their way back to us when we least expect it.
        """,
        
        "The Garden": """
        Old Mr. Chen had tended his garden for forty years. Every morning at dawn, he would walk among 
        the roses, tomatoes, and herbs, speaking to them like old friends. His neighbors thought him eccentric, 
        but they couldn't deny the magic of his garden. The flowers bloomed brighter, the vegetables grew 
        larger, and the air always smelled of jasmine and mint.
        
        When asked about his secret, Mr. Chen would simply smile and say, "Love and patience. The garden 
        teaches you both." One spring, a young girl moved in next door. She was shy and lonely, struggling 
        to make friends in the new neighborhood. Mr. Chen invited her to help in the garden. Day by day, 
        as they planted seeds and pulled weeds together, she began to bloom just like the flowers. 
        The garden had worked its magic once again.
        """,
        
        "Digital Disconnect": """
        Sarah realized she hadn't looked at her phone in three days. It started as an accident – her 
        charger broke during a weekend camping trip. At first, the anxiety was unbearable. What if someone 
        needed her? What if she missed an important email? But as the hours passed, something unexpected 
        happened: she felt lighter.
        
        Without the constant ping of notifications, she noticed things she'd been missing. The way morning 
        light filtered through pine trees. The sound of the river, which wasn't a white noise app but actual 
        water flowing over rocks. Real conversations with friends around a campfire, without anyone checking 
        their screens. When she returned home and finally charged her phone, the 47 notifications seemed 
        trivial. She had discovered something more valuable than connectivity: presence.
        """,
        
        "The Last Letter": """
        When Margaret's grandmother passed away, she left behind a box of letters. Hundreds of them, 
        spanning seventy years, written in elegant cursive on yellowing paper. Margaret spent weeks reading 
        them, discovering a woman she never knew existed. Her grandmother had been an artist, a dreamer, 
        someone who wrote poetry in the margins of recipes and found beauty in ordinary moments.
        
        One letter stood out, dated the day Margaret was born. "Today I held my granddaughter for the 
        first time," it read. "She has my mother's eyes and my stubborn chin. I wonder what kind of woman 
        she'll become. Will she chase her dreams like I did, or will she be wiser and more cautious? 
        Either way, I hope she knows she's loved beyond measure." Margaret cried, holding the letter to 
        her chest. Her grandmother had seen her, really seen her, even as a baby. The letters weren't 
        just memories – they were a legacy of love.
        """,
        
        "The Coffee Shop Writer": """
        Every Tuesday at 3 PM, a man in a gray coat would sit at the corner table of the coffee shop, 
        typing away on an old laptop. The regulars called him "The Writer," though no one knew what he 
        wrote or if he'd ever been published. He always ordered the same thing: black coffee and a blueberry 
        muffin. He never spoke to anyone, completely absorbed in his work.
        
        One day, a curious barista couldn't help but ask, "What are you writing?" The man looked up, 
        surprised to be addressed after years of solitary visits. "Love letters," he said quietly. 
        "To my wife who passed away five years ago. I write her one every week, telling her about my life, 
        my thoughts, the things I wish I could still share with her." The barista's eyes welled with tears. 
        "That's beautiful," she whispered. The man smiled sadly. "It keeps her alive in my heart." 
        From that day on, his coffee and muffin were always on the house.
        """
    }
    
    # Story selection
    st.markdown("### Choose a Story to Analyze")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        story_choice = st.selectbox(
            "Select a sample story or enter your own:",
            ["Custom Story"] + list(SAMPLE_STORIES.keys())
        )
    
    with col2:
        st.metric("Sample Stories", len(SAMPLE_STORIES))
    
    # Text input area
    if story_choice == "Custom Story":
        story_text = st.text_area(
            "Enter your story here:",
            height=300,
            placeholder="Type or paste your story here for analysis..."
        )
    else:
        story_text = st.text_area(
            f"Story: {story_choice}",
            value=SAMPLE_STORIES[story_choice],
            height=300
        )
    
    if story_text and len(story_text.strip()) > 0:
        # Download NLTK data if needed
        try:
            import nltk
            from wordcloud import WordCloud
            from textblob import TextBlob
            import matplotlib.pyplot as plt
            from collections import Counter
            import re
            
            # Download required NLTK data with fallback
            with st.spinner("Checking NLTK data..."):
                try:
                    # Try punkt_tab first (newer NLTK), fallback to punkt (older NLTK)
                    try:
                        nltk.data.find('tokenizers/punkt_tab')
                    except LookupError:
                        try:
                            nltk.download('punkt_tab', quiet=True)
                        except:
                            try:
                                nltk.download('punkt', quiet=True)
                            except:
                                st.warning("Could not download punkt tokenizer. Some features may not work.")
                    
                    # Download other required data with error handling
                    required_data = [
                        'averaged_perceptron_tagger',
                        'maxent_ne_chunker',
                        'words',
                        'stopwords'
                    ]
                    
                    for package in required_data:
                        try:
                            nltk.download(package, quiet=True)
                        except:
                            pass  # Continue even if some packages fail
                            
                except Exception as e:
                    st.warning(f"Some NLTK data could not be downloaded. Analysis may be limited.")
            
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize, sent_tokenize
            from nltk import pos_tag, ne_chunk
            from nltk.tree import Tree
            
        except Exception as e:
            st.error(f"Error loading NLP libraries: {str(e)}")
            st.stop()
        
        # Basic metrics
        words = word_tokenize(story_text.lower())
        sentences = sent_tokenize(story_text)
        word_count = len([w for w in words if w.isalnum()])
        sentence_count = len(sentences)
        char_count = len(story_text)
        avg_word_length = sum(len(w) for w in words if w.isalnum()) / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Display basic metrics
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Words", f"{word_count:,}")
        with col2:
            st.metric("Sentences", sentence_count)
        with col3:
            st.metric("Characters", f"{char_count:,}")
        with col4:
            st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        with col5:
            st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f}")
        
        st.markdown("---")
        
        # Analysis tabs
        tabs = st.tabs([
            "📊 Overview",
            "☁️ Word Cloud",
            "💭 Sentiment",
            "🏷️ Named Entities",
            "📖 Readability",
            "🔑 Keywords"
        ])
        
        # Tab 1: Overview
        with tabs[0]:
            st.subheader("📊 Story Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Text Statistics**")
                
                # Unique words
                unique_words = len(set([w.lower() for w in words if w.isalnum()]))
                lexical_diversity = unique_words / max(word_count, 1)
                
                st.write(f"**Unique Words:** {unique_words:,}")
                st.write(f"**Lexical Diversity:** {lexical_diversity:.2%}")
                st.write(f"**Longest Word:** {max((w for w in words if w.isalnum()), key=len, default='')}")
                st.write(f"**Shortest Sentence:** {min(sentences, key=len, default='')[:50]}...")
                st.write(f"**Longest Sentence:** {max(sentences, key=len, default='')[:50]}...")
            
            with col2:
                st.markdown("**Word Frequency (Top 10)**")
                
                # Remove stopwords
                stop_words = set(stopwords.words('english'))
                filtered_words = [w.lower() for w in words if w.isalnum() and w.lower() not in stop_words and len(w) > 2]
                word_freq = Counter(filtered_words).most_common(10)
                
                for word, count in word_freq:
                    st.write(f"**{word}:** {count} times")
            
            # Sentence length distribution
            st.markdown("---")
            st.markdown("**Sentence Length Distribution**")
            sentence_lengths = [len(word_tokenize(s)) for s in sentences]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(sentence_lengths, bins=min(20, len(set(sentence_lengths))), color='#388bfd', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Words per Sentence')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Sentence Lengths')
            ax.set_facecolor('#161b22')
            fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Tab 2: Word Cloud
        with tabs[1]:
            st.subheader("☁️ Word Cloud Visualization")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("**Settings**")
                max_words = st.slider("Max Words", 50, 500, 200)
                background_color = st.selectbox("Background", ["black", "white"])
                colormap = st.selectbox("Color Scheme", ["viridis", "plasma", "inferno", "magma", "cool", "hot"])
            
            with col1:
                # Generate word cloud
                stop_words = set(stopwords.words('english'))
                
                wordcloud = WordCloud(
                    width=800,
                    height=400,
                    background_color=background_color,
                    stopwords=stop_words,
                    max_words=max_words,
                    colormap=colormap,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(story_text)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                fig.patch.set_facecolor('#0d1117')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                plt.close()
                
                # Download word cloud
                wordcloud_path = "temp_wordcloud.png"
                wordcloud.to_file(wordcloud_path)
                with open(wordcloud_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Word Cloud",
                        data=file,
                        file_name="story_wordcloud.png",
                        mime="image/png"
                    )
        
        # Tab 3: Sentiment Analysis
        with tabs[2]:
            st.subheader("💭 Sentiment Analysis")
            
            blob = TextBlob(story_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Polarity", f"{polarity:.3f}")
                st.caption("Range: -1 (negative) to +1 (positive)")
                
                if polarity > 0.5:
                    st.success("😊 Very Positive")
                elif polarity > 0.1:
                    st.info("🙂 Positive")
                elif polarity > -0.1:
                    st.warning("😐 Neutral")
                elif polarity > -0.5:
                    st.warning("🙁 Negative")
                else:
                    st.error("😢 Very Negative")
            
            with col2:
                st.metric("Subjectivity", f"{subjectivity:.3f}")
                st.caption("Range: 0 (objective) to 1 (subjective)")
                
                if subjectivity > 0.6:
                    st.info("📝 Highly Subjective (Personal opinions)")
                elif subjectivity > 0.3:
                    st.info("📰 Mixed (Some opinions)")
                else:
                    st.info("📊 Objective (Factual)")
            
            with col3:
                # Overall sentiment
                if polarity >= 0:
                    sentiment_emoji = "😊" if polarity > 0.3 else "🙂"
                    sentiment_text = "Positive" if polarity > 0.3 else "Slightly Positive"
                else:
                    sentiment_emoji = "😢" if polarity < -0.3 else "🙁"
                    sentiment_text = "Negative" if polarity < -0.3 else "Slightly Negative"
                
                st.metric("Overall Sentiment", sentiment_text)
                st.markdown(f"### {sentiment_emoji}")
            
            # Sentiment by sentence
            st.markdown("---")
            st.markdown("**Sentiment Timeline (by sentence)**")
            
            sentence_polarities = [TextBlob(sent).sentiment.polarity for sent in sentences]
            
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(1, len(sentence_polarities) + 1)
            colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in sentence_polarities]
            ax.bar(x, sentence_polarities, color=colors, alpha=0.7)
            ax.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Sentence Number')
            ax.set_ylabel('Polarity')
            ax.set_title('Sentiment Flow Across Story')
            ax.set_facecolor('#161b22')
            fig.patch.set_facecolor('#0d1117')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_edgecolor('white')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Tab 4: Named Entity Recognition
        with tabs[3]:
            st.subheader("🏷️ Named Entity Recognition")
            
            # Tokenize and tag
            tokens = word_tokenize(story_text)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)
            
            # Extract entities
            named_entities = []
            for chunk in entities:
                if isinstance(chunk, Tree):
                    entity_text = " ".join([token for token, pos in chunk.leaves()])
                    entity_type = chunk.label()
                    named_entities.append((entity_text, entity_type))
            
            if named_entities:
                # Group by type
                entity_types = {}
                for entity, etype in named_entities:
                    if etype not in entity_types:
                        entity_types[etype] = []
                    entity_types[etype].append(entity)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Entities Found**")
                    st.metric("Total Entities", len(named_entities))
                    st.metric("Entity Types", len(entity_types))
                    
                    # Show entity type breakdown
                    for etype, entities_list in sorted(entity_types.items()):
                        st.write(f"**{etype}:** {len(entities_list)}")
                
                with col2:
                    st.markdown("**Entity Details**")
                    
                    for etype, entities_list in sorted(entity_types.items()):
                        with st.expander(f"{etype} ({len(entities_list)})"):
                            unique_entities = list(set(entities_list))
                            for entity in sorted(unique_entities):
                                count = entities_list.count(entity)
                                st.write(f"• {entity} ({count}x)")
                
                # Visualization
                st.markdown("---")
                st.markdown("**Entity Distribution**")
                
                fig, ax = plt.subplots(figsize=(10, 5))
                entity_counts = {etype: len(elist) for etype, elist in entity_types.items()}
                sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
                
                ax.barh([e[0] for e in sorted_entities], [e[1] for e in sorted_entities], color='#388bfd')
                ax.set_xlabel('Count')
                ax.set_ylabel('Entity Type')
                ax.set_title('Named Entities by Type')
                ax.set_facecolor('#161b22')
                fig.patch.set_facecolor('#0d1117')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("No named entities detected in this story.")
        
        # Tab 5: Readability
        with tabs[4]:
            st.subheader("📖 Readability Metrics")
            
            # Calculate readability scores
            def flesch_reading_ease(text):
                words = len([w for w in word_tokenize(text.lower()) if w.isalnum()])
                sentences = len(sent_tokenize(text))
                syllables = sum([max(1, len([c for c in w if c in 'aeiouAEIOU'])) for w in word_tokenize(text) if w.isalnum()])
                
                if words == 0 or sentences == 0:
                    return 0
                
                score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
                return max(0, min(100, score))
            
            def flesch_kincaid_grade(text):
                words = len([w for w in word_tokenize(text.lower()) if w.isalnum()])
                sentences = len(sent_tokenize(text))
                syllables = sum([max(1, len([c for c in w if c in 'aeiouAEIOU'])) for w in word_tokenize(text) if w.isalnum()])
                
                if words == 0 or sentences == 0:
                    return 0
                
                score = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
                return max(0, score)
            
            fre_score = flesch_reading_ease(story_text)
            fk_grade = flesch_kincaid_grade(story_text)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Flesch Reading Ease", f"{fre_score:.1f}")
                
                if fre_score >= 90:
                    st.success("Very Easy (5th grade)")
                elif fre_score >= 80:
                    st.info("Easy (6th grade)")
                elif fre_score >= 70:
                    st.info("Fairly Easy (7th grade)")
                elif fre_score >= 60:
                    st.info("Standard (8-9th grade)")
                elif fre_score >= 50:
                    st.warning("Fairly Difficult (10-12th grade)")
                elif fre_score >= 30:
                    st.warning("Difficult (College)")
                else:
                    st.error("Very Difficult (Graduate)")
            
            with col2:
                st.metric("Flesch-Kincaid Grade", f"{fk_grade:.1f}")
                st.caption(f"Suitable for grade {int(fk_grade)} and above")
            
            with col3:
                # Reading time estimate
                reading_time = word_count / 200  # Average reading speed: 200 words/min
                st.metric("Reading Time", f"{reading_time:.1f} min")
                st.caption("Based on 200 words/min")
            
            # Additional metrics
            st.markdown("---")
            st.markdown("**Detailed Analysis**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Average Syllables per Word:** {sum([max(1, len([c for c in w if c in 'aeiouAEIOU'])) for w in words if w.isalnum()]) / max(word_count, 1):.2f}")
                st.write(f"**Average Words per Sentence:** {avg_sentence_length:.1f}")
                st.write(f"**Longest Word Length:** {len(max((w for w in words if w.isalnum()), key=len, default=''))} characters")
            
            with col2:
                # Parts of speech distribution
                tagged_words = pos_tag([w for w in words if w.isalnum()])
                pos_counts = Counter([tag for word, tag in tagged_words])
                
                st.write("**Top Parts of Speech:**")
                for pos, count in pos_counts.most_common(5):
                    st.write(f"• {pos}: {count} ({count/len(tagged_words)*100:.1f}%)")
        
        # Tab 6: Keywords
        with tabs[5]:
            st.subheader("🔑 Keyword Extraction")
            
            # Extract keywords using frequency and POS tagging
            stop_words = set(stopwords.words('english'))
            words_lower = [w.lower() for w in word_tokenize(story_text) if w.isalnum()]
            
            # Filter by POS (nouns, verbs, adjectives)
            tagged_words = pos_tag([w for w in word_tokenize(story_text) if w.isalnum()])
            important_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']
            keywords_pos = [word.lower() for word, tag in tagged_words if tag in important_tags and word.lower() not in stop_words and len(word) > 2]
            
            keyword_freq = Counter(keywords_pos).most_common(20)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Top Keywords (by frequency)**")
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                words_list = [kw[0] for kw in keyword_freq[:15]]
                freqs = [kw[1] for kw in keyword_freq[:15]]
                
                ax.barh(words_list[::-1], freqs[::-1], color='#388bfd')
                ax.set_xlabel('Frequency')
                ax.set_title('Top 15 Keywords')
                ax.set_facecolor('#161b22')
                fig.patch.set_facecolor('#0d1117')
                ax.tick_params(colors='white')
                ax.title.set_color('white')
                ax.xaxis.label.set_color('white')
                for spine in ax.spines.values():
                    spine.set_edgecolor('white')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**Keyword List**")
                for i, (keyword, freq) in enumerate(keyword_freq, 1):
                    st.write(f"{i}. **{keyword}** ({freq}x)")
            
            # Download keywords
            st.markdown("---")
            keywords_text = "\n".join([f"{kw}: {freq}" for kw, freq in keyword_freq])
            st.download_button(
                label="📥 Download Keywords",
                data=keywords_text,
                file_name="story_keywords.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Select a sample story or enter your own text to begin analysis.")
        st.markdown("---")
        st.markdown("""
        **Story Analysis Features:**
        - ☁️ Word cloud visualization
        - 💭 Sentiment analysis (polarity & subjectivity)
        - 🏷️ Named entity recognition (people, places, organizations)
        - 📖 Readability metrics (Flesch scores, grade level)
        - 🔑 Keyword extraction with POS tagging
        - 📊 Comprehensive text statistics
        - 📈 Sentiment timeline across sentences
        - 🎨 Customizable visualizations
        """)

# Footer
st.markdown("---")
st.caption("🎬 DataSense Unstructured Data Explorer | Built with Streamlit")
