# 🖼️ Image Analysis Features - Documentation

## Overview
The Image Analysis module provides comprehensive AI-powered image processing and analysis capabilities using state-of-the-art libraries including **rembg**, **DeepFace**, and **ONNX Runtime**.

## Technology Stack

### Core Libraries
- **rembg (v2.0.61)**: AI-powered background removal using U^2-Net deep learning model
- **DeepFace (v0.0.95)**: Advanced facial recognition and analysis framework
- **OpenCV (cv2)**: Computer vision and image processing
- **Pillow (PIL)**: Python Imaging Library for basic operations
- **scikit-image**: Advanced image analysis and filters
- **matplotlib**: Visualization and plotting
- **scikit-learn**: Machine learning for color clustering
- **ONNX Runtime (v1.16.3)**: Optimized inference engine for deep learning models

## Features Breakdown

### 1. 🎨 Color Analysis
Comprehensive color space analysis and extraction:

#### Features:
- **RGB Channel Histograms**: Detailed distribution of Red, Green, and Blue channels
- **Dominant Color Extraction**: Uses K-means clustering to identify top 5 dominant colors
- **Color Palette Generation**: Visual representation with RGB values and percentages
- **Brightness & Contrast Metrics**:
  - Average brightness (mean intensity)
  - Contrast (standard deviation)
  - Min/Max intensity values
  
#### Use Cases:
- Brand color extraction from logos
- Image theme analysis
- Quality assessment
- Color scheme generation

---

### 2. ✂️ AI-Powered Background Removal
Advanced background removal using deep learning:

#### Technology:
- **rembg library** with U^2-Net architecture
- Pre-trained on human segmentation datasets
- Automatic subject detection
- High-quality edge preservation

#### Features:
- One-click background removal
- Side-by-side comparison (before/after)
- PNG export with transparency
- Download processed images

#### Use Cases:
- Product photography
- Portrait editing
- E-commerce listings
- Social media content
- Professional headshots

---

### 3. 👤 AI Face Analysis (DeepFace)
Comprehensive facial recognition and emotion analysis:

#### Powered by DeepFace Framework:
- Multi-task deep learning models
- Real-time analysis
- High accuracy facial attribute recognition

#### Analysis Capabilities:
1. **Demographic Analysis**:
   - Age estimation
   - Gender prediction
   - Ethnicity/race classification

2. **Emotion Detection** (7 emotions):
   - Happy
   - Sad
   - Angry
   - Surprise
   - Fear
   - Disgust
   - Neutral

3. **Visualization**:
   - Confidence scores for each emotion
   - Horizontal bar chart showing emotion distribution
   - Top 3 emotions displayed prominently

#### Use Cases:
- User experience research
- Marketing campaign analysis
- Security and surveillance
- Interactive applications
- Emotion-based content recommendations

---

### 4. 🔍 Edge Detection
Multiple advanced edge detection algorithms:

#### Available Algorithms:
1. **Canny Edge Detection**:
   - Multi-stage algorithm
   - Noise reduction
   - Best for general purpose

2. **Sobel Operator**:
   - First-order derivative
   - Good for gradient detection

3. **Prewitt Filter**:
   - Similar to Sobel
   - Simpler kernel

4. **Roberts Cross**:
   - Quick, simple edges
   - 2x2 kernel

5. **Scharr Operator**:
   - More accurate than Sobel
   - Better rotation invariance

#### Metrics Provided:
- Total edge pixels
- Edge coverage percentage
- Maximum edge strength

#### Use Cases:
- Object boundary detection
- Pattern recognition
- Image segmentation
- Quality inspection

---

### 5. 🎭 Filters & Effects
8 professional image filters and effects:

#### Available Filters:

1. **Blur (Gaussian)**:
   - Noise reduction
   - Smooth transitions
   - Sigma = 3

2. **Sharpen**:
   - Edge enhancement
   - Detail improvement
   - 3x3 kernel convolution

3. **Emboss**:
   - 3D relief effect
   - Artistic rendering
   - Directional lighting simulation

4. **Negative**:
   - Color inversion
   - Artistic effect
   - 255 - pixel value

5. **Sepia**:
   - Vintage/retro look
   - Warm brown tones
   - Matrix transformation

6. **Grayscale**:
   - Black and white conversion
   - Color channel averaging
   - Classic photography

7. **Equalize Histogram**:
   - Contrast enhancement
   - Dynamic range improvement
   - Automatic brightness adjustment

8. **Contrast Stretch**:
   - Intensity rescaling
   - 2nd-98th percentile range
   - Improved visibility

#### Features:
- Real-time preview
- Side-by-side comparison
- Download filtered images
- Preserve original quality

---

### 6. 📊 Advanced Analysis
Comprehensive image quality and statistical metrics:

#### Quality Metrics:

1. **Sharpness Score**:
   - Laplacian variance method
   - Measures image focus
   - Higher = sharper

2. **Entropy**:
   - Information content measure
   - Shannon entropy calculation
   - Indicates image complexity

3. **Signal-to-Noise Ratio (SNR)**:
   - Mean/Standard deviation ratio
   - Quality indicator
   - Higher = cleaner image

#### Image Statistics:
- Color mode (RGB, RGBA, Grayscale, etc.)
- Aspect ratio calculation
- Total pixel count
- Unique colors (sampled)

#### Visualizations:
- Intensity distribution histogram
- Grayscale conversion
- Statistical plots

---

## Workflow Example

### Typical Usage Flow:
```
1. Select "🖼️ Image Analysis" from sidebar
2. Upload image (JPG, PNG, GIF, BMP, WebP, TIFF)
3. View image metadata (dimensions, format, size)
4. Navigate through analysis tabs:
   - Color Analysis → Extract dominant colors
   - Background Removal → Remove background with AI
   - Face Analysis → Detect emotions and demographics
   - Edge Detection → Choose algorithm and detect edges
   - Filters & Effects → Apply artistic filters
   - Advanced Analysis → View quality metrics
5. Download processed images
```

---

## Technical Implementation

### Image Processing Pipeline:

```python
# 1. Upload and Load
image_file = st.file_uploader(...)
image = Image.open(image_path)
img_array = np.array(image)

# 2. Background Removal (rembg)
from rembg import remove
output_data = remove(input_data)

# 3. Face Analysis (DeepFace)
from deepface import DeepFace
analysis = DeepFace.analyze(
    img_path=image_path,
    actions=['age', 'gender', 'race', 'emotion']
)

# 4. Edge Detection (scikit-image)
from skimage import feature
edges = feature.canny(gray, sigma=2)

# 5. Color Clustering (scikit-learn)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
dominant_colors = kmeans.cluster_centers_
```

---

## Performance Considerations

### Optimization Strategies:
1. **Pixel Sampling**: Sample 10,000 pixels for color analysis (faster processing)
2. **Lazy Loading**: Import heavy libraries only when needed
3. **Temporary Files**: Automatic cleanup of processed images
4. **Progress Indicators**: Spinners for AI operations
5. **Error Handling**: Graceful fallbacks for face detection failures

### Resource Usage:
- **DeepFace**: Downloads pre-trained models (~50-100MB) on first use
- **rembg**: Uses U^2-Net model (~176MB)
- **Memory**: Processes images in-memory for speed
- **CPU/GPU**: Automatically uses GPU if available (via TensorFlow/ONNX)

---

## Supported Image Formats

### Input Formats:
- ✅ JPG/JPEG
- ✅ PNG (with transparency)
- ✅ GIF
- ✅ BMP
- ✅ WebP
- ✅ TIFF

### Output Formats:
- PNG (for transparency - background removal)
- Original format (for filters)

---

## Error Handling

### Common Scenarios:
1. **No Face Detected**: Shows warning with helpful tip
2. **Large Images**: Automatic sampling for performance
3. **Unsupported Format**: File uploader restriction
4. **Processing Errors**: Try-except blocks with user-friendly messages

---

## Future Enhancements

### Potential Features:
- 🔮 Object detection (YOLO, ONNX models)
- 🎨 Style transfer
- 🔍 OCR (Optical Character Recognition)
- 📐 Image similarity search
- 🖼️ Batch processing
- 🎯 Custom model integration
- 🌈 Advanced color grading
- 🔧 Image restoration

---

## Dependencies

### Required Packages:
```bash
pip install rembg deepface onnxruntime opencv-python pillow \
            scikit-image matplotlib scikit-learn scipy numpy
```

### Deep Learning Frameworks:
- TensorFlow 2.13.1 (DeepFace backend)
- ONNX Runtime 1.16.3 (rembg inference)
- Keras 2.15.0

---

## Troubleshooting

### Common Issues:

1. **"No module named 'deepface'"**
   ```bash
   pip install deepface
   ```

2. **"rembg model not found"**
   - First run downloads model automatically
   - Requires internet connection
   - Models cached in ~/.u2net/

3. **Face detection fails**
   - Ensure frontal, well-lit face
   - Try different image
   - Check image resolution (>200px recommended)

4. **Memory errors with large images**
   - Resize image before processing
   - Use pixel sampling options
   - Process locally for very large files

---

## Usage Examples

### Example 1: E-commerce Product Photo
```
1. Upload product image
2. Use "Background Removal" to isolate product
3. Download PNG with transparency
4. Use in online store listings
```

### Example 2: User Sentiment Analysis
```
1. Upload user photo/selfie
2. Navigate to "Face Analysis"
3. View emotion distribution
4. Extract happiness score for UX metrics
```

### Example 3: Logo Color Extraction
```
1. Upload logo image
2. Go to "Color Analysis"
3. View dominant colors palette
4. Use RGB values for brand guidelines
```

---

## Best Practices

### For Best Results:
1. **Image Quality**: Use high-resolution images (>1000px)
2. **Lighting**: Well-lit images work better for face analysis
3. **Background Removal**: Clear subject separation from background
4. **Face Analysis**: Frontal faces, not profile views
5. **Edge Detection**: Choose algorithm based on image type

---

## API & Integration

### Streamlit Integration:
```python
# Access from sidebar
data_type = st.radio(
    "Select analysis type:",
    ["🎥 Video", "📝 Text", "🎵 Audio", "🖼️ Image Analysis"]
)

# Upload and process
image_file = st.file_uploader("Upload image", type=["jpg", "png"])
if image_file:
    # Process with AI models
    ...
```

---

## Credits

### Libraries & Models:
- **rembg**: https://github.com/danielgatis/rembg
- **DeepFace**: https://github.com/serengil/deepface
- **ONNX Runtime**: https://onnxruntime.ai/
- **U^2-Net**: Salient Object Detection model
- **scikit-image**: https://scikit-image.org/

---

## Version History

### v1.0.0 (2025-10-17)
- Initial release
- All 6 analysis tabs implemented
- AI-powered features integrated
- Production-ready

---

## Contact & Support

For issues, feature requests, or contributions:
- GitHub: jaideep-27/ict
- File: run2.py (lines 1190-1639)

---

**Made with ❤️ using Streamlit, rembg, DeepFace, and ONNX Runtime**
