# 📖 Story Analysis Feature - Documentation

## Overview
The Story Analysis module provides comprehensive Natural Language Processing (NLP) analysis of narrative text using **textblob**, **wordcloud**, and **nltk**.

## Features

### 🎯 5 Hardcoded Sample Stories
1. **The Lost Key** - A tale about lost and found
2. **The Garden** - Story of growth and friendship
3. **Digital Disconnect** - Modern life and presence
4. **The Last Letter** - Legacy and family bonds
5. **The Coffee Shop Writer** - Love and remembrance

### 📊 Six Analysis Tabs

#### 1. Overview Tab
- **Text Statistics**:
  - Word count, sentence count, character count
  - Average word/sentence length
  - Unique words and lexical diversity
  - Longest/shortest words and sentences
- **Word Frequency**: Top 10 most common words (excluding stopwords)
- **Sentence Length Distribution**: Histogram visualization

#### 2. Word Cloud Tab
- **Customizable Visualization**:
  - Adjustable max words (50-500)
  - Background color (black/white)
  - Color schemes (viridis, plasma, inferno, magma, cool, hot)
  - Automatic stopword removal
- **Download**: Save word cloud as PNG

#### 3. Sentiment Analysis Tab
- **Metrics**:
  - Polarity (-1 to +1): negative to positive
  - Subjectivity (0 to 1): objective to subjective
  - Overall sentiment classification
- **Sentiment Timeline**: Bar chart showing sentiment flow across sentences
- **Visual Indicators**: Emoji-based sentiment representation

#### 4. Named Entity Recognition Tab
- **Entity Types**:
  - PERSON: People names
  - GPE: Geo-political entities (cities, countries)
  - ORGANIZATION: Companies, institutions
  - And more...
- **Visualization**: Entity distribution chart
- **Details**: Expandable lists with frequency counts

#### 5. Readability Metrics Tab
- **Scores**:
  - Flesch Reading Ease (0-100)
  - Flesch-Kincaid Grade Level
  - Reading time estimate
- **Detailed Analysis**:
  - Average syllables per word
  - Parts of speech distribution
  - Complexity indicators

#### 6. Keyword Extraction Tab
- **POS-Tagged Keywords**: Nouns, verbs, adjectives
- **Frequency Analysis**: Top 20 keywords
- **Visualization**: Horizontal bar chart
- **Download**: Export keywords to TXT file

## Technology Stack

### Core Libraries
- **NLTK (3.9.1)**: Natural Language Toolkit
  - Tokenization (punkt)
  - POS tagging (averaged_perceptron_tagger)
  - Named entity chunking (maxent_ne_chunker)
  - Stopwords corpus
- **textblob (0.18.0)**: Simplified text processing
  - Sentiment analysis
  - Part-of-speech tagging
- **wordcloud (1.9.4)**: Word cloud generation
  - Customizable visualizations
  - Color schemes and styling

### Automatic Setup
- NLTK data auto-downloads on first use:
  - punkt (sentence tokenization)
  - averaged_perceptron_tagger (POS tagging)
  - maxent_ne_chunker (NER)
  - words (word corpus)
  - stopwords (English)

## Usage

### Basic Workflow
```
1. Run: streamlit run run2.py
2. Select "📖 Story Analysis" from sidebar
3. Choose a sample story or select "Custom Story"
4. Enter/paste your text (if custom)
5. Explore the 6 analysis tabs
6. Download visualizations and data
```

### Example Analysis: "The Lost Key"
- **Words**: ~150
- **Polarity**: Positive (happy ending)
- **Named Entities**: Emma (PERSON), coffee shop (GPE)
- **Keywords**: key, searched, rain, relief
- **Readability**: 7th-8th grade level

## Metrics Explained

### Sentiment Polarity
- **+1.0**: Extremely positive
- **+0.5**: Positive
- **0.0**: Neutral
- **-0.5**: Negative
- **-1.0**: Extremely negative

### Sentiment Subjectivity
- **1.0**: Completely subjective (opinions)
- **0.5**: Mixed (facts + opinions)
- **0.0**: Completely objective (facts only)

### Flesch Reading Ease
- **90-100**: Very Easy (5th grade)
- **80-90**: Easy (6th grade)
- **70-80**: Fairly Easy (7th grade)
- **60-70**: Standard (8-9th grade)
- **50-60**: Fairly Difficult (10-12th grade)
- **30-50**: Difficult (College)
- **0-30**: Very Difficult (Graduate)

### Flesch-Kincaid Grade
- Indicates the U.S. school grade level needed to understand the text
- Example: Score of 8.5 = suitable for 8th-9th grade

### Lexical Diversity
- Ratio of unique words to total words
- Higher = more varied vocabulary
- Range: 0.0 (repetitive) to 1.0 (all unique)

## Downloads Available

### Word Cloud
- Format: PNG image
- Filename: `story_wordcloud.png`
- Resolution: 800x400 pixels

### Keywords
- Format: Plain text
- Filename: `story_keywords.txt`
- Content: Word:frequency pairs

## Use Cases

### 1. Creative Writing Analysis
- Analyze tone and sentiment
- Check readability for target audience
- Identify overused words
- Track sentiment arc

### 2. Content Evaluation
- Assess reading difficulty
- Identify key themes
- Extract main topics
- Analyze writing style

### 3. Educational Applications
- Teaching narrative structure
- Vocabulary analysis
- Grammar and POS study
- Reading comprehension

### 4. Marketing/Copywriting
- Test message sentiment
- Optimize readability
- Identify keywords
- Audience targeting

## Sample Stories Summary

| Story | Theme | Tone | Grade Level | Key Entities |
|-------|-------|------|-------------|--------------|
| The Lost Key | Anxiety → Relief | Positive | 7-8 | Emma, coffee shop |
| The Garden | Growth, Friendship | Very Positive | 8-9 | Mr. Chen, garden |
| Digital Disconnect | Technology, Presence | Reflective | 9-10 | Sarah, camping |
| The Last Letter | Family, Legacy | Emotional | 8-9 | Margaret, grandmother |
| The Coffee Shop Writer | Love, Loss | Bittersweet | 7-8 | Writer, barista, wife |

## Technical Details

### POS Tags Used for Keyword Extraction
- **NN, NNS**: Nouns (singular, plural)
- **NNP, NNPS**: Proper nouns
- **VB, VBD, VBG, VBN, VBP, VBZ**: Verbs (all forms)
- **JJ, JJR, JJS**: Adjectives (all forms)

### Named Entity Types (NLTK)
- **PERSON**: People, characters
- **ORGANIZATION**: Companies, agencies
- **GPE**: Geo-political entities
- **LOCATION**: Non-GPE locations
- **DATE**: Dates and times
- **MONEY**: Monetary values
- **PERCENT**: Percentages
- **FACILITY**: Buildings, airports, highways
- **PRODUCT**: Objects, vehicles, food

## Troubleshooting

### NLTK Data Not Found
The app automatically downloads required NLTK data on first use. If you see errors:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('stopwords')
```

### Word Cloud Not Generating
- Ensure text has sufficient words (minimum ~20)
- Check that stopwords are being removed
- Try different color schemes

### No Named Entities Found
- Story may not contain recognizable entities
- Try text with more proper nouns (names, places)
- NLTK NER works best with formal text

## Performance

### Processing Speed
- **Small stories** (~200 words): < 1 second
- **Medium stories** (~500 words): 1-2 seconds
- **Large stories** (~1000 words): 2-4 seconds
- **First run**: +5 seconds (NLTK data download)

### Memory Usage
- Minimal memory footprint
- NLTK models: ~50MB
- Wordcloud generation: <10MB

## Future Enhancements (Potential)

- 🎯 More sample stories (10-20 total)
- 📊 Story comparison (side-by-side analysis)
- 🔍 Advanced NER (spaCy integration)
- 📈 Emotion analysis (beyond sentiment)
- 🌍 Multi-language support
- 📝 Writing style analysis
- 🎨 Custom word cloud shapes
- 💾 Save/load analysis reports

## Version History

### v1.0.0 (2025-10-17)
- Initial release
- 5 sample stories
- 6 analysis tabs
- All core NLP features

---

**Technology**: textblob, wordcloud, nltk  
**Integration**: Streamlit DataSense Platform  
**Status**: ✅ Production Ready

