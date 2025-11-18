# UNA News Reviewer API 🗞️

Professional news article review and editing system for the Union of News Agencies (UNA) of the Organization of Islamic Cooperation.

## 🌟 Features

### Core Functionality
- **Automatic News Review**: Reviews Arabic news articles according to UNA editorial guidelines
- **Title Simplification**: Removes honorific phrases and prayers (e.g., "حضرة صاحب الجلالة" → "الملك")
- **Paragraph Division**: Automatically divides articles into logical paragraphs using structured output
- **RAG (Retrieval Augmented Generation)**: Uses semantic search to retrieve relevant guidelines and examples
- **Document Management**: Upload and manage guideline and example documents in DOCX format

### Technical Highlights
- **Structured Output with Pydantic**: Guarantees proper paragraph division using OpenAI's structured output feature
- **Professional Prompt Engineering**: Carefully crafted prompts for consistent results
- **Semantic Embeddings**: Uses `text-embedding-3-large` for high-accuracy similarity search
- **GPT-4o Integration**: Latest OpenAI model with native JSON schema support

## 📋 Requirements

- Python 3.10+
- Django 5.2+
- OpenAI API Key
- SQLite (default) or PostgreSQL (production)

## 🚀 Installation

### 1. Clone and Setup Environment

```bash
cd News_Reviewer_api
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run Migrations

```bash
python manage.py migrate
```

### 5. Start Server

```bash
python manage.py runserver
```

The API will be available at `http://127.0.0.1:8000/`

## 📚 API Endpoints

### 1. Upload Guidelines
Upload DOCX file containing editorial guidelines.

**Endpoint**: `POST /api/upload-instruction/`

**Request**:
```json
{
  "file": "<DOCX file>",
  "title": "UNA Editorial Guidelines"
}
```

**Response**:
```json
{
  "detail": "تم تحديث التعليمات بنجاح.",
  "chunks_created": 5,
  "title": "UNA Editorial Guidelines"
}
```

### 2. Upload Examples
Upload DOCX file containing example articles.

**Endpoint**: `POST /api/upload-example/`

**Request**:
```json
{
  "file": "<DOCX file>",
  "title": "Example News Articles"
}
```

**Response**:
```json
{
  "detail": "تم تحديث الأمثلة بنجاح.",
  "chunks_created": 3,
  "title": "Example News Articles"
}
```

### 3. Review News Article
Submit an article for review and editing.

**Endpoint**: `POST /api/review-news/`

**Request**:
```json
{
  "news_text": "جلالةُ السلطان المعظم ورئيس الوزراء...",
  "top_guidelines": 5,
  "top_examples": 3
}
```

**Response**:
```json
{
  "review": "مدريد في 5 نوفمبر / العُمانية / شهد السلطان هيثم بن طارق...\n\nتمثلت الاتفاقية في الإعفاء المتبادل...\n\nوشملت مذكرات التفاهم مجالات الثقافة..."
}
```

## 🛠️ How It Works

### 1. Document Ingestion
1. User uploads DOCX files (guidelines/examples)
2. System extracts text and chunks it (max 4000 chars per chunk)
3. Generates embeddings using `text-embedding-3-large`
4. Stores chunks with embeddings in database

### 2. Semantic Search (RAG)
1. User submits news article for review
2. System generates embedding for the article
3. Retrieves most similar guideline and example chunks using cosine similarity
4. Uses retrieved context to inform the review process

### 3. Article Review
1. **Preprocessing**: Applies regex patterns to simplify obvious honorifics
2. **Structured Generation**: Uses OpenAI's `beta.chat.completions.parse` with Pydantic schema
3. **Paragraph Division**: Model returns JSON with `paragraphs` array (3-7 paragraphs)
4. **Post-processing**: Final cleanup of any remaining honorifics
5. **Output**: Returns properly formatted article with blank lines between paragraphs

## 🎯 Editorial Rules Applied

### Title Simplification
| Original | Simplified |
|----------|-----------|
| حضرة صاحب الجلالة الملك | الملك |
| جلالة السلطان المعظم | السلطان |
| صاحب السمو الملكي الأمير | الأمير |
| سموه الكريم | سموه |

### Prayer Phrases (Removed Completely)
- حفظه الله
- رعاه الله
- نصره الله
- أطال الله عمره
- حفظهما الله

### Redundant Words (Removed)
- خالص
- المعظم
- الكريم (when redundant)

## 🔧 Configuration

### Models Used

```python
# Embedding Model (Higher accuracy)
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"

# Completion Model (Supports structured outputs)
DEFAULT_COMPLETION_MODEL = "gpt-4o-2024-08-06"
```

### Chunking

```python
MAX_CHUNK_CHAR_LENGTH = 4000  # Maximum characters per chunk
```

### Retrieval

```python
top_guidelines = 5  # Number of guideline chunks to retrieve
top_examples = 3    # Number of example chunks to retrieve
```

## 📖 Example Usage

### Python Client

```python
import requests

# Upload guidelines
with open('guidelines.docx', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/api/upload-instruction/',
        files={'file': f},
        data={'title': 'UNA Guidelines'}
    )

# Review article
response = requests.post(
    'http://127.0.0.1:8000/api/review-news/',
    json={
        'news_text': 'حضرة صاحب الجلالة السلطان هيثم بن طارق...',
        'top_guidelines': 5,
        'top_examples': 3
    }
)

reviewed_article = response.json()['review']
print(reviewed_article)
```

### JavaScript/Fetch

```javascript
// Review article
const response = await fetch('http://127.0.0.1:8000/api/review-news/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    news_text: 'حضرة صاحب الجلالة السلطان...',
    top_guidelines: 5,
    top_examples: 3
  })
});

const data = await response.json();
console.log(data.review);
```

## 🏗️ Project Structure

```
News_Reviewer_api/
├── Config/              # Django settings
├── reviewer/            # Main app
│   ├── models.py        # DocumentChunk model
│   ├── serializers.py   # DRF serializers
│   ├── services.py      # Core business logic (NEW & IMPROVED)
│   ├── views.py         # API endpoints
│   └── migrations/
├── requirements.txt     # Dependencies
├── manage.py
└── README.md
```

## 🧪 Testing

Test the system with a sample article:

```bash
curl -X POST http://127.0.0.1:8000/api/review-news/ \
  -H "Content-Type: application/json" \
  -d '{
    "news_text": "جلالة الملك المعظم حفظه الله يستقبل الرئيس...",
    "top_guidelines": 5,
    "top_examples": 3
  }'
```

## 🎓 Key Improvements in This Version

### 1. **Structured Output with Pydantic**
- Uses OpenAI's `beta.chat.completions.parse` API
- Guarantees JSON response with `paragraphs` array
- **Solves the paragraph division problem completely**

### 2. **Better Model Selection**
- `text-embedding-3-large`: Higher quality embeddings (vs. small)
- `gpt-4o-2024-08-06`: Native structured output support

### 3. **Professional Code Architecture**
- Clear separation of concerns
- Comprehensive docstrings
- Type hints throughout
- Better error handling

### 4. **Improved Prompt Engineering**
- Concise and focused prompts
- Clear structure with system/user roles
- Explicit JSON schema requirements

## 📝 Notes

- **Paragraph Division**: The system now uses structured outputs to **guarantee** proper paragraph division. The model is forced to return a JSON object with a `paragraphs` array.

- **Preprocessing**: Regex patterns handle obvious cases before sending to AI, reducing API costs and improving consistency.

- **Post-processing**: Final cleanup ensures any edge cases missed by the model are caught.

- **Temperature**: Set to 0.3 for more deterministic/consistent output.

## 🤝 Contributing

This is a professional system for UNA. All modifications should maintain:
- Editorial guideline compliance
- Structured output integrity
- Professional code standards

## 📄 License

Proprietary - UNA (Union of News Agencies)

---

**Built with ❤️ for professional Arabic journalism**
