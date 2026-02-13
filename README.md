# VidGenius - YouTube Video Summarizer

AI-powered YouTube video summarizer with RAG (Retrieval-Augmented Generation) capabilities using Google Gemini.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Google Gemini API Key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd Youtube-Video-Summarizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
Open your browser and go to: http://localhost:8501

## 📋 Features

- 📺 YouTube video transcript extraction
- 🤖 AI-powered summarization using Google Gemini
- 💬 Interactive Q&A with RAG pipeline
- 🎯 Context-aware responses using ChromaDB vector database
- 🎨 Clean and intuitive Streamlit interface

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=INFO
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

## 📁 Project Structure

```
Youtube-Video-Summarizer/
├── app.py                 # Main Streamlit application
├── src/                   # Source code
│   ├── core/             # Core functionality
│   │   └── rag_pipeline.py
│   └── utils/            # Utility functions
├── tests/                # Test files
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
└── .env.example         # Environment variables template
```

## 🧪 Testing

### Run Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ app.py tests/

# Lint code
flake8 src/ app.py

# Type checking
mypy src/
```

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app"
4. Select your repository
5. Set `app.py` as the main file
6. Add `GOOGLE_API_KEY` in Secrets (Advanced settings)
7. Deploy!

### Other Platforms

- **Hugging Face Spaces**: Deploy as a Streamlit Space
- **Railway**: Connect GitHub repo and deploy
- **Render**: Deploy as a Web Service

## 🔒 Security

- Never commit your `.env` file (it's in `.gitignore`)
- Keep your API keys secure
- Rotate API keys regularly
- Use environment variables for all sensitive data

## 📝 Usage

1. Enter a YouTube video URL
2. Click "Summarize" to get an AI-generated summary
3. Ask questions about the video content
4. Get context-aware answers using RAG

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**API Key Error**
- Verify your `GOOGLE_API_KEY` is set correctly in `.env`
- Check the API key is valid and has Gemini API enabled

**Module Not Found**
- Make sure all dependencies are installed: `pip install -r requirements.txt`

**Port Already in Use**
- Streamlit default port is 8501
- Use a different port: `streamlit run app.py --server.port 8502`

**ChromaDB Issues**
- Delete the `chroma_db/` directory and restart the app
- This will recreate the vector database

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Google Gemini API](https://ai.google.dev/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

---

**Made with ❤️ using Streamlit and Google Gemini**
