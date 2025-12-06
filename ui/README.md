# GraphBuilder-RAG UI

Simple Streamlit interface for querying the GraphBuilder-RAG system.

## Setup

1. **Install Streamlit:**
   ```bash
   pip install streamlit
   ```

2. **Ensure API is running:**
   ```bash
   # In another terminal
   python api/main.py
   ```

3. **Run the UI:**
   ```bash
   streamlit run ui/app.py
   ```

4. **Open browser:**
   - UI will open automatically at `http://localhost:8501`
   - Or manually navigate to the URL shown in terminal

## Features

- 🔍 **Simple Query Interface** - Ask questions in natural language
- 💡 **Answer Display** - Clear, formatted answers with citations
- 📋 **Claims Extraction** - See individual factual claims with evidence
- 📚 **Source Attribution** - View graph entities and text chunks used
- ✅ **Verification Status** - See if claims are verified by the knowledge graph
- 🧠 **Reasoning Trace** - Understand how the answer was generated
- ⚙️ **Configurable Settings** - Adjust max chunks, temperature, verification

## Usage

1. Enter your question in the text box
2. Click "Query" or press Enter
3. View the answer, claims, sources, and verification
4. Use sample questions for quick testing
5. Adjust settings in the sidebar as needed

## Requirements

- API server running at `http://localhost:8000`
- Streamlit installed (`pip install streamlit`)
- Documents ingested and processed in the system
