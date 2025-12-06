"""
Simple Streamlit UI for GraphBuilder-RAG Query Interface
"""
import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="GraphBuilder-RAG Query",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 GraphBuilder-RAG Query Interface")
st.markdown("Ask questions and get answers with graph + text context")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    max_chunks = st.slider("Max Text Chunks", 1, 20, 5)
    require_verification = st.checkbox("Require Verification", value=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
    
    st.markdown("---")
    st.markdown("### System Info")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")

# Main query interface
question = st.text_input(
    "Enter your question:",
    placeholder="e.g., What did Albert Einstein discover?",
    help="Ask any question about the ingested documents"
)

if st.button("🚀 Query", type="primary") or (question and st.session_state.get('last_question') != question):
    if question:
        st.session_state['last_question'] = question
        
        with st.spinner("🔎 Searching knowledge graph and documents..."):
            try:
                response = requests.post(
                    f"{API_URL}/api/v1/query",
                    json={
                        "question": question,
                        "max_chunks": max_chunks,
                        "require_verification": require_verification,
                        "temperature": temperature
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Parse the answer - it might be a JSON string or dict
                    answer_obj = result.get("answer", {})
                    if isinstance(answer_obj, str):
                        try:
                            answer_obj = json.loads(answer_obj)
                        except:
                            answer_obj = {"answer": answer_obj}
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    answer_text = answer_obj.get("answer", "No answer available")
                    st.success(answer_text)
                    
                    # Display in columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Claims
                        st.markdown("### 📋 Claims")
                        claims = answer_obj.get("claims", [])
                        if claims:
                            for i, claim in enumerate(claims, 1):
                                with st.expander(f"Claim {i}: {claim.get('claim', '')[:50]}...", expanded=True):
                                    st.markdown(f"**📝 Statement:**")
                                    st.write(claim.get('claim', 'N/A'))
                                    
                                    evidence_type = claim.get('evidence_type', 'N/A')
                                    evidence_emoji = {
                                        'graph': '🔗',
                                        'text': '📄',
                                        'inference': '🧠'
                                    }.get(evidence_type, '❓')
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("Evidence Type", f"{evidence_emoji} {evidence_type}")
                                    with col_b:
                                        st.metric("Confidence", f"{claim.get('confidence', 0):.0%}")
                                    
                                    if claim.get('evidence_ids'):
                                        st.markdown("**🔍 Evidence:**")
                                        for eid in claim.get('evidence_ids', []):
                                            st.code(eid, language=None)
                        else:
                            st.info("No structured claims - answer provided as free text")
                        
                        # Sources
                        st.markdown("### 📚 Sources")
                        sources = result.get("sources", [])
                        if sources:
                            # Categorize sources
                            chunks = [s for s in sources if s.startswith("Chunk:")]
                            entities = [s for s in sources if s.startswith("Entity:")]
                            docs = [s for s in sources if s.startswith("Doc:")]
                            
                            if chunks:
                                st.markdown("**📄 Text Chunks:**")
                                for chunk in chunks:
                                    st.code(chunk, language=None)
                            
                            if entities:
                                st.markdown("**🔗 Graph Entities:**")
                                for entity in entities:
                                    st.code(entity, language=None)
                            
                            if docs:
                                st.markdown("**📑 Documents:**")
                                for doc in docs:
                                    st.code(doc, language=None)
                        else:
                            st.info("No sources available")
                    
                    with col2:
                        # Reasoning trace
                        st.markdown("### 🧠 Reasoning Trace")
                        reasoning = answer_obj.get("reasoning_trace", "")
                        if reasoning:
                            st.markdown("**How the answer was generated:**")
                            st.info(reasoning)
                        else:
                            st.warning("No reasoning trace provided")
                        
                        # Verification (if available)
                        st.markdown("### ✅ Verification")
                        verification = result.get("verification", {})
                        if verification and verification.get("status"):
                            status = verification.get("status", "unknown")
                            status_color = {
                                "verified": "🟢",
                                "unsupported": "🟡",
                                "contradicted": "🔴",
                                "unknown": "⚪"
                            }.get(status, "⚪")
                            
                            st.metric(
                                "Verification Status",
                                f"{status_color} {status.upper()}",
                                delta=f"Score: {verification.get('score', 'N/A')}"
                            )
                            
                            if verification.get("verified_edges"):
                                st.write("**✅ Verified Edges:**")
                                for edge in verification["verified_edges"]:
                                    st.code(edge, language=None)
                            
                            if verification.get("contradicted_edges"):
                                st.write("**❌ Contradicted Edges:**")
                                for edge in verification["contradicted_edges"]:
                                    st.code(edge, language=None)
                        else:
                            st.info("Verification disabled or not available")
                    
                    # Metadata
                    with st.expander("📊 Query Metadata", expanded=False):
                        metadata = result.get("metadata", {})
                        if metadata:
                            st.json(metadata)
                        else:
                            st.json(result)
                    
                else:
                    st.error(f"❌ Error: {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timeout - query took too long")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Connection error - is the API running?")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a question")

# Sample questions
with st.expander("💡 Sample Questions", expanded=False):
    samples = [
        "What did Albert Einstein discover?",
        "Who won the Nobel Prize?",
        "What did Isaac Newton publish?",
        "Tell me about Marie Curie's work",
        "What awards did Einstein receive?"
    ]
    
    for sample in samples:
        if st.button(sample, key=f"sample_{sample}"):
            st.session_state['last_question'] = sample
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>GraphBuilder-RAG • Hybrid Retrieval (FAISS + Neo4j) • LLM-powered QA with Verification</small>
    </div>
    """,
    unsafe_allow_html=True
)
