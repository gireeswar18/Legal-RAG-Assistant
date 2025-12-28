# import pckgs
import pickle
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from rag.retriever import get_top_document
from rag.ask_llm import ask_llm

# load faiss index -> it has embeddings
index = faiss.read_index("data/faiss.index")

# load documents
with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Legal RAG Assistant", layout="centered")

st.title("Legal RAG Assistant")
st.caption("Answers are only generated from the available legal documents")

with st.form("qa_form"):
    question = st.text_area("Ask a legal question:")
    submitted = st.form_submit_button("Send")

if submitted and question.strip():

    with st.spinner("Thinking..."):    
        # get top matches for question
        doc_indices = get_top_document(index, embedder, question)
        contexts = [documents[i] for i in doc_indices]
        combined_context = "\n\n".join(contexts)
        # ask our llm
        answer = ask_llm(combined_context, question)

    st.subheader("Answer: ")
    st.write(answer)