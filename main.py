# import pckgs
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from rag.retriever import get_top_document
from rag.ask_llm import ask_llm

# load faiss index -> it has embeddings
index = faiss.read_index("data/faiss.index")

# load documents
with open("data/documents.pkl", "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Ask a legal question: (q for quitting)")
while True:
    # ask question
    question = input("You: ")

    if question == "q":
        break

    # get top matches for question
    doc_indices = get_top_document(index, embedder, question)
    contexts = [documents[i] for i in doc_indices]
    combined_context = "\n\n".join(contexts)

    # ask our llm
    answer = ask_llm(combined_context, question)
    print("Model: ", answer)
    print()