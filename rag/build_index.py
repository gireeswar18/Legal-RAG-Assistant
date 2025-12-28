# import pckgs
import faiss
import pickle
from chunk_text import chunk_text
from sentence_transformers import SentenceTransformer

# Takes docs and gives FAISS + embedding model + embeddings
def build_faiss_index(documents):
    # use a model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # get embeddings
    embeddings = model.encode(documents, show_progress_bar=True)

    # for our model its 384
    dimension = embeddings.shape[1]

    # initialize faiss index for dimension using FlatL2 which calc dist between vectors
    index = faiss.IndexFlatL2(dimension)

    # add our embeddings
    index.add(embeddings)

    return index

# ------------------MAIN CODE------------------
# read book
with open("../data/book.txt", "r", encoding="utf-8") as f:
    text = f.read()

# chunk them
documents = chunk_text(text)

print("Total chunks:", len(documents))

# create embeddings
index = build_faiss_index(documents)

# save them
faiss.write_index(index, "../data/faiss.index")

with open("../data/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index built and saved successfully")