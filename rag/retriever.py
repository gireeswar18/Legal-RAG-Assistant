# import pckgs
import faiss
from sentence_transformers import SentenceTransformer

# get the index of top document (most matching)
def get_top_document(index, model, question):
    # get embedding for our question
    question_embedding = model.encode([question])

    # calc distance between our qn embedding and dataset embedding with top-k nearest match
    distances, indices = index.search(question_embedding, k=3)

    return indices[0]