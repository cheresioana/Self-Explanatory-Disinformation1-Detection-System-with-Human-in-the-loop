import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_sbert_embedding(text):
    embeddings = sbert_model.encode(text)
    return np.array(embeddings)