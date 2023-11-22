#importing SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json
import os

model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)

DATASET_LOCATION = "./data/"
with open(os.path.join(DATASET_LOCATION, "functions_per_file(without _ ).json"), "r") as f:
    data = json.load(f)

sentences = []

for file in data:
    sentences.append(file["fileName"][0:-3] + ", "+ file["functions"])

sentence_vecs = model.encode(sentences)

input_issue = "PullRequest.is_merged does not handle token expiration"
input_vec = model.encode(input_issue)

cosineSimilarity = cosine_similarity(
    input_vec.reshape(1, -1),
    sentence_vecs
)

data[cosineSimilarity.argmax()]




