#importing SentenceTransformer
from sentence_transformers import SentenceTransformer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'sentence-transformers/all-mpnet-base-v2'
model = SentenceTransformer(model_name)

DATASET_LOCATION = "../data/"

with open(os.path.join(DATASET_LOCATION, "functions_per_file.json"), "r") as f:
    data = json.load(f)
    
function_embeds = {}

for file in data:
    for function in file['functions']:
        if not function.startsWith("_"):
            function_embeds[file['filename'] + '/' +  function["function_name"]] = model.encode(file['filename'] + '/' +  function["function_name"], convert_to_tensor=True)
            
print(len(function_embeds) + " Embeddings generated")