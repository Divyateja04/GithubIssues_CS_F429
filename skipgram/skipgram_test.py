# Importing the Libraries
import math
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extracting and opening dataset
DATASET_LOCATION = "./data/"
file_pairs1 = pd.read_csv(
    DATASET_LOCATION + "file_pairs_1.csv", header=None, index_col=False)
file_pairs2 = pd.read_csv(
    DATASET_LOCATION + "file_pairs_2.csv", header=None, index_col=False)


# Creating a dictionary of all the unique files
file_pair_indices = {}

counter = 0
# Creating a dictionary of all the unique files for this repo
file_pair_indices = {}

counter = 0
for (x, y) in file_pairs1.values:
    if x not in file_pair_indices.keys():
        file_pair_indices[x] = counter
        counter += 1
    if y not in file_pair_indices.keys():
        file_pair_indices[y] = counter
        counter += 1
for (x, y) in file_pairs2.values:
    if x not in file_pair_indices.keys():
        file_pair_indices[x] = counter
        counter += 1
    if y not in file_pair_indices.keys():
        file_pair_indices[y] = counter
        counter += 1


# Importing the skipgram model
class SkipgramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipgramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim * 3)
        self.linear2 = nn.Linear(embedding_dim * 3, vocab_size)

    def forward(self, inputs__):
        embeds = self.embedding(inputs__)
        out = self.linear(embeds)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def prediction(self, inputs):
        embeds = self.embedding(inputs)
        return embeds


model = SkipgramModel(len(file_pair_indices), 128)
model.load_state_dict(
    torch.load(
        "./skipgram/commit_skipgram.pth", map_location=torch.device(device)
    )
)
model.eval()


# Similarity Check
def similarity_check(target):
    if target not in file_pair_indices.keys():
        return []
    
    # Predict what vector comes for the given target
    target_vector = model.prediction(
        torch.tensor(file_pair_indices[target]).to(device))

    similarities = []
    # For every other vector in the vocabulary
    for i in tqdm(range(len(file_pair_indices))):
        # If it's same as the target, skip
        if list(file_pair_indices)[i] == target:
            continue

        # Find the vector embedding of the other vector
        vector = model.prediction(
            torch.tensor(file_pair_indices[list(file_pair_indices)[i]]).to(device)
        )

        # Find the cosine similarity between the two vectors
        cosine_sim = F.cosine_similarity(
            vector, target_vector, dim=0).data.tolist()

        # Append the cosine similarity and the word to the similarities list
        similarities.append([list(file_pair_indices)[i], cosine_sim])

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:20]

# Write code to get accuracy of this skipgram
def get_skipgram_accuracy():
    total = 0
    correct = 0
    
    temp_store_scores = {}
    
    for x in tqdm(file_pair_indices.keys()):
        for y in (file_pair_indices.keys()):
            if x == y or x + " " + y in temp_store_scores.keys() or y + " " + x in temp_store_scores.keys():
                continue
            
            target_vector = model.prediction(torch.tensor(file_pair_indices[x]).to(device))
            vector = model.prediction(torch.tensor(file_pair_indices[y]).to(device))

            # Find the cosine similarity between the two vectors
            cosine_sim = F.cosine_similarity(vector, target_vector, dim=0).data.tolist()
            
            temp_store_scores[x + " " + y] = math.abs(cosine_sim)

    total = len(temp_store_scores)
    for x in tqdm(temp_store_scores.keys()):
        print(x + " " + str(temp_store_scores[x]))
        if x.split(" ")[0] == x.split(" ")[1]:
            continue
        if temp_store_scores[x] > 0.5:
            correct += 1
    
    return correct/total

print(get_skipgram_accuracy())