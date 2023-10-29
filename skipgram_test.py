# Importing the Libraries
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The device being used is: {device}")

# Extracting and opening dataset
DATASET_LOCATION = "./data/"
file_pairs = pd.read_csv(
    DATASET_LOCATION + "file_pairs_1.csv", header=None, index_col=False)
print(f"The number of pairs is: {len(file_pairs)}")

# Creating a dictionary of all the unique files
file_pair_indices = {}

counter = 0
for (x, y) in file_pairs.values:
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
        self.linear2 = nn.Linear(embedding_dim * 3, embedding_dim)

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
        "./models/commit_skipgram.pth", map_location=torch.device(device)
    )
)
model.eval()


# Similarity Check
def similarity_check(target, vocabulary):
    # Predict what vector comes for the given target
    target_vector = model.prediction(
        torch.tensor(file_pair_indices[target]).to(device))

    similarities = []
    # For every other vector in the vocabulary
    for i in tqdm(range(len(vocabulary))):
        # If it's same as the target, skip
        if list(vocabulary)[i] == target:
            continue

        # Find the vector embedding of the other vector
        vector = model.prediction(
            torch.tensor(file_pair_indices[list(vocabulary)[i]]).to(device)
        )

        # Find the cosine similarity between the two vectors
        cosine_sim = F.cosine_similarity(
            vector, target_vector, dim=0).data.tolist()

        # Append the cosine similarity and the word to the similarities list
        similarities.append([list(vocabulary)[i], cosine_sim])

    return sorted(similarities, key=lambda x: x[1], reverse=True)[:20]


sims = similarity_check("github/PullRequestReview.py", file_pair_indices)
for sim in sims:
    print(sim)
