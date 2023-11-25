"""
Clustering the outputs from skipgram model
We import the models from already pre trained model on the HPC, and try to cluster this data together
"""

# Importing the Libraries
import json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import logging
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    datefmt="%d-%b-%y %H:%M:%S",
    filename=f"./logs/skipgram_logs/clustering-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_app.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Importing the data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extracting and opening dataset
DATASET_LOCATION = "../data/"
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

# Create a model instance
model = SkipgramModel(len(file_pair_indices), 128)
model.load_state_dict(
    torch.load(
        "./commit_skipgram.pth", map_location=torch.device(device)
    )
)
model.eval()

vectors = []

# For every vector in the file_pair_indices
for i in tqdm(range(len(file_pair_indices))):
    # Find the vector embedding of the other vector
    vector = model.prediction(
        torch.tensor(file_pair_indices[list(file_pair_indices)[i]]).to(device)
    )
    vectors.append(vector.detach().numpy())

vectors = pd.DataFrame(vectors)
vectors = vectors.values

# We use PCA
pca = PCA(n_components=7)
xIn7Dims = pca.fit_transform(vectors)

# Set a KMeans clustering
kmeans = KMeans(n_clusters=8, n_init="auto")

# Compute cluster centers and predict cluster indices
X_clustered = kmeans.fit_predict(xIn7Dims)

# We need to find the cluster centers
centers = kmeans.cluster_centers_

# Define our own color map
LABEL_COLOR_MAP = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "pink",
    5: "orange",
    6: "purple",
    7: "black",
}
label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

for index, center in enumerate(centers):
    original_dimension_vector = torch.tensor(pca.inverse_transform(center)).to(device)

    closest_vector = None
    closest_api = None
    closest_vector_similarity = 0

    # Now we find the closest vector to this vector
    # For every other vector in the file_pair_indices
    for i in range(len(file_pair_indices)):
        # Find the vector embedding of the other vector
        vector = model.prediction(
            torch.tensor(file_pair_indices[list(file_pair_indices)[i]]).to(device)
        )

        # Find the cosine similarity between the two vectors
        cosine_sim = F.cosine_similarity(vector, original_dimension_vector, dim=0).data

        # We check if we have higher similarity
        if float(cosine_sim) > closest_vector_similarity:
            closest_vector_similarity = cosine_sim
            closest_vector = vector
            closest_api = list(file_pair_indices)[i]

    # we print the closest vector
    logging.info(f"{LABEL_COLOR_MAP[index]} ==> {closest_api}")

# Plot the scatter digram
plt.figure(figsize=(7, 7))
plt.scatter(xIn7Dims[:, 0], xIn7Dims[:, 3], c=label_color, alpha=0.5)
plt.savefig("./clustering.png")
