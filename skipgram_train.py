# Importing the Libraries
import os
from datetime import datetime
import pandas as pd
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    datefmt="%d-%b-%y %H:%M:%S",
    filename=f"./logs/skipgram_logs/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}_app.log",
    filemode="w",
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Checking pytorch devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info(device)

# Extracting and opening dataset
DATASET_LOCATION = "./data/"
file_pairs = pd.read_csv(
    DATASET_LOCATION + "file_pairs_1.csv", header=None, index_col=False)

logging.info(f"The number of pairs is: {len(file_pairs)}")

file_pair_indices = {}

# Creating a dictionary of all the unique files
counter = 0
for (x, y) in file_pairs.values:
    if x not in file_pair_indices.keys():
        file_pair_indices[x] = counter
        counter += 1
    if y not in file_pair_indices.keys():
        file_pair_indices[y] = counter
        counter += 1


# Custom Dataset class which contains all the file pairs
class CustomDataset(Dataset):
    def __init__(self):
        self.file_pairs = list(file_pairs.values)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        return (
            int(file_pair_indices[self.file_pairs[idx][0]]),
            int(file_pair_indices[self.file_pairs[idx][1]]),
        )


customDataset = CustomDataset()

# Custom Dataloader
dataLoader = DataLoader(customDataset, batch_size=128, shuffle=True)


# Skip gram Model
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
model.to(device)
logging.info(model)

# Loss functions and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-4)


#  Training the model
def train():
    model.train()

    total_loss = 0
    total_quantity = 0

    for batch_idx, (x, y) in enumerate(dataLoader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_quantity += len(x)

    return total_loss / total_quantity


previous_loss = 1000000000

for epoch in tqdm(range(0, 50 + 1)):
    loss = train()
    logging.info(f"[Epoch: {epoch}] Loss: {loss}")

    #  Saving the model
    if loss < previous_loss:
        previous_loss = loss
        torch.save(model.state_dict(), "./models/commit_skipgram.pth")
