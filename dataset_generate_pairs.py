import json
import os
import pandas as pd

DATASET_LOCATION = "./data/"

with open(os.path.join(DATASET_LOCATION, "commits.json"), "r") as f:
    commit_data = json.load(f)

print(f"The number of commits is: {len(commit_data)}")

file_pairs = []
for commit in commit_data:
    for x in commit["files"]:
        for y in commit["files"]:
            if x != y and not x.endswith('.txt') and not y.endswith('.txt'):
                file_pairs.append((x, y))

print(f"The number of file pairs is: {len(file_pairs)}")

pd.DataFrame(file_pairs).to_csv(os.path.join(DATASET_LOCATION, "file_pairs.csv"), index=False)