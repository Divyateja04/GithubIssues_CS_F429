import pandas as pd
import json

# Dataset
# open file
with open('data/commits.json') as f:
    data = json.load(f)
    
dataset = []

all_files = set()
for commit in data:
    for file in commit['files']:
        if not file.endswith('.txt') and not file.endswith('.rst'):
            all_files.add(file)

print(f"The number of files is: {len(all_files)}")
print(f"The number of commits is: {len(data)}")
print(f"The number of total pairs is: {len(all_files) * len(data)}")

for commit in data:
    for file in list(all_files):
        if file not in commit['files']:
            dataset.append(
                {               
                    'question': "Is " + file + " useful in the commit " + commit['commit'] + "?",
                    "answer": "no"  
                }
            )
        if file in commit['files']:
            dataset.append(
                {
                    'question': "Is " + file + " useful in the commit " + commit['commit'] + "?",
                    "answer": "yes"  
                }
            )
    pd.DataFrame(dataset).to_csv('data/qa_new.csv', index=False)
        