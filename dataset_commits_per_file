import json
import os

DATASET_LOCATION = "./data/"
with open(os.path.join(DATASET_LOCATION, "commits.json"), "r") as f:
    data = json.load(f)


Dic = {}

#initializing dictionary with values as list
for obj in data:
    for file in obj["files"]:
        Dic[file] = []

#adding commits for a file in dictionary
for obj in data:
    for file in obj["files"]:
        Dic[file].append(obj["commit"])

#converting dict to list of json
jsonList = []
for file,commitList in Dic.items():
    temp = {"fileName" : file,"commits":commitList}
    jsonList.append(temp)

#writing to file
with open("data/commits_per_file.json",'w') as f:
    json.dump(jsonList,f,indent=4)