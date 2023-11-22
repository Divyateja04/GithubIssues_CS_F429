#from keybert import KeyBERT
import json
import os
#from tqdm import tqdm
DATASET_LOCATION = "./data/"
with open(os.path.join(DATASET_LOCATION, "functions_per_file.json"), "r") as f:
    data = json.load(f)





# test = "Hello im under the water please help me"
# output = []
# output_object = {}
# for issue in data[:10]:
#     input = ""
#     if(issue["title"] != None):
#         input += issue["title"]
#     input +=" "
#     if(issue["body"]!=None):
#         input += issue["body"]
#     issue["keywords"] =  kw_model.extract_keywords(input,keyphrase_ngram_range=(1, 1), top_n=2)

# print(data[0]["functions"].split(','))
for file in data:
    # new_function = []
    functions = file["functions"].split(',')
    for i in range(len(functions)):
        func = functions[i]
        if("_" in func):
            temp = func.split("_")
            functions[i] = " ".join(temp)
        func = functions[i]
        for j in range(len(func)):
            if(ord(func[j])>=65 and ord(func[j])<=90 and j!=0):
                functions[i] = func[:j] + " " + func[j:]
                
    file["functions"] = ", ".join(functions)
    # print(file["functions"])

with open("data/functions_per_file(without _ ).json",'w') as f:
    json.dump(data,f,indent=4)