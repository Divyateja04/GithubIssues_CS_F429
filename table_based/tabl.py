import json
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

with open("../data/commits.json", "r") as f:
    data = json.load(f)
    
for elem in data:
    elem["files"] = ", ".join(elem["files"])
    if elem["author"] is None:
        elem["author"] = "Unknown"
    
query = "What files can be used for adding valid Repository visibility value?"
# Have to iterate through all slices of 25
# since the model can only handle 25 rows at a time
# (it's a limitation of the model)

# keep track of best answers
best_answers = {}
for i in range(0, len(data), 25):
    try:
        curr_data = data[i:i+25]
        
        table = pd.DataFrame.from_dict(curr_data)
        # tapex accepts uncased input since it is pre-trained on the uncased corpus
        encoding = tokenizer(table=table, query=query, return_tensors="pt", max_new_tokens=1024)
        outputs = model.generate(**encoding, max_length=1024, num_beams=5, early_stopping=True)
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
       
        # Split string based on the comma
        answer = answer.split(",")
        for ans in answer:
            if ans in best_answers.keys():
                best_answers[ans] += 1
            else:
                best_answers[ans] = 1
    except:
        pass

# print top 5 answers
print(sorted(best_answers.items(), key=lambda x: x[1], reverse=True)[:5])
    

