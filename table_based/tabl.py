import json
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wikisql")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-base-finetuned-wikisql")

with open("../data/commits.json", "r") as f:
    data = json.load(f)
    
data = data[:20]
    
# Convert every list in the dictionary to a string
for elem in data:
    elem["files"] = ", ".join(elem["files"])
    if elem["author"] is None:
        elem["author"] = "Unknown"
    
table = pd.DataFrame.from_dict(data)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "What file do I have to edit to add support for issue comments?"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
