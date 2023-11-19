# Importing the Libraries
import json
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from skipgram.skipgram_test import similarity_check

print("Initialization complete\n")

print("Hello there, what would you like to do today?")
print("1. Create a new issue")
print("2. Check most recent summary")
print("3. Find file correlations")

while True:
    user_input = input("\nEnter your choice: ")
    
    if user_input == "1":
        print("You have chosen to create a new issue")
        print("Please enter the following information")
        description = input("Description: ")
        print("Thank you, your issue has been created")
        break
    
    if user_input == "2":
        print("You have chosen to check the most recent summary")
        print("Here is the most recent summary")
        print("Summary: ")
        break
    
    if user_input == "3":
        print("You have chosen to find file correlations")    
        file_name = input("Enter file name: ")

        sims = similarity_check(file_name)

        if len(sims) > 0:
            print("Similarity scores: ")
            for sim in sims:
                print(sim[0] + " " + str(sim[1]))
        else:
            print("No similar files found")


