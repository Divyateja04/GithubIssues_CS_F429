# Importing the Libraries
import json
import os
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# Pytorch
import torch
import torch.nn.functional as F
# Transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
# Github API v3
from github import Github
from github import Auth
from skipgram.skipgram_test import similarity_check

print("Initializing...")
load_dotenv()
# ======================CUDA Part======================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ======================Summarizer Part======================
tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer').to(device)
# ======================Github Part======================
# using an access token we create a Github instance and Get the repository 
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))
g = Github(auth=auth)
repo = g.get_repo("PyGithub/PyGithub")

print("Initialization complete\n")

"""
This function takes in a string and returns a summary of the string
"""
def get_summary_response(input_text):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=5000, return_tensors="pt").to(device)
  gen_out = model.generate(**batch,max_length=256,num_beams=5, num_return_sequences=1, temperature=1.5)
  output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
  return output_text

"""
This function calls the get_summary_response function and returns the summary
after retrieving the commits from the repository
"""
def generate_commit_summary():
    print("You have chosen to check the most recent commit summary") 
    # create an array to store the commit messages
    commit_messages = []

    try:
        # Iterate per commit
        commits = repo.get_commits()
        for i in tqdm(range(0, 15)):
            commit = commits[i]
            # Get the commit message
            commit_message = commit.commit.message
            # remove new lines
            commit_message = commit_message.replace("\r\n", " ").replace("\n", " ")
            # remove double quotes and spaces
            commit_message = commit_message.replace("\"", "").replace("  ", "")            
            # Append the commit message to the list
            commit_messages.append(commit_message)
            
        # join all the commit messages into one string
        commits_combination = ", ".join(commit_messages)
        # generate summary
        print("Here is the most recent commit summary: ")
        print(get_summary_response(commits_combination)[0])
        print("\n")
    except Exception as e:
        print("Could not retrieve commit messages")
        print(e)
    

'''
This function uses the skipgram model to find file correlations
'''
def get_file_correlations():
    print("You have chosen to find file correlations")    
    file_name = input("Enter file name: ")

    sims = similarity_check(file_name)

    if len(sims) > 0:
        print("Similarity scores: ")
        for sim in sims:
            print(sim[0] + " " + str(sim[1]))
    else:
        print("No similar files found")

def get_user_report():
    print("You have chosen to give a user report")
    username = input("Enter github username: ")
    user = g.get_user(username)
    print("\n")
    print("User: " + user.name)
    print("Public Repos: " + str(user.public_repos))
    print("Followers: " + str(user.followers))
    print("Following: " + str(user.following))
    print("\n")
        
    commits_per_user = []
    
    # Open commits dataset from json
    with open('data/commits.json') as f:
        data = json.load(f)
    
    # Iterate through commits dataset
    for commit in data:
        # Check if the commit belongs to the user
        if commit["author"] == username:
            commits_per_user.append(commit["commit"])
    try:
        commits_per_user = commits_per_user[:20]
        # join all the commit messages into one string
        commits_per_user_combination = ", ".join(commits_per_user)
        # remove emails from the string to avoid summarizing them
        commits_per_user_combination = commits_per_user_combination.replace("<", "").replace(">", "").replace("#", "").replace("`", "")
        # generate summary
        print(f"Here is the commit report for the user {username}: ")
        print(get_summary_response(commits_per_user_combination)[0])
        print("\n") 
    except Exception as e:
        print(f"Could not generate report: {e}")
    
while True:
    print("Hello there, what would you like to do today?")
    print("1. Create a new issue from description")
    print("2. Check most recent commit summary")
    print("3. Find file correlations")
    print("4. Give user report (Enter github username)")
    print("E. Exit")

    user_input = input("\nEnter your choice: ")
    
    if user_input == "1":
        print("You have chosen to create a new issue")
        print("Please enter the following information")
        description = input("Description: ")
        print("Thank you, your issue has been created")
        break
    
    if user_input == "2":
        generate_commit_summary()
    
    if user_input == "3":
        get_file_correlations()
    
    if user_input == "4":
        get_user_report()        
    
    if user_input == "E":
        # To close github connections after use
        g.close()
        
        print("Thank you for using our program")
        break


