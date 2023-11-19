import json
import os
# Github API v3
from github import Github
# Authentication is defined via github.Auth
from github import Auth
# Import pandas to create a dataframe
import pandas as pd
# Import tqdm
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


# using an access token
auth = Auth.Token(os.getenv("GITHUB_TOKEN"))

# First create a Github instance:
# Public Web Github
g = Github(auth=auth)

# Get the repository that you want to get the data from
repo = g.get_repo("PyGithub/PyGithub")

file_pairs = []
commit_messages = []

last_processed_commit = 0

try:
    # Iterate per commit
    commits = repo.get_commits()
    for i in tqdm(range(last_processed_commit, repo.get_commits().totalCount)):
        last_processed_commit = i
        commit = commits[i]
        # Get the commit message
        commit_message = commit.commit.message
        # remove new lines
        commit_message = commit_message.replace("\r\n", " ").replace("\n", " ")
        # Append the commit message to the list
        # commit_messages.append(commit_message)
        # Keep track of the files in the commit
        files = []
        # Iterate per file in the commit
        for file in commit.files:
            # Append the filename to the list
            files.append(file.filename)

        commit_object = {
            "commit": commit_message,
            "files": files
        }

        commit_messages.append(commit_object)
    # Save the commits to a json file
    with open('data/commits.json', 'w') as f:
        json.dump(commit_messages, f, indent=4)

except Exception as e:
    print(e)
    print("Last processed commit: ", last_processed_commit)

# To close connections after use
g.close()
