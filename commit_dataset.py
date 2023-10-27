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

# Iterate per commit
for i in tqdm(range(repo.get_commits().totalCount)):
    commit = repo.get_commits()[i]
    # Get the commit message
    commit_message = commit.commit.message
    # remove new lines
    commit_message = commit_message.replace("\r\n", " ").replace("\n", " ")
    # Append the commit message to the list
    commit_messages.append(commit_message)
    # Keep track of the files in the commit
    files = []
    # Iterate per file in the commit
    for file in commit.files:
        # Append the filename to the list
        files.append(file.filename)

    # Create pairs between all the files for correlations
    for i in range(len(files)):
        for j in range(len(files)):
            if files[i] != files[j]:
                file_pairs.append((files[i], files[j]))

    pd.DataFrame(file_pairs).to_csv(
        "data/file_pairs.csv", index=False, header=None)
    pd.DataFrame(commit_messages).to_csv(
        "data/commit_messages.csv", index=False, header=None)

# To close connections after use
g.close()
