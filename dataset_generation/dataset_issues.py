import os
import json
# Github API v3
from github import Github
# Authentication is defined via github.Auth
from github import Auth
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

issues = []

# Iterate per issue
for i in tqdm(range(repo.get_issues().totalCount)):
    issue = repo.get_issues()[i]
    # Get the issue title
    issue_obj = []
    # Append the title and body to the list
    issue_obj.append(issue.title.replace("\r\n", " ").replace("\n", " "))
    issue_obj.append(issue.body.replace("\r\n", " ").replace("\n", " "))
    # Append the issue to the list
    issues.append(issue_obj)

# Save the issues to a json file
with open('data/issues.json', 'w') as f:
    json.dump(issues, f, indent=4)

# To close connections after use
g.close()
