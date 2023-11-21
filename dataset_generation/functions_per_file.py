import os
import ast
import json

def extract_function_names(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    function_names = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_names += node.name
            function_names += ", "
    
    return function_names

def process_folder(folder_path):
    
    file_functions = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                function_names = ""
                # file_functions[file] = ""
                file_path = os.path.join(root, file)
                function_names = extract_function_names(file_path)
                file_functions[file] = function_names

    return file_functions

if __name__ == "__main__":
    folder_path = "C:\\Users\\kumar\\VSC_Projects\\NLP\\GithubIssues_CS_F429\\PyGithub\\github"
    functions_per_file = process_folder(folder_path)

    print("\nFunction names:")
    for name in functions_per_file:
        print("file: " + name + "  functions: " + functions_per_file[name] + "\n")
jsonList = []

f = open("C:\\Users\\kumar\\VSC_Projects\\NLP\\GithubIssues_CS_F429\\data\\functions_per_file.json", "w+", encoding="utf-8")

for fileName in functions_per_file:
    temp = {"fileName" : fileName,"commits":functions_per_file[fileName]}
    jsonList.append(temp)


#writing to file
with open("C:\\Users\\kumar\\VSC_Projects\\NLP\\GithubIssues_CS_F429\\data\\functions_per_file.json",'w') as f:
    json.dump(jsonList,f,indent=4)