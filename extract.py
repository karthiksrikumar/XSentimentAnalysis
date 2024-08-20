import json

# Load the Jupyter Notebook content
notebook_path = '/mnt/data/Twitter_Sentiment_Analysis.ipynb'

with open(notebook_path, 'r') as file:
    notebook_content = json.load(file)

# Extract the content of the cells
notebook_content['cells'][:3]  # Display the first few cells to understand the structure
