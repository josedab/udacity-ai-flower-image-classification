import json

def get_category_mappings(category_mappings_file_path):
    if(len(category_mappings_file_path)==0):
        raise Exception("Category mappings is empty. Please specify a category mappings file.")

    with open(category_mappings_file_path, 'r') as f:
        category_mappings = json.load(f)

    return category_mappings