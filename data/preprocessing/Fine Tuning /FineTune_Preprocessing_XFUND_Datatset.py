# Import necessary libraries

import json # For loading JSON files that contain annotations
import os # For file operations like reading file names from a directory


# Convert XFUNSD Dataset to a LiLT fine-tuning compatible format

# Define base paths
BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/XFUND/XFUND"
OUTPUT_BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/XFUND/XFUND_LiLT_Format"

# List of languages in XFUND
LANGUAGES = ["ja", "it", "pt", "zh", "de", "fr", "es"]

# Define label mapping for Named Entity Recognition (NER) based on the XFUNSD dataset categories. This mapping is not used directly in near tagging, but is kept here for reference

LABEL_MAPPING = {
    "question": 1, # Questions in the form 
    "answer": 2, # Answers corresponding to the questions
    "header": 3, # Section headers
    "other": 0 # Any other text that doesn't fit the above categories
}

def load_annotations(json_path):
    ''' 
    Loads annotations from a JSON file

    Parameters:
        json_path (str): Path to the JSON file containing annotations

    Returns:
        list: A list of annotation data if successful
        None: If the file does not exist, has an invalid format, or cannot be loaded
    '''
    # Check if the file exists
    if not os.path.exists(json_path): 
        print(f"Warning: Annotation file {json_path} not found.") # Alert if file is missing
        return None
    
    # Open the file and read the contents
    with open(json_path, "r", encoding="utf-8") as f:
        try: 
            data = json.load(f) # Load JSON content

            if isinstance(data, dict): # Handle case where JSON is a dictionary 
                data = [data]  # Convert to list for consistency 
              
            if not isinstance(data, list): # Ensure the expected list format
                print(f"Error: Unexpected JSON format in {json_path}.")
                return None
            
            return data # Return parsed data
        
        except json.JSONDecodeError as e: # Handle JSON parsing errors
            print(f"Error loading JSON file {json_path}: {e}.") # Print error message
            return None


def convert_dataset(lang, dataset_type):
    ''' 
    Converts XFUND dataset to a LiLT fine-tuning compatible format

    Parameters:
        lang (str): Language of the dataset
        dataset_type (str): Type of dataset (e.g., 'training_data' or 'validation_data')

    Returns:
        None: The function processes and saves converted annotation files
    '''

    print(f"Processing {lang} - {dataset_type}...") # Processing start
    
    # Define paths
    json_path = os.path.join(BASE_PATH, lang, dataset_type, "annotations", f"{lang}.{'train' if dataset_type == 'training_data' else 'val'}.json")
    output_folder = os.path.join(OUTPUT_BASE_PATH, lang, dataset_type)
    os.makedirs(output_folder, exist_ok=True) # Ensure output directory exists
    
    # Load annotations
    annotations = load_annotations(json_path)
    if not annotations: # Check if annotations are loaded successfully
        return
    
    for entry in annotations: # Iterate over each entry (file) in the annotations
        if not isinstance(entry, dict): # Check if the entry is a dictionary
            print(f"Warning: Skipping malformed entry in {json_path}.") # Skip malformed entries
            continue
        
        for doc in entry.get("documents", []): # Iterate over each document in the entry
            if not isinstance(doc, dict): # Check if the document is a dictionary
                print(f"Skipping invalid document format in {json_path}.") # Skip invalid documents
                continue
            
            # Get the document ID using the 'id' field and default to 'unknown' if missing
            doc_id = doc.get("id", "unknown") 

            # Initialize the converted data structure 
            converted_data = { 
                "id": doc_id, # Extract ID from filename (remove extension)
                "words": [], # List to store all words in the document
                "bboxes": [], # List of bounding boxes corresponding to words
                "ner_tags": [] # List of Named Entity Recognition (NER) labels for each word
            }

            # Create a mapping between block IDs and their corresponding word indices
            word_counter = 0 

            block_to_word_idx = {} # Mapping from block ID to word indices
            
            for obj in doc.get("document", []): # Process each object in the document
                if not isinstance(obj, dict): # Check if the object is a dictionary
                    print(f"Warning: Skipping invalid entry in document {doc.get('id', 'unknown')}") # Skip invalid entries
                    continue 
                
                label = obj.get("label", "other") # Extract label, default to "other" if missing
                block_to_word_idx[obj["id"]] = word_counter  # Map block ID to current word index
                
                # Process each word in the object and assign near tag labels
                for idx, word in enumerate(obj.get("words", [])): # Iterate over each word in the object
                    if not isinstance(word, dict): # Check if the word is a dictionary
                        print(f"Warning: Skipping invalid word entry in document {doc.get('id', 'unknown')}") # Skip invalid words
                        continue # Skip to the next word
                    
                    text = word.get("text", "").strip() # Extract text content of the word
                    bbox = word.get("box", []) # Extract bounding box coordinates of the word
                    
                    if text: # Check if the word text is not empty
                        converted_data["words"].append(text) # Store the word text
                        converted_data["bboxes"].append(bbox) # Store the bounding box coordinates
                        # Apply near tag scheme:
                        # - For 'question': first token = 1, subsequent tokens = 2
                        # - For 'answer': first token = 3, subsequent tokens = 4
                        # - For 'header': first token = 5, subsequent tokens = 6
                        # - For 'other': label remains 0 (default)
                        if label == "question": # Assign near tag based on label
                            ner_tag = 1 if idx == 0 else 2 # First token = 1, subsequent tokens = 2
                        elif label == "answer": # Assign near tag based on label
                            ner_tag = 3 if idx == 0 else 4 # First token = 3, subsequent tokens = 4
                        elif label == "header": # Assign near tag based on label
                            ner_tag = 5 if idx == 0 else 6 # First token = 5, subsequent tokens = 6
                        else: # Default to 0 for "other" label
                            ner_tag = 0 # Default to 0 for "other" label

                        # Store the NER label    
                        converted_data["ner_tags"].append(ner_tag) # Store the NER label
                        word_counter += 1 # Increment word index
                    else: # Handle empty word text
                        print(f"Warning: Empty word text in document {doc.get('id', 'unknown')}") # Alert about empty word text
                
            # Removed conversion of linking relationships since 'relations' are not required
            
            # Save converted file
            output_file = os.path.join(output_folder, f"{doc_id}.json") # Output file path
            with open(output_file, "w", encoding="utf-8") as f: # Open the output file
                json.dump(converted_data, f, indent=4) # Write the converted data to the file
            print(f"Saved: {output_file}") # Confirmation message

# Convert all languages for both training and testing data
for lang in LANGUAGES: # Process each language
    convert_dataset(lang, "training_data") # Convert training data
    convert_dataset(lang, "testing_data") # Convert testing data

print("All languages processed successfully!") # Completion message