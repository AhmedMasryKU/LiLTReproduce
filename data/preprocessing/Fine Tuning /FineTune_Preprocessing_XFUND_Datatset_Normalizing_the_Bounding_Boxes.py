# Import necessary libraries

import json # For loading JSON files that contain annotations
import os # For file operations like reading file names from a directory
from PIL import Image # For opening images to get dimensions


# Convert XFUNSD Dataset to a LiLT fine-tuning compatible format with normalized bounding boxes

# Define base paths
BASE_PATH = "/Users/yahia/Downloads/FinalProject/FineTuning/Datasets/XFUND/XFUNDDataset"

# Define output base path
OUTPUT_BASE_PATH = "/Users/yahia/Downloads/FinalProject/FineTuning/Datasets/XFUND/XFUNDFinalFormat"

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
                data = [data] # Convert to list for consistency 
              
            if not isinstance(data, list): # Ensure the expected list format
                print(f"Error: Unexpected JSON format in {json_path}.")
                return None
            
            return data # Return parsed data
        
        except json.JSONDecodeError as e: # Handle JSON parsing errors
            print(f"Error loading JSON file {json_path}: {e}.") # Print error message
            return None

def convert_dataset(lang, dataset_type):
    ''' 
    Converts XFUND dataset to a LiLT fine-tuning compatible format with normalized bounding boxes

    This function processes the annotations, applies a near tag labelling scheme:
      - For 'question': first token = 1, subsequent tokens = 2
      - For 'answer': first token = 3, subsequent tokens = 4
      - For 'header': first token = 5, subsequent tokens = 6
      - For 'other': label remains 0 (default)
      
    It also normalizes the bounding boxes using the dimensions of the corresponding image so that 
    each coordinate is scaled between 0 and 1000 which is necessary for the LiLT fine-tuning 

    Parameters:
        lang (str): Language of the dataset
        dataset_type (str): Type of dataset (e.g., 'training_data' or 'testing_data')

    Returns:
        None: The function processes and saves converted annotation files
    '''

    print(f"Processing {lang} - {dataset_type}...") # Processing start
    
    # Define paths for annotations and images
    json_path = os.path.join(BASE_PATH, lang, dataset_type, "annotations", f"{lang}.{'train' if dataset_type == 'training_data' else 'val'}.json") # Annotation file path
    img_folder = os.path.join(BASE_PATH, lang, dataset_type, "images") # Image folder path
    output_folder = os.path.join(OUTPUT_BASE_PATH, lang, dataset_type) # Output folder path
    os.makedirs(output_folder, exist_ok=True) # Ensure output directory exists
    
    # Load annotations
    annotations = load_annotations(json_path) # Load annotations from the JSON file
    if not annotations: # Check if annotations are loaded successfully
        return
    
    for entry in annotations: # Iterate over each entry in the annotations
        if not isinstance(entry, dict): # Check if the entry is a dictionary 
            print(f"Warning: Skipping malformed entry in {json_path}.") # Skip malformed entries
            continue
        
        for doc in entry.get("documents", []): # Iterate over each document in the entry
            if not isinstance(doc, dict): # Check if the document is a dictionary
                print(f"Skipping invalid document format in {json_path}.") # Skip invalid documents
                continue
            
            # Get the document ID using the 'id' field and default to 'unknown' if missing
            doc_id = doc.get("id", "unknown")
            
            # Construct image file path
            image_filename = f"{doc_id}.jpg" # image filename is the same as the document ID in JPG format
            image_path = os.path.join(img_folder, image_filename) # Full path to the image file
            if not os.path.exists(image_path): # Check if the image file exists
                print(f"Warning: Image {image_filename} not found, skipping.") # Alert if image is missing
                continue # Skip the document if image is missing
            
            # Open image to get dimensions for normalization
            try:
                with Image.open(image_path) as img: # Open the image file
                    width, height = img.size # Get the dimensions of the image
            except Exception as e: # Handle image opening errors
                print(f"Error opening image {image_path}: {e}") # Print error message
                continue 
            
            # Initialize the converted data structure 
            converted_data = { 
                "id": doc_id, # Extract ID from filename (remove extension)
                "words": [], # List to store all words in the document
                "bboxes": [], # List of normalized bounding boxes for each word
                "ner_tags": [] # List of Named Entity Recognition (NER) labels for each word (using near tag scheme)
            }
            
            word_counter = 0  # Counter to keep track of word indices
            
            # Process each object in the document
            for obj in doc.get("document", []): # Iterate over each object in the document
                if not isinstance(obj, dict): # Check if the object is a dictionary
                    print(f"Warning: Skipping invalid entry in document {doc.get('id', 'unknown')}.") # Skip invalid entries
                    continue   
                
                # Get the label; default to 'other' if missing
                label = obj.get("label", "other") 
                
                # Process each word in the object with near tag labelling
                for idx, word in enumerate(obj.get("words", [])): # Iterate over each word in the object
                    if not isinstance(word, dict): # Check if the word is a dictionary
                        print(f"Warning: Skipping invalid word entry in document {doc.get('id', 'unknown')}.") # Skip invalid words
                        continue
                    
                    text = word.get("text", "").strip() # Extract and clean text
                    bbox = word.get("box", []) # Extract bounding box coordinates
                    
                    if text:
                        converted_data["words"].append(text) # Store the word text
                        
                        # Normalize the bounding box if it contains 4 coordinates
                        if bbox and len(bbox) == 4:
                            # bbox format: [x_min, y_min, x_max, y_max]
                            norm_bbox = [ # Normalize bounding box coordinates
                                int(round(bbox[0] / width * 1000)), # Scale x_min to 0-1000
                                int(round(bbox[1] / height * 1000)), # Scale y_min to 0-1000
                                int(round(bbox[2] / width * 1000)), # Scale x_max to 0-1000
                                int(round(bbox[3] / height * 1000)) # Scale y_max to 0-1000
                            ] 
                        else:
                            norm_bbox = bbox # Use as is if not in expected format
                        converted_data["bboxes"].append(norm_bbox) # Store the normalized bounding box
                        
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
                            ner_tag = 0  # Default for "other"

                        # Store the NER label    
                        converted_data["ner_tags"].append(ner_tag) # Store the NER label
                        
                        word_counter += 1 # Increment the word counter
                    else: # Handle empty word text
                        print(f"Warning: Empty word text in document {doc.get('id', 'unknown')}.") # Alert about empty word text
            
            # Save the converted file in JSON format
            output_file = os.path.join(output_folder, f"{doc_id}.json") # Output file path for the converted data
            with open(output_file, "w", encoding="utf-8") as f: # Open the output file
                json.dump(converted_data, f, indent=4) # Write the converted data to the file
            print(f"Saved: {output_file}") # Confirmation message

# Convert all languages for both training and testing data
for lang in LANGUAGES: # Process each language
    convert_dataset(lang, "training_data") # Convert training data
    convert_dataset(lang, "testing_data") # Convert testing data

print("All languages processed successfully!") # Completion message