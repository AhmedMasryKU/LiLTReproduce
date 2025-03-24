# Import necessary libraries
import json # For loading and parsing JSON files
import os # For file operations like reading directory contents
import random # For randomly selecting files to test

# Define the base output path and the languages used in XFUND
OUTPUT_BASE_PATH = "/Users/yahia/Downloads/FinalProject/FineTuning/Datasets/XFUND/XFUNDFinalFormat"
LANGUAGES = ["ja", "it", "pt", "zh", "de", "fr", "es"] # List of languages in the XFUND dataset
DATASET_TYPES = ["training_data", "testing_data"] # Types of datasets to process (training and testing)

def validate_output_format(data):
    '''
    Validates the structure of the processed JSON file

    Parameters:
        data (dict): Parsed JSON data from the output file

    Returns:
        bool: True if the structure is correct, False otherwise
    '''
    # Required keys in the output JSON file
    required_keys = ["id", "words", "bboxes", "ner_tags"]
    for key in required_keys:
        # Check if each required key is present in the JSON data
        if key not in data:
            print(f"Error: Missing key '{key}' in processed JSON file") # Alert if a key is missing
            return False
    return True # Return True if all required keys are present

def validate_bounding_boxes(bboxes):
    '''
    Validates bounding box normalization to ensure coordinates are within the expected range [0, 1000]

    Parameters:
        bboxes (list): List of bounding box coordinates

    Returns:
        bool: True if all bounding boxes are normalized correctly, False otherwise
    '''
    for bbox in bboxes:
        # Check that each bounding box has exactly 4 elements (x_min, y_min, x_max, y_max)
        if len(bbox) != 4:
            print(f"Error: Bounding box does not have 4 elements: {bbox}") # Alert about incorrect bbox length
            return False
        # Check that each coordinate in the bounding box is within the range [0, 1000]
        for coord in bbox:
            if coord < 0 or coord > 1000:
                print(f"Error: Bounding box coordinate out of range [0, 1000]: {bbox}") # Alert if any coordinate is out of range
                return False
    return True  # Return True if all bounding boxes are normalized correctly

def validate_ner_tags(words, ner_tags):
    '''
    Validates the NER tagging to ensure that the number of tags matches the number of words

    Parameters:
        words (list): List of words in the document
        ner_tags (list): List of NER tags corresponding to the words

    Returns:
        bool: True if the number of words and NER tags are consistent, False otherwise
    '''
    # Check that the number of words matches the number of NER tags
    if len(words) != len(ner_tags):
        print(f"Error: Mismatch between the number of words ({len(words)}) and NER tags ({len(ner_tags)})") # Alert if counts differ
        return False
    return True # Return True if the number of words matches the number of NER tags

def test_processed_files():
    '''
    Tests the processed JSON files for correct format, bounding box normalization, and NER tagging

    Iterates through each language and dataset type to validate processed files
    Aggregates validation results and prints a summary at the end

    Returns:
        None: Prints validation results for each file
    '''
    # Initialize counters for statistics
    total_files = 0 # Total number of files checked
    valid_files = 0 # Total number of files that passed validation
    errors = 0 # Total number of files with errors

    # Iterate over each language and dataset type to validate files
    for lang in LANGUAGES:
        for dataset_type in DATASET_TYPES:
            # Construct the output folder path for the current language and dataset type
            output_folder = os.path.join(OUTPUT_BASE_PATH, lang, dataset_type)
            if not os.path.exists(output_folder): # Check if the folder exists
                print(f"Warning: Folder '{output_folder}' not found.") # Alert if folder is missing
                continue
            
            # List all JSON files in the output folder
            files = [f for f in os.listdir(output_folder) if f.endswith(".json")]
            total_files += len(files) # Update total file count

            for file_name in files:
                file_path = os.path.join(output_folder, file_name)
                try:
                    # Open and load the JSON file
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Validate the output format (checks for required keys)
                        if not validate_output_format(data):
                            errors += 1 # Increment error count
                            continue
                        
                        # Validate bounding box normalization (check range and format)
                        if not validate_bounding_boxes(data["bboxes"]):
                            errors += 1 # Increment error count
                            continue

                        # Validate NER tagging consistency (check word and tag count)
                        if not validate_ner_tags(data["words"], data["ner_tags"]):
                            errors += 1 # Increment error count
                            continue

                        # Increment the count of valid files if all checks pass
                        valid_files += 1
                except Exception as e:
                    # Handle any errors during file reading or processing
                    errors += 1 # Increment error count
                    print(f"Error processing file '{file_name}': {e}") # Print error message

    # Print the summary of validation results
    print("\n--- Validation Summary ---")
    print(f"Total files checked: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Errors: {errors}")

# Run the test script to validate processed files
test_processed_files()