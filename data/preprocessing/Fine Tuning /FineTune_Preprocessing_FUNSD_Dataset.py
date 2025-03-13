# Import necessary libraries

import json # For loading JSON files that contain annotations
import os # For file operations like reading file names from a directory
import cv2 # For image corresponding to the annotation processing: drawing bounding boxes, etc           
import matplotlib.pyplot as plt # For displaying images and drawing bounding boxes on the images


# Loading and Viewing FUNSD Dataset Annotations


# The FUNSD dataset contains annotations in JSON format

# Path to FUNSD dataset
FUNSD_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/dataset/training_data/annotations" 

# Load a sample annotation file
sample_file = os.path.join(FUNSD_PATH, "00040534.json")  # Path to a sample annotation file

# Check if the annotation file exists 
if not os.path.exists(sample_file): 
    print(f"Error: Annotation file '{sample_file}' not found!")
else:
    with open(sample_file, "r", encoding="utf-8") as f:  # Open the file
        data = json.load(f)
    print("Annotation file loaded successfully!")

# Print JSON structure
print(json.dumps(data, indent=4)) 


# Visualizing the Bounding Boxes on the Image


# Path to the image corresponding to the annotation
image_path = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/dataset/training_data/images/00040534.png"

# Check if the image file exists and laod it 
if not os.path.exists(image_path): 
    print(f"Error: File '{image_path}' not found!")
else:
    image = cv2.imread(image_path) # Read the image
    if image is None:
        print("Error: cv2.imread() failed to load the image.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB for Matplotlib
        print("Image loaded and converted successfully!")

# Draw bounding boxes
for item in data["form"]:
    x1, y1, x2, y2 = item["box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw box
    cv2.putText(image, item["text"], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), 1, cv2.LINE_AA)  # Add text label

# Show image with bounding boxes
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.show()


# Convert FUNSD Dataset to LiLT Hugging Face comptable Format


# Define label mapping for Named Entity Recognition (NER) based on the FUNSD dataset categories

LABEL_MAPPING = {
    "question": 1, # Questions in the form 
    "answer": 2,  # Answers corresponding to the questions
    "header": 3, # Section headers
    "other": 0 # Any other text that doesn't fit the above categories
}

def convert_dataset(input_path, image_folder, output_path): # Function to convert FUNSD dataset to Hugging Face comptable format for fine-tuning
    """
    Converts FUND dataset to a LiLT Hugging Face fine-tuning comptable format 

    Parameters:
    - input_path (str): Path to the folder containing the FUNSD dataset JSON annotation files
    - image_folder (str): Path to the folder containing the corresponding scanned document images
    - output_path (str): Path where the converted JSON files should be saved

    The function extracts text, bounding boxes, and NER labels and converts linking relationships
    from block-level annotations to word-level annotations
    """

    # Ensure the output directory exists, create it if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over each annotation file in the dataset directory
    for filename in os.listdir(input_path):
        if filename.endswith(".json"): # Process only JSON files
            file_path = os.path.join(input_path, filename) # Get full path of the annotation file

            # Load the annotation JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f) # Parse JSON cotent data 

            # Initialize the converted data structure
            converted_data = {
                "id": filename.replace(".json", ""),  # Extract ID from filename (remove extension)
                "words": [], # List to store all words in the document
                "bboxes": [], # List of bounding boxes corresponding to words
                "ner_tags": [], # List of Named Entity Recognition (NER) labels for each word
                "relations": [], # List of relationships between words (linking)
                "image_path": os.path.join(image_folder, filename.replace(".json", ".png")) # Path to the corresponding image
            }

            # Create a mapping between block IDs and their corresponding word indices
            word_id_mapping = {} # Create a mapping between block IDs and their corresponding word indices
            word_counter = 0 # Counter to assign word indices sequentially

            # Process each text block in the "form" field of the JSON data
            for item in data.get("form", []): # Default to an empty list if "form" is missing
                label_text = item.get("label", "other") # Extract label, default to "other" if missing
                label_id = LABEL_MAPPING.get(label_text, 0) # Convert label to numeric value

                for word in item.get("words", []): # Process each word in the block
                    converted_data["words"].append(word["text"]) # Store the word text
                    converted_data["bboxes"].append(word["box"]) # Store the bounding box coordinates
                    converted_data["ner_tags"].append(label_id) # Store the NER label
                    
                    # Store word index for linking
                    word_id_mapping[item["id"]] = word_counter # Map block ID to word index
                    word_counter += 1 # Increment word index

            # Convert linking relationships
            for item in data.get("form", []): # Process each text block in the "form" field
                for link in item.get("linking", []): # Process each linking relationship
                    if link[0] in word_id_mapping and link[1] in word_id_mapping: # Check if both IDs are in the mapping
                        converted_data["relations"].append([word_id_mapping[link[0]], word_id_mapping[link[1]]]) # Add the relationship

            # Save the converted JSON file
            output_file = os.path.join(output_path, filename) # Output file path
            with open(output_file, "w", encoding="utf-8") as f: # Open the output file
                json.dump(converted_data, f, indent=4) # Write the converted data to the file

    print(f"Dataset conversion completed. Converted files saved in: {output_path}")

'''
Example Before Conversion:
{
    "form": [
        {
            "id": "block1", "label": "question", "words": [
                {"text": "What", "box": [10, 10, 50, 30]},
                {"text": "is", "box": [55, 10, 70, 30]}
            ],
            "linking": [["block1", "block2"]]
        },
        {
            "id": "block2", "label": "answer", "words": [
                {"text": "John", "box": [10, 50, 60, 70]},
                {"text": "Doe", "box": [65, 50, 120, 70]}
            ],
            "linking": [["block2", "block1"]]
        }
    ]
}

Example After Conversion:
{
    "id": "sample",
    "words": ["What", "is", "John", "Doe"],
    "bboxes": [
        [10, 10, 50, 30], [55, 10, 70, 30],
        [10, 50, 60, 70], [65, 50, 120, 70]
    ],
    "ner_tags": [1, 1, 2, 2],
    "relations": [[0, 2], [1, 3]],
    "image_path": "image_folder/sample.png"
}
'''


# Define base paths
BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/dataset"
OUTPUT_BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/FUND_LiLT_Format/dataset"

# Get the list of training and testing datasets
dataset_types = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]

# Loop through training and testing datasets
for dataset_type in dataset_types:
    annotation_path = os.path.join(BASE_PATH, dataset_type, "annotations")
    image_path = os.path.join(BASE_PATH, dataset_type, "images")
    output_path = os.path.join(OUTPUT_BASE_PATH, dataset_type)

    # Convert the dataset
    convert_dataset(annotation_path, image_path, output_path)