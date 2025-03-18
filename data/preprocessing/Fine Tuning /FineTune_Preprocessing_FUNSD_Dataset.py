# Import necessary libraries

import json  # For loading JSON files that contain annotations
import os    # For file operations like reading file names from a directory
import cv2   # For image corresponding to the annotation processing: drawing bounding boxes, etc           
import matplotlib.pyplot as plt  # For displaying images and drawing bounding boxes on the images


# Loading and Viewing FUNSD Dataset Annotations

# The FUNSD dataset contains annotations in JSON format

# Path to FUNSD dataset
FUNSD_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/FUND/dataset/training_data/annotations" 

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
image_path = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/FUND/dataset/training_data/images/00040534.png"

# Check if the image file exists and load it 
if not os.path.exists(image_path): 
    print(f"Error: File '{image_path}' not found!")
else:
    image = cv2.imread(image_path)  # Read the image
    if image is None:
        print("Error: cv2.imread() failed to load the image.")
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib
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


# Convert FUNSD Dataset to LiLT Hugging Face Compatible Format

# Define label mapping for Named Entity Recognition (NER) based on the FUNSD dataset categories
'''
The mapping is used to determine the type and not the final numeric tag.
The final numeric tag is determined by the near tag scheme in the conversion function.
''' 
LABEL_MAPPING = {
    "question": "question",  # Questions in the form 
    "answer": "answer",      # Answers corresponding to the questions
    "header": "header",      # Section headers
    "other": "other"         # Any other text that doesn't fit the above categories
}

def convert_dataset(input_path, image_folder, output_path):
    """
    Converts FUND dataset to a LiLT fine-tuning compatible format

    Parameters:
    - input_path (str): Path to the folder containing the FUNSD dataset JSON annotation files
    - image_folder (str): Path to the folder containing the corresponding scanned document images
    - output_path (str): Path where the converted JSON files should be saved

    The function extracts text, bounding boxes, and NER labels and converts block-level annotations to word-level annotations. 
    The near tag scheme is applied as follows:
      - For 'question': first token = 1, subsequent tokens = 2
      - For 'answer': first token = 3, subsequent tokens = 4
      - For 'header': first token = 5, subsequent tokens = 6
      - For 'other': label remains 0
    """

    # Ensure the output directory exists, create it if necessary
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over each annotation file in the dataset directory
    for filename in os.listdir(input_path):
        if filename.endswith(".json"):  # Process only JSON files
            file_path = os.path.join(input_path, filename)  # Get full path of the annotation file

            # Load the annotation JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Parse JSON content

            # Initialize the converted data structure.
            # Note: The image_path field and relations are removed as they are not required for fine-tuning.
            converted_data = {
                "id": filename.replace(".json", ""),  # Extract ID from filename (remove extension)
                "words": [],      # List to store all words in the document
                "bboxes": [],     # List of bounding boxes corresponding to words
                "ner_tags": []    # List of Named Entity Recognition (NER) labels for each word
            }

            # Create a mapping between block IDs and their corresponding first word index (for potential linking purposes)
            word_id_mapping = {}
            word_counter = 0  # Counter to assign word indices sequentially

            # Process each text block in the "form" field of the JSON data
            for item in data.get("form", []):
                label_text = item.get("label", "other")  # Extract label, default to "other" if missing
                
                # Process each word in the block and assign near tag labels
                for idx, word in enumerate(item.get("words", [])):
                    converted_data["words"].append(word["text"])   # Store the word text
                    converted_data["bboxes"].append(word["box"])     # Store the bounding box coordinates
                    
                    if label_text == "question":  # Assign near tag based on label
                        ner_tag = 1 if idx == 0 else 2  # First token = 1, subsequent tokens = 2
                    elif label_text == "answer":
                        ner_tag = 3 if idx == 0 else 4  # First token = 3, subsequent tokens = 4
                    elif label_text == "header":
                        ner_tag = 5 if idx == 0 else 6  # First token = 5, subsequent tokens = 6
                    else:
                        ner_tag = 0  # Default to 0 for "other" label
                    
                    converted_data["ner_tags"].append(ner_tag)  # Store the NER label

                # Map the block ID to the index of its first word
                word_id_mapping[item["id"]] = word_counter
                word_counter += len(item.get("words", []))

            # Save the converted JSON file
            output_file = os.path.join(output_path, filename)  # Output file path
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(converted_data, f, indent=4)  # Write the converted data to the file

    print(f"Dataset conversion completed. Converted files saved in: {output_path}")

'''
Example Before Conversion:
{
    "form": [
        {
            "id": "block1",
            "label": "question",
            "words": [
                {"text": "What", "box": [10, 10, 50, 30]},
                {"text": "is", "box": [55, 10, 70, 30]},
                {"text": "your", "box": [75, 10, 100, 30]},
                {"text": "name", "box": [105, 10, 140, 30]}
            ],
            "linking": [["block1", "block2"]]
        },
        {
            "id": "block2",
            "label": "answer",
            "words": [
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
    "words": ["What", "is", "your", "name", "John", "Doe"],
    "bboxes": [
        [10, 10, 50, 30],
        [55, 10, 70, 30],
        [75, 10, 100, 30],
        [105, 10, 140, 30],
        [10, 50, 60, 70],
        [65, 50, 120, 70]
    ],
    "ner_tags": [1, 2, 2, 2, 3, 4]
}
'''


# Define base paths
BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/FUND/dataset"  # Path to the FUNSD dataset
OUTPUT_BASE_PATH = "/Users/yahia/Downloads/Final Project /Fine Tuning /Funds Dataset/FUND/FUND_LiLT_Format/dataset"  # Output path for converted dataset

# Get the list of training and testing datasets
dataset_types = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]  # List of dataset types (e.g., training_data, testing_data)

# Loop through training and testing datasets
for dataset_type in dataset_types:
    annotation_path = os.path.join(BASE_PATH, dataset_type, "annotations")  # Path to the annotation folder
    image_path = os.path.join(BASE_PATH, dataset_type, "images")  # Path to the image folder
    output_path = os.path.join(OUTPUT_BASE_PATH, dataset_type)  # Output path for the converted dataset

    # Convert the dataset
    convert_dataset(annotation_path, image_path, output_path)