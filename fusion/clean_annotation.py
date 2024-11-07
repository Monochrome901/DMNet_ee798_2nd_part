import os

# Define the paths to your images and text files
image_folder = "dataset/Density/val/images"
text_folder = "dataset/Density/val/ground_annotations"

# Function to check if a text file is empty (excluding empty lines)
def is_text_file_empty(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove lines that are empty or contain only spaces
        lines = [line.strip() for line in lines if line.strip()]
        return len(lines) == 0

# Function to clean the text file by removing empty lines
def clean_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Remove empty lines (lines that are just whitespace)
        cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    # Write the cleaned lines back to the file
    with open(file_path, 'w') as file:
        file.write('\n'.join(cleaned_lines))

# Process files
for text_file in os.listdir(text_folder):
    # Check if the corresponding image exists
    image_file = text_file.replace('.txt', '.jpg')  # Assuming image files are .jpg
    text_file_path = os.path.join(text_folder, text_file)
    image_file_path = os.path.join(image_folder, image_file)

    # if is_text_file_empty(text_file_path):
    #     # Remove the text file and its corresponding image if the text file is empty
    #     os.remove(text_file_path)
    #     if os.path.exists(image_file_path):
    #         os.remove(image_file_path)
    #     print(f"Removed {text_file} and {image_file} (empty text file).")
    # else:
    # Clean the text file by removing empty lines between annotations
    clean_text_file(text_file_path)
    print(f"Cleaned {text_file} (removed empty lines).")
