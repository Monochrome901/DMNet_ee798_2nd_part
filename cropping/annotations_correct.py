import os

# Define the folder containing the text files
folder_path = 'dataset/val/annotations'  # Change this to your folder path

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # Process only text files
        file_path = os.path.join(folder_path, filename)

        # Read the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Modify each line
        modified_lines = []
        for line in lines:
            # Split the line by comma, remove the last two elements, and join back
            line_split = line.strip().split(',')
            modified_line = ','.join(line_split[:-2])  # Remove the last two numbers
            modified_lines.append(modified_line)

        # Write the modified lines back to the file (overwrite)
        with open(file_path, 'w') as file:
            file.write('\n'.join(modified_lines) + '\n')

print("Modification complete!")
