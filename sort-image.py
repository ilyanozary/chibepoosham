import os
# Define the folder path and the start number
folder_path = "evaluation_dataset/Images_Blurred"
start_number = 1

# Loop through the files in the folder
for file in os.listdir(folder_path):
    # Get the file extension
    file_ext = os.path.splitext(file)[1]
    # Construct the new file name with the start number and the extension
    new_file_name = str(start_number) + file_ext
    # Rename the file using the os.rename function
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file_name))
    # Increment the start number by one
    start_number += 1

