import os

# Define paths
label_dirs = ["train/labels", "valid/labels", "test/labels"]

def preprocess_labels(label_dir):
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        
        # Read the label file
        with open(file_path, "r") as f:
            lines = f.readlines()
        
        # Process each line
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:  # Skip empty lines
                continue
            
            # Replace the first number (class) with 0
            parts[0] = "0"
            new_lines.append(" ".join(parts) + "\n")
        
        # Remove trailing empty lines
        if new_lines:
            new_lines[-1] = new_lines[-1].rstrip()  # Remove newline from the last line
        
        # Write the cleaned and remapped lines back to the file
        with open(file_path, "w") as f:
            f.writelines(new_lines)

# Process all label directories
for label_dir in label_dirs:
    if os.path.exists(label_dir):  # Check if the directory exists
        preprocess_labels(label_dir)
        print(f"Processed {label_dir}")
    else:
        print(f"Directory {label_dir} does not exist. Skipping.")

print("Preprocessing completed successfully!")
