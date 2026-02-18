import argparse
import os
from pathlib import Path

def merge_text_files(folder_path):
    """
    Merges all .txt files in the specified folder into a single file named full_analysis.txt.
    """
    # Convert string path to a Path object for easier manipulation
    directory = Path(folder_path)
    
    if not directory.is_dir():
        print(f"Error: The path '{folder_path}' is not a valid directory.")
        return

    output_filename = "full_analysis.txt"
    output_path = directory / output_filename
    
    # Get all .txt files in the directory, excluding the output file itself
    skip_files = {output_filename, "processed_feature_columns.txt"}
    txt_files = sorted([f for f in directory.glob("*.txt") if f.name not in skip_files])

    if not txt_files:
        print(f"No .txt files found in '{folder_path}'.")
        return

    print(f"Found {len(txt_files)} files. Merging...")

    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(txt_files):
                filename = file_path.name
                
                # Write the separator and title
                # We add a newline before the separator for all files except the first one
                if i > 0:
                    outfile.write("\n\n")
                
                outfile.write("-" * 27 + "\n")
                outfile.write(f"TITLE: {filename}\n")
                outfile.write("-" * 27 + "\n\n")

                # Read and write the content of the current file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        outfile.write(infile.read())
                except Exception as e:
                    outfile.write(f"[Error reading file {filename}: {e}]\n")
                
        print(f"Successfully merged files into: {output_path}")

    except Exception as e:
        print(f"An error occurred while writing the output file: {e}")

def main():
    # Set up CLI argument parsing
    parser = argparse.ArgumentParser(
        description="Merge all .txt files in a folder into a single 'full_analysis.txt' file."
    )
    parser.add_argument(
        "folder", 
        type=str, 
        help="The path to the folder containing .txt files"
    )

    args = parser.parse_args()
    
    # Run the merger
    merge_text_files(args.folder)

if __name__ == "__main__":
    main()