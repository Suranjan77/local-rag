import os
from bs4 import BeautifulSoup
import warnings

# Suppress warnings that might appear if parsing non-standard HTML fragments
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

def clean_html_from_text(text):
    """
    Uses BeautifulSoup to extract text and remove HTML tags.
    """
    # 'html.parser' is fast and built-in, but 'lxml' is faster if installed
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def process_folder(folder_path):
    """
    Recursively finds all .md files in the folder and removes HTML tags.
    """
    files_processed = 0
    errors = 0

    print(f"--- Starting cleanup in: {folder_path} ---")

    # os.walk allows us to look into subdirectories as well
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                
                try:
                    # 1. Read the file
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_content = f.read()

                    # 2. Clean the content
                    cleaned_content = clean_html_from_text(original_content)

                    # 3. Optimization: Only write if changes were actually made
                    if original_content != cleaned_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        print(f"cleaned: {file}")
                        files_processed += 1
                    else:
                        # Optional: Verify files that didn't need changes
                        # print(f"skipped (no HTML): {file}")
                        pass

                except Exception as e:
                    print(f"ERROR in {file}: {e}")
                    errors += 1

    print("--- Processing Complete ---")
    print(f"Total files updated: {files_processed}")
    print(f"Total errors: {errors}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Change this to the path of the folder you want to clean
    # You can use "." for the current directory
    TARGET_FOLDER = "/home/sur/repo/local-rag/data/mds/" 
    
    # Check if path exists before running
    if os.path.exists(TARGET_FOLDER):
        process_folder(TARGET_FOLDER)
    else:
        print(f"Error: The folder '{TARGET_FOLDER}' does not exist.")
