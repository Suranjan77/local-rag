from bs4 import BeautifulSoup
from pathlib import Path
import logging
import re

_log = logging.getLogger(__name__)

def clean_html_from_text(text: str):
    soup = BeautifulSoup(text, "lxml")
    cleaned_text = soup.get_text()

    # 1. Remove lines that only contain the "|" character (and optional whitespace)
    # (?m) enables multi-line mode so ^ and $ match the start/end of lines
    cleaned_text = re.sub(r'(?m)^\s*\|\s*$', '', cleaned_text)
    
    # Replace 2 or more consecutive newlines with exactly 2 newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text

def process_md_files(md_files: list[Path]):
    files_processed = 0
    errors = 0

    for md_file in md_files:
        try:
            with md_file.open('r', encoding='utf-8') as f:
                original_content = f.read()

                cleaned_content = clean_html_from_text(original_content)

                # Optimisation: Only write if changes were actually made
                if original_content != cleaned_content:
                    with md_file.open('w', encoding='utf-8') as f:
                        num_char_written = f.write(cleaned_content)
                        print(f"cleaned: {md_file.name} written {num_char_written} characters.")
                        files_processed += 1
                else:
                    pass

        except Exception as e:
            print(f"ERROR in {md_file.name}: {e}")
            errors += 1

    print(f"Total files updated: {files_processed}")
    print(f"Total errors: {errors}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_folder = Path(__file__).parent / "../../out/" 
    
    # Read all markdown files inside the out folder
    md_docs = [doc_path for doc_path in data_folder.glob('**/*.md')]
    
    # md_docs = [data_folder / "test_doc.md"]
    
    process_md_files(md_docs)
