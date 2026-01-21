from langchain_text_splitters import MarkdownHeaderTextSplitter
from pathlib import Path
import logging
import json

_log = logging.getLogger(__name__)

def batch_chunk_md(md_files: list[Path]):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    split_out_dir = Path(__file__).parent / "../../out/chunks"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    for md_file in md_files:
        with md_file.open("r", encoding="utf-8") as fp:
            md_data = fp.read()
            md_splits = md_splitter.split_text(md_data)
            
            chunk_data = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } 
                for doc in md_splits
            ]

            out_file = md_file.stem + "_chunks.json"

            with (split_out_dir / out_file).open('w', encoding='utf-8') as f:
                json.dump(chunk_data, f, indent=4, ensure_ascii=False)
                print(f"chunked: {md_file.name}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    data_folder = Path(__file__).parent / "../../out/" 
    
    # Read all markdown files inside the out folder
    # md_docs = [doc_path for doc_path in data_folder.glob('**/*.md')]
    
    md_docs = [data_folder / "test_doc.md"]
    
    batch_chunk_md(md_docs)
