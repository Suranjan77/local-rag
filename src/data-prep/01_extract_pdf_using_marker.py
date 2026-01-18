import logging
import time
import json
from pathlib import Path

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
_log = logging.getLogger(__name__)

def get_converter_config(output_dir: Path):
    return {
        "output_format": "chunks",
        "use_llm": False,
        "disable_image_extraction": True,
        "llm_service": "marker.services.ollama.OllamaService",
        "ollama_base_url": "http://localhost:11434",
        "ollama_model": "llama-vision:latest",
        "output_dir": str(output_dir),
    }

def batch_process_marker(input_paths: list[Path], output_dir: Path):
    config_dict = get_converter_config(output_dir)
    config_parser = ConfigParser(config_dict)

    _log.info("Loading Marker models and artifacts... (this may take a moment)")
    
    converter = PdfConverter(
        config=config_parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service()
    )
    
    _log.info("Models loaded. Starting batch processing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for doc_path in input_paths:
        try:
            start_time = time.time()
            doc_filename = doc_path.stem
            
            rendered = converter(str(doc_path))
            text, _, images = text_from_rendered(rendered)
            
            out_file = output_dir / f"{doc_filename}.json"
            
            with out_file.open("w", encoding="utf-8") as f:
                if isinstance(text, str):
                    f.write(text)
                else:
                    json.dump(text, f, indent=4)
            
            duration = time.time() - start_time
            _log.info(f"Converted {doc_filename} in {duration:.2f}s")
            
        except Exception as e:
            _log.error(f"Failed to convert {doc_path.name}: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent / "../../"
    data_folder = base_dir / "data/pdfs"
    output_folder = base_dir / "out"

    # pdf_files = list(data_folder.rglob("*.pdf"))
    pdf_files = [data_folder / 'test_doc.pdf']

    if not pdf_files:
        _log.warning(f"No PDFs found in {data_folder}")
    else:
        _log.info(f"Found {len(pdf_files)} PDFs to process.")
        batch_process_marker(pdf_files, output_folder)
