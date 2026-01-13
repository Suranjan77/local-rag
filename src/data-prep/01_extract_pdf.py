import logging
import time
from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

_log = logging.getLogger(__name__)

def batch_parse_to_md(input_documents: list[Path]):
    logging.basicConfig(level=logging.DEBUG)

    for input_doc_path in input_documents:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.do_formula_enrichment = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )

        doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        start_time = time.time()
        conv_result = doc_converter.convert(input_doc_path)
        end_time = time.time() - start_time


        doc_filename = conv_result.input.file.stem
        _log.info(f"Document {doc_filename} converted in {end_time:.2f} seconds.")

        output_dir = Path(__file__).parent / "../../out"
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / f"{doc_filename}.md").open("w", encoding="utf-8") as fp:
            num_of_char_written = fp.write(conv_result.document.export_to_markdown())
            print(f"Written {num_of_char_written} characters.")

if __name__ == "__main__":
    data_folder = Path(__file__).parent / "../../data"
    # Read all pdfs inside the data folder
    # pdf_docs = [doc_path for doc_path in (data_folder / "pdfs").glob('**/*.pdf')]
    pdf_docs = [data_folder / "test_doc.pdf"]
    batch_parse_to_md(pdf_docs)
