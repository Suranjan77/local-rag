from pathlib import Path
from llama_cpp import Llama
import numpy as np
import logging
import json

_log = logging.getLogger(__name__)

MODEL_PATH = "/home/sur/quant_models/Qwen3-Embedding-8B-Q8_0.gguf"

def embedd_chunks(chunk_files: list[Path]):
    embedding_model = Llama(
            model_path=MODEL_PATH,
            embedding=True,
            n_ctx=4096,
            verbose=False
        )

    output_path = Path(__file__).parent / "../../out/embeddings"
    output_path.mkdir(parents=True, exist_ok=True)

    for chunk_file in chunk_files:
        with chunk_file.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
            chunk_contents = [c["content"]for c in chunks]
            embeddings = []
            for chunk in chunk_contents:
                res = embedding_model.create_embedding(chunk)
                embedding_vec = res['data'][0]['embedding']

                norm = np.linalg.norm(embedding_vec)
                embeddings.append((embedding_vec / norm).tolist() if norm > 0 else embedding_vec)

        # Saving raw vector/float-array in binary form
        out_file = chunk_file.stem + "_embeddings.bin"
        with (output_path / out_file).open("wb") as f:
            embeddings_np = np.array(embeddings, dtype=np.float32)
            header = np.array([embeddings_np.shape[0], embeddings_np.shape[1]], dtype=np.int32)
            header.tofile(f)
            embeddings_np.tofile(f)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data_folder = Path(__file__).parent / "../../out/chunks" 
    
    # Read all chunk files inside the out folder
    chunk_files = [doc_path for doc_path in data_folder.glob('**/*.json')]
    
    # chunk_files = [data_folder / "test_doc_chunks.json"]
    
    embedd_chunks(chunk_files)
