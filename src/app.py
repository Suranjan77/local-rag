import logging
import json
import numpy as np
import faiss
from pathlib import Path
from llama_cpp import Llama

ROOT_DIR = Path(__file__).parent / "../"
EMBEDDING_MODEL_PATH = ROOT_DIR / "models/Qwen3-Embedding-8B-Q8_0.gguf"
CHAT_MODEL_PATH = ROOT_DIR / "models/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf"
CHUNKS_DIR = ROOT_DIR / "out/chunks"
EMBEDS_DIR = ROOT_DIR / "out/embeddings"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class LocalRAG:
    def __init__(self, embed_path: str, chat_path: str, chunks_dir: Path, embeds_dir: Path):
        self.chunks_dir = chunks_dir
        self.embeds_dir = embeds_dir
        self.text_registry = []
        self.index = None
        
        logger.info("Initializing Embedding Model...")
        self.embed_model = Llama(
            model_path=embed_path,
            embedding=True,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )

        logger.info("Initialising Chat Model ...")
        self.chat_model = Llama(
            model_path=chat_path,
            n_ctx=5000,
            n_gpu_layers=-1,    # Offload all layers to GPU
            n_batch=2048,
            n_threads=12,
            verbose=False
        )

        self._build_index()

    def _build_index(self):
        logger.info("Loading data and building FAISS index...")
        
        chunk_files = sorted(list(self.chunks_dir.glob('**/*.json')))
        if not chunk_files:
            raise FileNotFoundError(f"No .json chunk files found in {self.chunks_dir}")

        all_vectors = []
        total_docs = 0

        for json_path in chunk_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Create a flat list of content strings
                    current_texts = [item["content"] for item in data]
                    self.text_registry.extend(current_texts)
            except Exception as e:
                logger.error(f"Failed to load JSON {json_path}: {e}")
                continue

            bin_filename = json_path.stem + "_embeddings.bin"
            bin_path = self.embeds_dir / bin_filename
            
            if not bin_path.exists():
                logger.warning(f"Missing embeddings for {json_path.name}")
                self.text_registry = self.text_registry[:-len(current_texts)]
                continue

            with open(bin_path, 'rb') as f:
                # Read Header: [rows, cols] (int32)
                header = np.fromfile(f, dtype=np.int32, count=2)
                rows, cols = header[0], header[1]
                
                # Read Data: float32
                vecs = np.fromfile(f, dtype=np.float32).reshape(rows, cols)
                all_vectors.append(vecs)
                total_docs += rows

        if not all_vectors:
            raise RuntimeError("No vectors loaded. Check your data paths.")

        # Consolidate into one big matrix
        embedding_matrix = np.vstack(all_vectors)

        # Sanity Check
        if len(self.text_registry) != embedding_matrix.shape[0]:
            raise ValueError(f"CRITICAL: Data mismatch. Text chunks: {len(self.text_registry)}, Vectors: {embedding_matrix.shape[0]}")

        d = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embedding_matrix)
        
        logger.info(f"Index built. Total Vectors: {self.index.ntotal} | Dimension: {d}")
        logger.info(f"FAISS is using instructions: {faiss.get_compile_options()}")

    def search(self, query: str, top_k: int = 3):
        res = self.embed_model.create_embedding(input=query)
        q_vec = np.array(res['data'][0]['embedding'], dtype=np.float32)

        # Normalize (CRITICAL for Cosine Similarity via Inner Product)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec = q_vec / norm
        
        q_vec = q_vec.reshape(1, -1)

        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "score": float(distances[0][i]),
                    "content": self.text_registry[idx]
                })
        return results

    def chat(self, user_query: str):
        retrieved = self.search(user_query, top_k=5)
        
        context_str = ""
        for i, doc in enumerate(retrieved):
            context_str += f"\n[Document {i+1}]:\n{doc['content']}\n"

        system_msg = (
            "You are a precise technical assistant. "
            "Answer the user's question using ONLY the context provided below. "
            "If the answer is not in the context, state that you do not know."
        )

        user_msg = (
            f"Context Information:\n{context_str}\n\n"
            f"Question: {user_query}"
        )

        stream = self.chat_model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=0.8,
            top_k=20,
            min_p=0,
            stream=True
        )

        print("\nAssistant: ", end="", flush=True)
        full_response = ""
        for chunk in stream:
            if "content" in chunk["choices"][0]["delta"]:
                token = chunk["choices"][0]["delta"]["content"]
                print(token, end="", flush=True)
                full_response += token
        print("\n")
        
        return full_response

if __name__ == "__main__":
    try:
        app = LocalRAG(
            embed_path=str(EMBEDDING_MODEL_PATH),
            chat_path=str(CHAT_MODEL_PATH),
            chunks_dir=CHUNKS_DIR,
            embeds_dir=EMBEDS_DIR
        )

        print("\n" + "="*50)
        print(f" RAG System Online")
        print("="*50 + "\n")

        while True:
            q = input("Query (or 'q' to quit): ")
            if q.lower() in ['q', 'exit', 'quit']:
                break
            
            app.chat(q)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Fatal Error: {e}")
