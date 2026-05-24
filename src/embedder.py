from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = 16):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings
