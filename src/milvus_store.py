from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


class MilvusStore:
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "industrial_maintenance_chunks",
        dim: int = 1024,
    ):
        self.collection_name = collection_name
        self.dim = dim

        connections.connect(
            alias="default",
            host=host,
            port=port,
        )

    def recreate_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_start", dtype=DataType.INT64),
            FieldSchema(name="page_end", dtype=DataType.INT64),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="equipment", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="fault_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Industrial maintenance RAG chunks",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            },
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        return collection

    def insert_chunks(self, chunks, embeddings):
        collection = Collection(self.collection_name)

        data = [
            [c["chunk_id"] for c in chunks],
            [c["doc_id"] for c in chunks],
            [c["source_file"] for c in chunks],
            [int(c["page_start"]) for c in chunks],
            [int(c["page_end"]) for c in chunks],
            [c.get("section", "")[:256] for c in chunks],
            [c["text"][:8192] for c in chunks],
            [c.get("language", "unknown")[:16] for c in chunks],
            [c.get("metadata", {}).get("equipment", "bearing")[:64] for c in chunks],
            [c.get("metadata", {}).get("fault_type", "")[:64] for c in chunks],
            embeddings.tolist(),
        ]

        collection.insert(data)
        collection.flush()
        collection.load()

        return collection.num_entities
