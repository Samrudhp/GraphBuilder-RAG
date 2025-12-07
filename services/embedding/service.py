"""
Embedding + FAISS Index Service

Handles:
- Text embeddings generation (sentence-transformers)
- FAISS index creation and management
- Batch embedding pipeline
- Chunk-level embedding storage
- Similarity search
"""
import logging
import pickle
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

# Force CPU-only BEFORE importing PyTorch/transformers
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'  # Critical for SentenceTransformer

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

# Set default device to CPU immediately
torch.set_default_device('cpu')

from shared.config.settings import get_settings
from shared.database.mongodb import get_mongodb

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using sentence-transformers."""
    
    def __init__(self):
        self.settings = get_settings().embedding
        self.model = None
        
    def _load_model(self):
        """Synchronous lazy load using SentenceTransformer."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.settings.model}")
            
            # Check if model is cached
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers" / f"sentence-transformers_{self.settings.model.replace('/', '-')}"
            is_cached = cache_dir.exists()
            
            if is_cached:
                logger.info(f"✅ Found cached model at {cache_dir}")
                logger.info("Loading from cache...")
            else:
                logger.info(f"⏳ Downloading {self.settings.model} (first time, ~90MB)...")
            
            try:
                # Use SentenceTransformer - handles tokenization + model internally
                self.model = SentenceTransformer(
                    self.settings.model,
                    device='cpu',  # Force CPU to avoid MPS issues
                    cache_folder=None  # Use default cache
                )
                logger.info("✅ Model loaded successfully")
                
            except Exception as e:
                if is_cached:
                    logger.warning(f"Cache error: {e}, re-downloading...")
                    # Clear cache and retry
                    import shutil
                    if cache_dir.exists():
                        shutil.rmtree(cache_dir)
                    self.model = SentenceTransformer(
                        self.settings.model,
                        device='cpu'
                    )
                    logger.info("✅ Model re-downloaded")
                else:
                    raise
            
            logger.info(f"✅ Embedding model ready on {self.model.device}")
            logger.info(f"Max sequence length: {self.model.max_seq_length}")
            logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (normalized)
        """
        self._load_model()
        
        # Use SentenceTransformer.encode() - handles tokenization, pooling, and normalization
        embedding = self.model.encode(
            text,
            convert_to_tensor=False,  # Return numpy array
            normalize_embeddings=True,  # Already normalized
            show_progress_bar=False
        )
        
        return embedding
    
    def embed_batch(self, texts: list[str], batch_size: Optional[int] = None) -> np.ndarray:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing (SentenceTransformer handles this internally)
            
        Returns:
            Array of embedding vectors (normalized)
        """
        self._load_model()
        
        logger.debug(f"Embedding batch of {len(texts)} texts")
        
        # Use SentenceTransformer.encode() - handles batching, tokenization, pooling, and normalization
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size or self.settings.batch_size,
            convert_to_tensor=False,  # Return numpy array
            normalize_embeddings=True,  # Already normalized
            show_progress_bar=False
        )
        
        return embeddings


class FAISSIndexService:
    """Service for managing FAISS vector index."""
    
    def __init__(self):
        self.settings = get_settings().faiss
        self.embedding_settings = get_settings().embedding
        self.mongodb = get_mongodb()
        self.embeddings_meta = self.mongodb.get_async_collection("embeddings_meta")
        
        self.index: Optional[faiss.Index] = None
        self.chunk_id_map: dict[int, str] = {}  # faiss_id -> chunk_id
        
        # Ensure index directory exists
        self.settings.index_path.mkdir(parents=True, exist_ok=True)
        
    def create_index(self, index_type: str = "IndexFlatIP"):
        """
        Create a new FAISS index.
        
        Args:
            index_type: Type of FAISS index (IndexFlatIP, IndexIVFFlat, IndexHNSWFlat)
        """
        dim = self.embedding_settings.dimension
        
        if index_type == "IndexFlatIP":
            # Flat index with inner product (cosine similarity)
            self.index = faiss.IndexFlatIP(dim)
            logger.info(f"Created IndexFlatIP with dimension {dim}")
            
        elif index_type == "IndexIVFFlat":
            # IVF index for faster approximate search
            nlist = self.settings.nlist
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Created IndexIVFFlat with {nlist} clusters")
            
        elif index_type == "IndexHNSWFlat":
            # HNSW index for very fast approximate search
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
            logger.info(f"Created IndexHNSWFlat with dimension {dim}")
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
    ):
        """
        Add embeddings to FAISS index.
        
        Args:
            embeddings: Array of embedding vectors (N x dim)
            chunk_ids: List of chunk IDs corresponding to embeddings
        """
        if self.index is None:
            self.create_index(self.settings.index_type)
        
        # Train index if needed (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("Index trained")
        
        # Get starting faiss ID
        start_id = self.index.ntotal
        
        # Add to index
        self.index.add(embeddings)
        
        # Update chunk ID map
        for i, chunk_id in enumerate(chunk_ids):
            faiss_id = start_id + i
            self.chunk_id_map[faiss_id] = chunk_id
        
        logger.info(f"Added {len(chunk_ids)} embeddings to index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> tuple[list[str], list[float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Tuple of (chunk_ids, scores)
        """
        if self.index is None:
            logger.warning("Index not loaded, attempting to load from disk")
            self.load_index()
            if self.index is None:
                logger.error("Failed to load index, returning empty results")
                return [], []
        
        # Set nprobe for IVF index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.settings.nprobe
        
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Map FAISS IDs to chunk IDs
        chunk_ids = []
        result_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:  # No result
                continue
            chunk_id = self.chunk_id_map.get(int(idx))
            if chunk_id:
                chunk_ids.append(chunk_id)
                result_scores.append(float(score))
        
        logger.debug(f"Found {len(chunk_ids)} results from {len(self.chunk_id_map)} mapped chunks")
        return chunk_ids, result_scores
    
    def save_index(self, path: Optional[Path] = None):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        save_path = path or self.settings.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save chunk ID map
        map_file = save_path / "chunk_map.pkl"
        with open(map_file, "wb") as f:
            pickle.dump(self.chunk_id_map, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[Path] = None):
        """Load FAISS index and metadata from disk."""
        load_path = path or self.settings.index_path
        
        index_file = load_path / "index.faiss"
        map_file = load_path / "chunk_map.pkl"
        
        if not index_file.exists():
            logger.warning(f"Index file not found: {index_file} - creating new index")
            self.create_index(self.settings.index_type)
            return
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Load chunk ID map
        if map_file.exists():
            with open(map_file, "rb") as f:
                self.chunk_id_map = pickle.load(f)
            logger.info(f"Index loaded from {load_path} (vectors: {self.index.ntotal}, chunks: {len(self.chunk_id_map)})")
        else:
            logger.warning(f"Chunk map not found: {map_file}")
            self.chunk_id_map = {}
            logger.info(f"Index loaded from {load_path} (vectors: {self.index.ntotal}, NO chunk map)")
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "dimension": self.embedding_settings.dimension,
            "index_type": type(self.index).__name__,
            "is_trained": getattr(self.index, "is_trained", True),
        }


class EmbeddingPipelineService:
    """Pipeline for embedding documents and building index."""
    
    def __init__(self):
        self.mongodb = get_mongodb()
        self.normalized_docs = self.mongodb.get_async_collection("normalized_docs")
        self.embeddings_meta = self.mongodb.get_async_collection("embeddings_meta")
        
        self.embedding_service = EmbeddingService()
        self.faiss_service = FAISSIndexService()
        
        # Load existing FAISS index on initialization
        self.faiss_service.load_index()
        
    async def embed_document(
        self,
        document_id: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> int:
        """
        Embed a document by chunking and storing in FAISS.
        
        Args:
            document_id: Normalized document ID
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            Number of chunks embedded
        """
        logger.info(f"Embedding document: {document_id}")
        
        # Get document
        doc = await self.normalized_docs.find_one({"document_id": document_id})
        if not doc:
            raise ValueError(f"Document not found: {document_id}")
        
        text = doc.get("text", "")
        if not text:
            logger.warning(f"Document has no text: {document_id}")
            return 0
        
        # Chunk text
        chunks = self._chunk_text(text, chunk_size, overlap)
        
        # Generate chunk IDs
        chunk_ids = [f"chunk_{uuid4().hex[:12]}" for _ in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_batch(chunks)
        
        # Add to FAISS
        self.faiss_service.add_embeddings(embeddings, chunk_ids)
        
        # Store metadata in MongoDB
        metadata_docs = []
        for i, (chunk_id, chunk_text) in enumerate(zip(chunk_ids, chunks)):
            metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "faiss_id": self.faiss_service.index.ntotal - len(chunks) + i,
                "text": chunk_text,
                "chunk_index": i,
                "metadata": {
                    "char_start": i * (chunk_size - overlap),
                    "char_end": i * (chunk_size - overlap) + len(chunk_text),
                },
            }
            metadata_docs.append(metadata)
        
        await self.embeddings_meta.insert_many(metadata_docs)
        
        logger.info(f"Embedded {len(chunks)} chunks for document {document_id}")
        return len(chunks)
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """
        Chunk text with overlap.
        
        Args:
            text: Input text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            start += chunk_size - overlap
            
            # Avoid tiny last chunk
            if start < len(text) and len(text) - start < overlap:
                break
        
        return chunks
    
    async def search_similar_chunks(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.5,
    ) -> list[dict]:
        """
        Search for similar text chunks.
        
        Args:
            query: Query text
            top_k: Number of results
            min_score: Minimum similarity score
            
        Returns:
            List of chunk results with metadata
        """
        # Ensure model is loaded (lazy loading)
        self.embedding_service._load_model()
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search FAISS
        chunk_ids, scores = self.faiss_service.search(query_embedding, top_k * 2)
        
        # Get chunk metadata from MongoDB
        results = []
        for chunk_id, score in zip(chunk_ids, scores):
            if score < min_score:
                continue
            
            metadata = await self.embeddings_meta.find_one({"chunk_id": chunk_id})
            if metadata:
                results.append({
                    "chunk_id": chunk_id,
                    "document_id": metadata["document_id"],
                    "text": metadata["text"],
                    "score": score,
                    "metadata": metadata.get("metadata", {}),
                })
            
            if len(results) >= top_k:
                break
        
        logger.debug(f"Retrieved {len(results)} chunks for query")
        return results
    
    async def rebuild_index(self):
        """Rebuild FAISS index from all embeddings in MongoDB."""
        logger.info("Rebuilding FAISS index from MongoDB...")
        
        # Clear current index
        self.faiss_service.create_index(self.faiss_service.settings.index_type)
        
        # Fetch all embeddings
        batch_size = 1000
        cursor = self.embeddings_meta.find({}).batch_size(batch_size)
        
        batch_embeddings = []
        batch_chunk_ids = []
        
        async for doc in cursor:
            chunk_id = doc["chunk_id"]
            text = doc["text"]
            
            # Re-generate embedding
            embedding = self.embedding_service.embed_text(text)
            
            batch_embeddings.append(embedding)
            batch_chunk_ids.append(chunk_id)
            
            if len(batch_embeddings) >= batch_size:
                embeddings_array = np.array(batch_embeddings)
                self.faiss_service.add_embeddings(embeddings_array, batch_chunk_ids)
                batch_embeddings = []
                batch_chunk_ids = []
        
        # Add remaining
        if batch_embeddings:
            embeddings_array = np.array(batch_embeddings)
            self.faiss_service.add_embeddings(embeddings_array, batch_chunk_ids)
        
        # Save index
        self.faiss_service.save_index()
        
        logger.info(f"Index rebuilt with {self.faiss_service.index.ntotal} vectors")
