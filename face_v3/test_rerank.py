"""
Python wrapper for C++ rerank computation implementation.
Provides high-performance rerank score calculation using C++ backend.
"""

import ctypes
import numpy as np
import os
from typing import List, Tuple
from loguru import logger
import time

class RerankComputeCpp:
    """Python wrapper for C++ rerank computation."""
    
    def __init__(self):
        """Initialize the C++ library."""
        self._lib = None
        self._load_library()
    
    def _load_library(self):
        """Load the C++ shared library."""
        # Try to find the library in different locations
        possible_paths = [
            # Same directory as this Python file
            os.path.join(os.path.dirname(__file__), 'libs', 'librerank_compute.so'),
            # Absolute path from project root
            '/path/to/librerank_compute.so',
            # System library path
            'librerank_compute.so'
        ]
        
        for path in possible_paths:
            try:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    self._lib = ctypes.CDLL(abs_path)
                    logger.info(f"Loaded C++ rerank compute library from: {abs_path}")
                    break
            except Exception as e:
                logger.debug(f"Failed to load library from {path}: {e}")
                continue
        
        if self._lib is None:
            raise RuntimeError(
                "Could not load C++ rerank compute library. "
                "Please ensure librerank_compute.so is built and accessible."
            )
        
        # Define function signature
        self._lib.compute_rerank_scores_from_embeddings.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # query_embedding
            ctypes.POINTER(ctypes.c_double),  # candidate_embeddings
            ctypes.c_int,                     # num_candidates
            ctypes.c_double,                  # threshold
            ctypes.POINTER(ctypes.c_double)   # rerank_scores (output)
        ]
        self._lib.compute_rerank_scores_from_embeddings.restype = ctypes.c_int
    
    def compute_rerank_scores(self, 
                            query_embedding: np.ndarray, 
                            candidate_embeddings: List[np.ndarray],
                            threshold: float = 0.7) -> np.ndarray:
        """
        Compute rerank scores using C++ implementation.
        
        Args:
            query_embedding: Query embedding (1024-dimensional)
            candidate_embeddings: List of candidate embeddings (1024-dimensional each)
            threshold: Threshold for voting (default: 0.7)
            
        Returns:
            Rerank scores array
        """
        if self._lib is None:
            raise RuntimeError("C++ library not loaded")
        
        # Validate inputs
        if query_embedding.shape[-1] != 1024:
            raise ValueError(f"Expected 1024-dimensional query embedding, got {query_embedding.shape[-1]}")
        
        num_candidates = len(candidate_embeddings)
        if num_candidates == 0:
            return np.array([]), 0, 0
        
        # Validate candidate embeddings
        for i, emb in enumerate(candidate_embeddings):
            if emb.shape[-1] != 1024:
                raise ValueError(f"Expected 1024-dimensional candidate embedding {i}, got {emb.shape[-1]}")
        
        # Prepare input arrays
        query_emb_flat = query_embedding.flatten().astype(np.float64)
        
        # Flatten all candidate embeddings into a single array
        candidate_embs_flat = np.vstack(candidate_embeddings).flatten().astype(np.float64)
        
        # Prepare output arrays
        rerank_scores = np.zeros(num_candidates, dtype=np.float64)
        
        # Convert to ctypes pointers
        query_ptr = query_emb_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        candidates_ptr = candidate_embs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        scores_ptr = rerank_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        
        # Call C++ function
        result = self._lib.compute_rerank_scores_from_embeddings(
            query_ptr,
            candidates_ptr,
            num_candidates,
            ctypes.c_double(threshold),
            scores_ptr
        )
        
        if result != 0:
            raise RuntimeError("C++ rerank computation failed")
        
        return rerank_scores
    
    @property
    def is_available(self) -> bool:
        """Check if C++ library is available."""
        return self._lib is not None


# Global instance
_cpp_rerank_compute = None

def get_cpp_rerank_compute() -> RerankComputeCpp:
    """Get global instance of C++ rerank compute."""
    global _cpp_rerank_compute
    if _cpp_rerank_compute is None:
        try:
            _cpp_rerank_compute = RerankComputeCpp()
        except Exception as e:
            logger.warning(f"Failed to initialize C++ rerank compute: {e}")
            _cpp_rerank_compute = None
    return _cpp_rerank_compute

def is_cpp_available() -> bool:
    """Check if C++ implementation is available."""
    cpp_compute = get_cpp_rerank_compute()
    return cpp_compute is not None and cpp_compute.is_available


if __name__ == "__main__":
    
    if not is_cpp_available():
        print("❌ C++ implementation not available")
        raise
    
    print("✅ C++ implementation is available")
    
    cpp_compute = get_cpp_rerank_compute()

    # Test parameters
    EMBEDDING_DIM = 1024
    NUM_CANDIDATES = 100
    THRESHOLD = 0.298
    
    # Generate test embeddings
    np.random.seed(42)
    query_embedding = np.random.normal(0, 1, EMBEDDING_DIM).astype(np.float64)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    
    candidate_embeddings = []
    for i in range(NUM_CANDIDATES):
        if i < 3:  # Similar to query
            noise = np.random.normal(0, 0.1, EMBEDDING_DIM)
            candidate = query_embedding + noise
        else:
            candidate = np.random.normal(0, 1, EMBEDDING_DIM)
        candidate = candidate / np.linalg.norm(candidate)
        candidate_embeddings.append(candidate.astype(np.float64))
    
    # Test C++ computation
    start_time = time.time()
    scores = cpp_compute.compute_rerank_scores(query_embedding, candidate_embeddings, THRESHOLD)
    cpp_time = (time.time() - start_time) * 1000
    
    print(f"✅ C++ computation successful!")
    print(f"  Execution time: {cpp_time:.3f} ms")
    print(f"  Scores shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    print(f"  Non-zero scores: {np.count_nonzero(scores)}")
    
    # Validate results
    assert len(scores) == NUM_CANDIDATES
    assert all(0.0 <= score <= 1.0 for score in scores)
    print("✅ All validations passed!")