#!/usr/bin/env python3
"""
Standalone model test file for face detection and extraction.
Processes images from a directory using only Triton inference server.
No database or MinIO interactions - just model inference.
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
from loguru import logger
import tritonclient.grpc as grpcclient
import skimage.transform as trans
from sklearn import preprocessing
from PIL import Image
import io


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FaceDetection:
    """Face detection result."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    face_size: float
    face_aligned: np.ndarray  # 112x112x3 aligned face
    landmarks: np.ndarray     # 5x2 landmarks


@dataclass
class ModelConfig:
    """Model configuration."""
    # Detection
    detection_model_name: str = "detection"
    detection_input_names: List[str] = None
    detection_input_types: List[str] = None
    detection_output_names: List[str] = None
    
    # Extraction
    extraction_model_name: str = "extraction"
    extraction_input_names: List[str] = None
    extraction_input_types: List[str] = None
    extraction_output_names: List[str] = None
    
    def __post_init__(self):
        # Detection defaults
        if self.detection_input_names is None:
            self.detection_input_names = ["input_0", "input_1", "input_2"]
        if self.detection_input_types is None:
            self.detection_input_types = ["UINT8", "FP32", "FP32"]
        if self.detection_output_names is None:
            self.detection_output_names = ["output_0", "output_1", "output_2"]
            
        # Extraction defaults
        if self.extraction_input_names is None:
            self.extraction_input_names = ["input_0"]
        if self.extraction_input_types is None:
            self.extraction_input_types = ["UINT8"]
        if self.extraction_output_names is None:
            self.extraction_output_names = ["output_0"]

# ============================================================================
# IMAGE UTILITIES
# ============================================================================

def square_crop(img: np.ndarray, input_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """
    Crop image to square maintaining aspect ratio.
    
    Args:
        img: Input image
        input_size: Target size (width, height)
        
    Returns:
        (cropped_image, scale_factor)
    """
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
        
    det_scale = float(new_height) / img.shape[0]
    resized_img = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img
    
    return det_img, det_scale


def align_face(img: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face using landmarks to 112x112.
    
    Args:
        img: Input image
        landmarks: 5x2 landmarks array
        
    Returns:
        Aligned face image (112x112x3)
    """
    # Standard face landmarks for 112x112 output
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    transform = trans.SimilarityTransform()
    transform.estimate(landmarks, dst)
    M = transform.params[0:2, :]

    face = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)
    return face


def normalize_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Normalize bounding box coordinates.
    
    Args:
        box: [x1, y1, x2, y2, confidence]
        width: Image width
        height: Image height
        
    Returns:
        Normalized box coordinates
    """
    xmin = max(0, box[0])
    ymin = max(0, box[1])
    xmax = min(width, box[2])
    ymax = min(height, box[3])
    return np.array([xmin, ymin, xmax, ymax, box[4]])


def bytes_to_image(data: bytes) -> np.ndarray:
    """Convert bytes to numpy image."""
    img = Image.open(io.BytesIO(data))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ============================================================================
# TRITON CLIENT
# ============================================================================

class StandaloneTritonClient:
    """Standalone Triton client for face detection and extraction."""
    
    def __init__(self, triton_url: str, model_config: ModelConfig):
        """
        Initialize Triton client.
        
        Args:
            triton_url: Triton server URL (e.g., "localhost:8001")
            model_config: Model configuration
        """
        self.triton_url = triton_url
        self.model_config = model_config
        
        # Initialize gRPC client
        self.client = grpcclient.InferenceServerClient(
            url=triton_url,
            verbose=False
        )
        
        # Check server health
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server at {triton_url} is not live")
        
        logger.info(f"Connected to Triton server at {triton_url}")
        
    def infer(self, model_name: str, inputs_data: List[np.ndarray], 
             inputs_name: List[str], inputs_type: List[str], 
             outputs_name: List[str]) -> List[np.ndarray]:
        """
        Run inference on Triton server.
        
        Args:
            model_name: Name of the model
            inputs_data: List of input numpy arrays
            inputs_name: List of input names
            inputs_type: List of input types
            outputs_name: List of output names
            
        Returns:
            List of output numpy arrays
        """
        # Prepare inputs
        inputs = []
        for i, input_data in enumerate(inputs_data):
            inputs.append(grpcclient.InferInput(inputs_name[i], input_data.shape, inputs_type[i]))
            inputs[i].set_data_from_numpy(input_data)

        # Prepare outputs
        outputs = []
        for out_name in outputs_name:
            outputs.append(grpcclient.InferRequestedOutput(out_name))

        # Run inference
        results = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
        )
        
        # Extract output arrays
        outputs_data = [results.as_numpy(outputs_name[i]) for i in range(len(outputs_name))]
        return outputs_data
    
    def batch_detect_faces(self, imgs: List[np.ndarray], 
                          input_size: Tuple[int, int] = (640, 640)) -> List[List[FaceDetection]]:
        """
        Detect faces in batch of images.
        
        Args:
            imgs: List of input images
            input_size: Detection input size
            
        Returns:
            List of face detections for each image
        """
        # Preprocess images
        det_imgs = []
        det_scales = []
        img_centers = []
        
        for img in imgs:
            img_center = img.shape[0] // 2, img.shape[1] // 2
            det_img, det_scale = square_crop(img, input_size)
            det_imgs.append(det_img)
            det_scales.append((det_scale, det_scale))
            img_centers.append(img_center)
            
        det_imgs = np.stack(det_imgs)
        det_scales = np.stack(det_scales).astype(np.float32)
        img_centers = np.stack(img_centers).astype(np.float32)
        
        # Run inference
        inputs_data = [det_imgs, det_scales, img_centers]
        outputs_data = self.infer(
            model_name=self.model_config.detection_model_name,
            inputs_data=inputs_data,
            inputs_name=self.model_config.detection_input_names,
            inputs_type=self.model_config.detection_input_types,
            outputs_name=self.model_config.detection_output_names
        )
        
        bboxes, lmks, num_boxes = outputs_data[0], outputs_data[1], outputs_data[2]
        
        # Parse results
        all_detections = self._parse_detection_results(imgs, bboxes, lmks, num_boxes)
        return all_detections
    
    def _parse_detection_results(self, images: List[np.ndarray], 
                               boxes_batch: np.ndarray, lmks_batch: np.ndarray, 
                               num_boxes: np.ndarray) -> List[List[FaceDetection]]:
        """Parse detection results into structured format."""
        batch_result = []
        start = 0
        
        for i in range(num_boxes.shape[0]):
            num_faces = num_boxes[i][0]
            if num_faces == 0:
                batch_result.append([])
                continue

            image = images[i]
            height, width = image.shape[:2]
            end = start + num_faces
            boxes = boxes_batch[start:end]
            lmks = lmks_batch[start:end]
            faces = []
            
            for j, lmk in enumerate(lmks):
                box = boxes[j]
                box = normalize_box(box, width, height)
                
                # Align face
                face_aligned = align_face(image, lmk)

                face = FaceDetection(
                    bbox=box[:4].tolist(),
                    confidence=box[4],
                    face_size=(box[2] - box[0]) * (box[3] - box[1]),
                    face_aligned=face_aligned,
                    landmarks=lmk
                )
                faces.append(face)
                
            batch_result.append(faces)
            start = end

        return batch_result
    
    def batch_extract_embeddings(self, faces: List[np.ndarray], normalize=True):
        """
        Extract embeddings using DALI-based model.
        Returns 1024-dim embeddings.
        
        Args:
            faces: List of face images as numpy arrays (BGR format)
            normalize: Whether to normalize embeddings
            
        Returns:
            embeddings_1024 - embeddings are 1024-dim
        """        
        inputs_data = [np.stack(faces)]
        
        outputs_data = self.infer(
            model_name=self.model_config.extraction_model_name,
            inputs_data=inputs_data,
            inputs_name=self.model_config.extraction_input_names,
            inputs_type=self.model_config.extraction_input_types,
            outputs_name=self.model_config.extraction_output_names
        )

        embeddings = outputs_data[0]
        
        if normalize:
            # Normalize the 1024-dim embeddings
            batch_embeddings = preprocessing.normalize(embeddings, axis=1)  # (B, 1024)
        else:
            batch_embeddings = embeddings

        return batch_embeddings


# ============================================================================
# DIRECTORY PROCESSOR
# ============================================================================

class DirectoryProcessor:
    """Process images from directory using Triton models."""
    
    def __init__(self, triton_url: str, model_config: ModelConfig):
        """
        Initialize directory processor.
        
        Args:
            triton_url: Triton server URL
            model_config: Model configuration
        """
        self.client = StandaloneTritonClient(triton_url, model_config)
        
    def process_directory(self, directory_path: str, 
                         batch_size: int = 16,
                         detection_batch_size: int = 16,
                         extraction_batch_size: int = 32,
                         recursive: bool = True,
                         save_results: bool = True,
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all images in directory.
        
        Args:
            directory_path: Path to directory containing images
            batch_size: Batch size for image processing (deprecated, use detection_batch_size)
            detection_batch_size: Batch size for detection model
            extraction_batch_size: Maximum batch size for extraction model
            recursive: Whether to search subdirectories
            save_results: Whether to save results to files
            output_dir: Output directory for results
            
        Returns:
            Processing statistics
        """
        path = Path(directory_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find image files
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        if recursive:
            image_files = [f for ext in extensions 
                          for f in path.rglob(f"*{ext}")]
            image_files.extend([f for ext in extensions 
                               for f in path.rglob(f"*{ext.upper()}")])
        else:
            image_files = [f for ext in extensions 
                          for f in path.glob(f"*{ext}")]
            image_files.extend([f for ext in extensions 
                               for f in path.glob(f"*{ext.upper()}")])
        
        logger.info(f"Found {len(image_files)} images to process")
        
        # Setup output directory
        if save_results and output_dir is None:
            output_dir = f"model_test_results_{int(time.time())}"
        
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving results to: {output_dir}")
        
        # Processing statistics
        stats = {
            "total_images": len(image_files),
            "processed": 0,
            "failed": 0,
            "total_faces_detected": 0,
            "total_embeddings_extracted": 0,
            "start_time": time.time(),
            "results": []
        }
        
        # Use detection_batch_size for processing batches
        effective_batch_size = detection_batch_size if detection_batch_size else batch_size
        
        # Process in batches
        for batch_start in range(0, len(image_files), effective_batch_size):
            batch_end = min(batch_start + effective_batch_size, len(image_files))
            batch_files = image_files[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//effective_batch_size + 1}/{(len(image_files)-1)//effective_batch_size + 1}")
            
            # Load batch images using the same pattern as batch_processor
            batch_data = []
            
            for img_path in batch_files:
                try:
                    with open(img_path, 'rb') as f:
                        image_data = f.read()
                    
                    image_np = bytes_to_image(image_data)
                    
                    batch_data.append({
                        'data': image_data,
                        'numpy': image_np,
                        'filename': img_path.name,
                        'path': img_path
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {e}")
                    stats["failed"] += 1
            
            if not batch_data:
                continue
                
            # Extract images and paths from batch_data
            batch_images = [item['numpy'] for item in batch_data]
            batch_paths = [item['path'] for item in batch_data]
                
            try:
                # 1. Detect faces
                start_time = time.time()
                detections = self.client.batch_detect_faces(batch_images)
                detection_time = time.time() - start_time
                
                # 2. Extract embeddings for all detected faces
                all_faces = []
                face_to_image_map = []
                
                for img_idx, faces in enumerate(detections):
                    for face in faces:
                        all_faces.append(face.face_aligned)
                        face_to_image_map.append(img_idx)
                
                embeddings = None
                extraction_time = 0
                
                if all_faces:
                    # Extract embeddings with chunking at directory processor level
                    start_time = time.time()
                    
                    all_embeddings = []
                    num_extraction_chunks = (len(all_faces) + extraction_batch_size - 1) // extraction_batch_size
                    
                    # Process faces in chunks to avoid exceeding extraction model batch size
                    for chunk_idx in range(num_extraction_chunks):
                        chunk_start = chunk_idx * extraction_batch_size
                        chunk_end = min(chunk_start + extraction_batch_size, len(all_faces))
                        chunk_faces = all_faces[chunk_start:chunk_end]
                        
                        if num_extraction_chunks > 1:
                            logger.debug(f"Processing extraction chunk {chunk_idx + 1}/{num_extraction_chunks} ({len(chunk_faces)} faces)")
                        
                        chunk_embeddings = self.client.batch_extract_embeddings(chunk_faces)
                        all_embeddings.append(chunk_embeddings)
                    
                    # Combine all chunks
                    if all_embeddings:
                        embeddings = np.vstack(all_embeddings)
                    else:
                        embeddings = np.array([]).reshape(0, 1024)
                    
                    extraction_time = time.time() - start_time
                    
                    if num_extraction_chunks > 1:
                        logger.info(f"Extracted embeddings for {len(all_faces)} faces using {num_extraction_chunks} chunks (max {extraction_batch_size} per chunk)")
                
                # 3. Organize results
                for img_idx, (img_path, faces) in enumerate(zip(batch_paths, detections)):
                    result = {
                        "image_path": str(img_path),
                        "image_name": img_path.name,
                        "num_faces": len(faces),
                        "faces": [],
                        "detection_time": detection_time / len(batch_images),
                        "extraction_time": extraction_time / max(1, len(all_faces)) if all_faces else 0
                    }
                    
                    face_start_idx = sum(len(detections[i]) for i in range(img_idx))
                    
                    for face_idx, face in enumerate(faces):
                        global_face_idx = face_start_idx + face_idx
                        
                        face_result = {
                            "bbox": face.bbox,
                            "confidence": face.confidence,
                            "face_size": face.face_size,
                            "landmarks": face.landmarks.tolist(),
                            "embedding": embeddings[global_face_idx].tolist() if embeddings is not None else None
                        }
                        result["faces"].append(face_result)
                        
                        # Save aligned face if requested
                        if save_results:
                            face_filename = f"{img_path.stem}_face_{face_idx}.jpg"
                            face_path = os.path.join(output_dir, face_filename)
                            cv2.imwrite(face_path, face.face_aligned)
                            np.save(face_path.split('.')[0], embeddings[global_face_idx])
                    
                    stats["results"].append(result)
                    stats["processed"] += 1
                    stats["total_faces_detected"] += len(faces)
                    stats["total_embeddings_extracted"] += len(faces)
                
                total_faces_in_batch = sum(len(d) for d in detections)
                logger.info(f"Batch processed: {len(batch_images)} images, "
                           f"{total_faces_in_batch} faces detected, "
                           f"detection: {detection_time:.3f}s, extraction: {extraction_time:.3f}s")
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                stats["failed"] += len(batch_images)
        
        # Final statistics
        stats["end_time"] = time.time()
        stats["total_time"] = stats["end_time"] - stats["start_time"]
        
        # Save results summary
        if save_results:
            summary_path = os.path.join(output_dir, "processing_summary.json")
            
            # Create JSON-serializable copy
            json_stats = {k: v for k, v in stats.items() if k != "results"}
            json_stats["sample_results"] = stats["results"][:5]  # Save only first 5 for space
            
            with open(summary_path, 'w') as f:
                json.dump(json_stats, f, indent=2, default=str)
            
            logger.info(f"Summary saved to: {summary_path}")
        
        return stats


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Standalone model test for face detection and extraction")
    parser.add_argument("directory", help="Directory containing images to process")
    parser.add_argument("--triton-url", default="localhost:8001", help="Triton server URL")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for image processing (deprecated, use --detection-batch-size)")
    parser.add_argument("--detection-batch-size", type=int, default=16, help="Batch size for detection model")
    parser.add_argument("--extraction-batch-size", type=int, default=32, help="Maximum batch size for extraction model")
    parser.add_argument("--recursive", action="store_true", help="Process subdirectories recursively")
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to files")
    
    # Model configuration
    parser.add_argument("--detection-model", default="detection", help="Detection model name")
    parser.add_argument("--extraction-model", default="extraction", help="Extraction model name")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    # Create model config
    model_config = ModelConfig(
        detection_model_name=args.detection_model,
        extraction_model_name=args.extraction_model
    )
    
    try:
        # Initialize processor
        processor = DirectoryProcessor(args.triton_url, model_config)
        
        # Process directory
        stats = processor.process_directory(
            directory_path=args.directory,
            batch_size=args.batch_size,
            detection_batch_size=args.detection_batch_size,
            extraction_batch_size=args.extraction_batch_size,
            recursive=args.recursive,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )
        
        # Print summary
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Total faces detected: {stats['total_faces_detected']}")
        logger.info(f"Total embeddings extracted: {stats['total_embeddings_extracted']}")
        logger.info(f"Total processing time: {stats['total_time']:.2f}s")
        
        if stats['processed'] > 0:
            avg_time_per_image = stats['total_time'] / stats['processed']
            logger.info(f"Average time per image: {avg_time_per_image:.3f}s")
            
            avg_faces_per_image = stats['total_faces_detected'] / stats['processed']
            logger.info(f"Average faces per image: {avg_faces_per_image:.2f}")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())