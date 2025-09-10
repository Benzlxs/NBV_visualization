"""Image loader utilities for camera images.

This module provides functionality for efficiently loading and managing
camera images from disk, with support for lazy loading, caching, and
batch operations.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Iterator
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class ImageLoader:
    """Loads and manages camera images for visualization."""
    
    def __init__(self, base_path: Path, image_extension: str = ".png"):
        """Initialize the image loader.
        
        Args:
            base_path: Base directory containing images
            image_extension: File extension for images (default: .png)
        """
        self.base_path = Path(base_path)
        self.image_extension = image_extension
        
        # Storage for loaded images
        # Using a dict for O(1) lookups by filename
        self.images: Dict[str, Image.Image] = {}
        
        # Cache for image metadata
        self.image_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Track loading progress
        self.loaded_count = 0
        self.total_count = 0
        
    def get_image_paths(self, filenames: List[str]) -> List[Path]:
        """Get full paths for image filenames.
        
        Args:
            filenames: List of image filenames from transforms.json
            
        Returns:
            List of Path objects for each image
        """
        paths = []
        for filename in filenames:
            # Keep the original extension if it exists and is an image format
            path_obj = Path(filename)
            if path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                # Keep original filename
                path = self.base_path / filename
            else:
                # Add default extension if no valid extension
                filename = path_obj.stem + self.image_extension
                path = self.base_path / filename
            
            paths.append(path)
        
        return paths
    
    def load_image(self, path: Path) -> Optional[Image.Image]:
        """Load a single image from disk.
        
        Args:
            path: Path to the image file
            
        Returns:
            PIL Image object or None if loading failed
        """
        try:
            # Check if file exists
            if not path.exists():
                logger.warning(f"Image not found: {path}")
                return None
            
            # Load image
            image = Image.open(path)
            
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Store metadata
            self.image_metadata[path.name] = {
                'size': image.size,
                'mode': image.mode,
                'format': image.format,
                'path': str(path)
            }
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to load image {path}: {e}")
            return None
    
    def load_images_batch(self, filenames: List[str], 
                         max_workers: int = 4,
                         show_progress: bool = True) -> Dict[str, Image.Image]:
        """Load multiple images in parallel.
        
        Args:
            filenames: List of image filenames to load
            max_workers: Number of parallel threads for loading
            show_progress: Whether to show loading progress
            
        Returns:
            Dictionary mapping filename to loaded Image
        """
        # Get full paths
        paths = self.get_image_paths(filenames)
        self.total_count = len(paths)
        self.loaded_count = 0
        
        # Track start time for progress reporting
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel loading
        # Images are I/O bound, so threads are appropriate
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all load tasks
            future_to_path = {
                executor.submit(self.load_image, path): (path, filename)
                for path, filename in zip(paths, filenames)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                path, filename = future_to_path[future]
                
                try:
                    image = future.result()
                    if image is not None:
                        self.images[filename] = image
                        self.loaded_count += 1
                    
                    # Show progress if requested
                    if show_progress and self.loaded_count % 10 == 0:
                        progress = (self.loaded_count / self.total_count) * 100
                        elapsed = time.time() - start_time
                        rate = self.loaded_count / elapsed if elapsed > 0 else 0
                        logger.info(f"Loading images: {self.loaded_count}/{self.total_count} "
                                  f"({progress:.1f}%) - {rate:.1f} images/sec")
                        
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Final progress report
        elapsed = time.time() - start_time
        logger.info(f"Loaded {self.loaded_count}/{self.total_count} images "
                   f"in {elapsed:.2f} seconds")
        
        return self.images
    
    def load_images_lazy(self, filenames: List[str]) -> 'LazyImageLoader':
        """Create a lazy loader for images.
        
        This returns an iterator that loads images on-demand,
        which is useful for very large datasets.
        
        Args:
            filenames: List of image filenames
            
        Returns:
            LazyImageLoader instance
        """
        paths = self.get_image_paths(filenames)
        return LazyImageLoader(paths, self)
    
    def get_image(self, filename: str) -> Optional[Image.Image]:
        """Get a loaded image by filename.
        
        Args:
            filename: Image filename
            
        Returns:
            Image if loaded, None otherwise
        """
        return self.images.get(filename)
    
    def get_image_size(self, filename: str) -> Optional[Tuple[int, int]]:
        """Get image dimensions without loading full image.
        
        Args:
            filename: Image filename
            
        Returns:
            Tuple of (width, height) or None
        """
        # Check if we have metadata cached
        if filename in self.image_metadata:
            return self.image_metadata[filename]['size']
        
        # Try to load just the header to get size
        path = self.base_path / filename
        try:
            with Image.open(path) as img:
                size = img.size
                # Cache the metadata
                self.image_metadata[filename] = {
                    'size': size,
                    'mode': img.mode,
                    'format': img.format
                }
                return size
        except Exception as e:
            logger.error(f"Failed to get image size for {filename}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all loaded images from memory."""
        self.images.clear()
        self.loaded_count = 0
        logger.info("Image cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded images.
        
        Returns:
            Dictionary with loading statistics
        """
        stats = {
            'total_images': self.total_count,
            'loaded_images': self.loaded_count,
            'cached_images': len(self.images),
            'metadata_entries': len(self.image_metadata),
        }
        
        # Add size statistics if we have metadata
        if self.image_metadata:
            sizes = [meta['size'] for meta in self.image_metadata.values()]
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]
            
            stats['image_dimensions'] = {
                'min_width': min(widths),
                'max_width': max(widths),
                'min_height': min(heights),
                'max_height': max(heights),
                'common_size': max(set(sizes), key=sizes.count)  # Most common size
            }
        
        return stats


class LazyImageLoader:
    """Iterator for lazy loading of images.
    
    This class provides on-demand loading of images, useful for
    large datasets where loading all images at once would use
    too much memory.
    """
    
    def __init__(self, paths: List[Path], loader: ImageLoader):
        """Initialize lazy loader.
        
        Args:
            paths: List of image paths
            loader: Parent ImageLoader instance
        """
        self.paths = paths
        self.loader = loader
        self.index = 0
    
    def __iter__(self) -> Iterator[Tuple[str, Optional[Image.Image]]]:
        """Iterate over images, loading them on demand.
        
        Yields:
            Tuple of (filename, image) for each image
        """
        for path in self.paths:
            image = self.loader.load_image(path)
            yield path.name, image
    
    def __len__(self) -> int:
        """Get number of images."""
        return len(self.paths)
    
    def get_at_index(self, index: int) -> Tuple[str, Optional[Image.Image]]:
        """Get image at specific index.
        
        Args:
            index: Image index
            
        Returns:
            Tuple of (filename, image)
        """
        if 0 <= index < len(self.paths):
            path = self.paths[index]
            image = self.loader.load_image(path)
            return path.name, image
        else:
            raise IndexError(f"Index {index} out of range")


def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (128, 128)) -> Image.Image:
    """Create a thumbnail from an image.
    
    Args:
        image: Original image
        size: Target thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    # Create a copy to avoid modifying original
    thumbnail = image.copy()
    
    # Use LANCZOS resampling for better quality
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    
    return thumbnail


def images_to_grid(images: List[Image.Image], 
                  cols: int = 10,
                  padding: int = 2) -> Image.Image:
    """Arrange multiple images in a grid.
    
    Useful for creating overview visualizations of all camera images.
    
    Args:
        images: List of PIL images (should be same size)
        cols: Number of columns in grid
        padding: Padding between images in pixels
        
    Returns:
        Single image containing the grid
    """
    if not images:
        raise ValueError("No images provided")
    
    # Get image dimensions (assume all same size)
    img_width, img_height = images[0].size
    
    # Calculate grid dimensions
    num_images = len(images)
    rows = (num_images + cols - 1) // cols  # Ceiling division
    
    # Adjust cols if we have fewer images than columns
    actual_cols = min(cols, num_images)
    
    # Calculate output image size
    # For a single row/column, no padding is needed
    grid_width = actual_cols * img_width + max(0, (actual_cols - 1) * padding)
    grid_height = rows * img_height + max(0, (rows - 1) * padding)
    
    # Create output image
    grid = Image.new('RGB', (grid_width, grid_height), color=(128, 128, 128))
    
    # Place images in grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * (img_width + padding)
        y = row * (img_height + padding)
        
        grid.paste(img, (x, y))
    
    return grid