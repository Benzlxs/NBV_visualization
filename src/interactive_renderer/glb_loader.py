"""GLB Model Loader for interactive rendering system.

This module provides functionality to load and parse GLB 3D models from the ABO dataset
for use in the interactive camera pose manipulation system.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json

try:
    import trimesh
    import pygltflib
except ImportError as e:
    logging.error(f"Required GLB libraries not installed: {e}")
    logging.error("Install with: pip install trimesh pygltflib")
    raise

logger = logging.getLogger(__name__)


class GLBModel:
    """Container for a loaded GLB 3D model with all its components."""
    
    def __init__(
        self,
        file_path: Path,
        mesh: Optional[trimesh.Trimesh] = None,
        materials: Optional[List[Dict[str, Any]]] = None,
        textures: Optional[Dict[str, np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize GLB model container.
        
        Args:
            file_path: Path to the original GLB file
            mesh: Trimesh geometry object
            materials: List of material definitions
            textures: Dictionary of texture arrays keyed by name
            metadata: Additional model metadata
        """
        self.file_path = file_path
        self.mesh = mesh
        self.materials = materials or []
        self.textures = textures or {}
        self.metadata = metadata or {}
        
        # Compute model statistics
        self._compute_statistics()
    
    def _compute_statistics(self) -> None:
        """Compute and cache model statistics."""
        if self.mesh is not None:
            self.vertex_count = len(self.mesh.vertices)
            self.face_count = len(self.mesh.faces)
            self.bounds = self.mesh.bounds
            self.bounding_box_size = self.mesh.extents
            self.centroid = self.mesh.centroid
            self.volume = self.mesh.volume if self.mesh.is_volume else 0.0
            self.is_watertight = self.mesh.is_watertight
        else:
            self.vertex_count = 0
            self.face_count = 0
            self.bounds = np.array([[0, 0, 0], [0, 0, 0]])
            self.bounding_box_size = np.array([0, 0, 0])
            self.centroid = np.array([0, 0, 0])
            self.volume = 0.0
            self.is_watertight = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics summary.
        
        Returns:
            Dictionary containing model statistics
        """
        return {
            'file_path': str(self.file_path),
            'vertex_count': self.vertex_count,
            'face_count': self.face_count,
            'bounds': self.bounds.tolist(),
            'bounding_box_size': self.bounding_box_size.tolist(),
            'centroid': self.centroid.tolist(),
            'volume': self.volume,
            'is_watertight': self.is_watertight,
            'material_count': len(self.materials),
            'texture_count': len(self.textures),
            'has_geometry': self.mesh is not None,
            'file_size_mb': self.file_path.stat().st_size / (1024 * 1024) if self.file_path.exists() else 0
        }
    
    def normalize_to_unit_sphere(self) -> 'GLBModel':
        """Create a copy of the model normalized to fit in a unit sphere.
        
        Returns:
            New GLBModel instance with normalized geometry
        """
        if self.mesh is None:
            logger.warning("Cannot normalize model without geometry")
            return self
        
        # Create a copy of the mesh
        normalized_mesh = self.mesh.copy()
        
        # Center the mesh at origin
        normalized_mesh.vertices = normalized_mesh.vertices.astype(np.float64) - self.centroid.astype(np.float64)
        
        # Scale to fit in unit sphere
        max_extent = np.max(self.bounding_box_size)
        if max_extent > 0:
            scale_factor = 2.0 / max_extent  # Fit in sphere of radius 1
            normalized_mesh.vertices = normalized_mesh.vertices * scale_factor
        
        # Create new model with normalized mesh
        return GLBModel(
            file_path=self.file_path,
            mesh=normalized_mesh,
            materials=self.materials,
            textures=self.textures,
            metadata={**self.metadata, 'normalized': True, 'scale_factor': scale_factor if max_extent > 0 else 1.0}
        )


class GLBLoader:
    """GLB model loader with support for ABO dataset models."""
    
    def __init__(self):
        """Initialize GLB loader."""
        self.loaded_models: Dict[str, GLBModel] = {}
        self.load_statistics = {
            'total_attempts': 0,
            'successful_loads': 0,
            'failed_loads': 0,
            'cached_loads': 0
        }
    
    def load_model(
        self,
        glb_path: Union[str, Path],
        use_cache: bool = True,
        validate_model: bool = True
    ) -> GLBModel:
        """Load a GLB model from file.
        
        Args:
            glb_path: Path to GLB file
            use_cache: Whether to use cached model if available
            validate_model: Whether to validate model structure
            
        Returns:
            Loaded GLB model
            
        Raises:
            FileNotFoundError: If GLB file doesn't exist
            ValueError: If GLB file is invalid or corrupt
        """
        glb_path = Path(glb_path)
        self.load_statistics['total_attempts'] += 1
        
        # Check if file exists
        if not glb_path.exists():
            self.load_statistics['failed_loads'] += 1
            raise FileNotFoundError(f"GLB file not found: {glb_path}")
        
        # Check cache
        cache_key = str(glb_path.absolute())
        if use_cache and cache_key in self.loaded_models:
            self.load_statistics['cached_loads'] += 1
            logger.debug(f"Returning cached GLB model: {glb_path.name}")
            return self.loaded_models[cache_key]
        
        try:
            logger.info(f"Loading GLB model: {glb_path.name}")
            
            # Load with trimesh (primary method)
            model = self._load_with_trimesh(glb_path, validate_model)
            
            # Cache the model
            if use_cache:
                self.loaded_models[cache_key] = model
            
            self.load_statistics['successful_loads'] += 1
            logger.info(f"Successfully loaded GLB model: {glb_path.name}")
            
            return model
            
        except Exception as e:
            self.load_statistics['failed_loads'] += 1
            error_msg = f"Failed to load GLB model {glb_path.name}: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _load_with_trimesh(self, glb_path: Path, validate: bool) -> GLBModel:
        """Load GLB model using trimesh library.
        
        Args:
            glb_path: Path to GLB file
            validate: Whether to validate the model
            
        Returns:
            Loaded GLB model
        """
        try:
            # Load the GLB file with trimesh
            scene = trimesh.load(str(glb_path), force='scene')
            
            # Extract geometry - combine all meshes in the scene
            if hasattr(scene, 'geometry') and scene.geometry:
                # Multiple meshes - combine them
                meshes = []
                for geometry in scene.geometry.values():
                    if hasattr(geometry, 'vertices'):
                        meshes.append(geometry)
                
                if meshes:
                    # Combine all meshes
                    combined_mesh = trimesh.util.concatenate(meshes)
                else:
                    logger.warning(f"No valid meshes found in GLB file: {glb_path.name}")
                    combined_mesh = None
            elif hasattr(scene, 'vertices'):
                # Single mesh
                combined_mesh = scene
            else:
                logger.warning(f"No geometry found in GLB file: {glb_path.name}")
                combined_mesh = None
            
            # Extract materials (basic implementation)
            materials = self._extract_materials(scene)
            
            # Extract textures (basic implementation)
            textures = self._extract_textures(scene)
            
            # Create model metadata
            metadata = {
                'loader': 'trimesh',
                'scene_geometry_count': len(scene.geometry) if hasattr(scene, 'geometry') else 0,
                'original_units': getattr(scene, 'units', 'unknown')
            }
            
            # Create GLB model
            model = GLBModel(
                file_path=glb_path,
                mesh=combined_mesh,
                materials=materials,
                textures=textures,
                metadata=metadata
            )
            
            # Validate if requested
            if validate:
                self._validate_model(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Trimesh loading failed for {glb_path.name}: {e}")
            raise
    
    def _extract_materials(self, scene: Any) -> List[Dict[str, Any]]:
        """Extract material information from scene.
        
        Args:
            scene: Trimesh scene object
            
        Returns:
            List of material dictionaries
        """
        materials = []
        
        # Basic material extraction - trimesh has limited material support
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'visual') and geometry.visual is not None:
                    material_info = {
                        'name': name,
                        'type': type(geometry.visual).__name__
                    }
                    
                    # Extract color if available
                    if hasattr(geometry.visual, 'main_color'):
                        material_info['color'] = geometry.visual.main_color.tolist()
                    
                    materials.append(material_info)
        
        return materials
    
    def _extract_textures(self, scene: Any) -> Dict[str, np.ndarray]:
        """Extract texture information from scene.
        
        Args:
            scene: Trimesh scene object
            
        Returns:
            Dictionary of texture arrays
        """
        textures = {}
        
        # Basic texture extraction - trimesh has limited texture support
        if hasattr(scene, 'geometry'):
            for name, geometry in scene.geometry.items():
                if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                    visual_material = geometry.visual.material
                    if hasattr(visual_material, 'image') and visual_material.image is not None:
                        textures[f"{name}_texture"] = np.array(visual_material.image)
        
        return textures
    
    def _validate_model(self, model: GLBModel) -> None:
        """Validate loaded model structure.
        
        Args:
            model: GLB model to validate
            
        Raises:
            ValueError: If model validation fails
        """
        if model.mesh is None:
            raise ValueError("Model has no geometry")
        
        if model.vertex_count == 0:
            raise ValueError("Model has no vertices")
        
        if model.face_count == 0:
            raise ValueError("Model has no faces")
        
        # Check for reasonable model size
        max_extent = np.max(model.bounding_box_size)
        if max_extent > 1000:
            logger.warning(f"Model is very large: {max_extent} units")
        
        if max_extent < 0.001:
            logger.warning(f"Model is very small: {max_extent} units")
        
        logger.debug(f"Model validation passed: {model.vertex_count} vertices, {model.face_count} faces")
    
    def load_abo_model(self, model_id: str, abo_dataset_path: Optional[Path] = None) -> GLBModel:
        """Load a specific ABO model by ID.
        
        Args:
            model_id: ABO model identifier (e.g., 'B01LR5RSG0')
            abo_dataset_path: Optional path to ABO dataset root
            
        Returns:
            Loaded GLB model
            
        Raises:
            FileNotFoundError: If ABO model file not found
        """
        if abo_dataset_path is None:
            # Use default ABO dataset path
            abo_dataset_path = Path("dataset/ABO/raw/3dmodels/original")
        
        # Find the model file in ABO directory structure
        model_path = self._find_abo_model_path(model_id, abo_dataset_path)
        
        if model_path is None:
            raise FileNotFoundError(f"ABO model {model_id} not found in {abo_dataset_path}")
        
        return self.load_model(model_path)
    
    def _find_abo_model_path(self, model_id: str, abo_path: Path) -> Optional[Path]:
        """Find the path to an ABO model file.
        
        Args:
            model_id: ABO model identifier
            abo_path: Path to ABO 3D models directory
            
        Returns:
            Path to GLB file if found, None otherwise
        """
        # ABO models are organized in subdirectories 0-9
        for subdir in range(10):
            potential_path = abo_path / str(subdir) / f"{model_id}.glb"
            if potential_path.exists():
                return potential_path
        
        return None
    
    def list_available_abo_models(self, abo_dataset_path: Optional[Path] = None) -> List[str]:
        """List all available ABO model IDs.
        
        Args:
            abo_dataset_path: Optional path to ABO dataset root
            
        Returns:
            List of available model IDs
        """
        if abo_dataset_path is None:
            abo_dataset_path = Path("dataset/ABO/raw/3dmodels/original")
        
        model_ids = []
        
        # Scan all subdirectories for GLB files
        for subdir in range(10):
            subdir_path = abo_dataset_path / str(subdir)
            if subdir_path.exists():
                for glb_file in subdir_path.glob("*.glb"):
                    model_ids.append(glb_file.stem)
        
        return sorted(model_ids)
    
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get loader statistics.
        
        Returns:
            Dictionary containing load statistics
        """
        return {
            **self.load_statistics,
            'cached_models': len(self.loaded_models),
            'cache_hit_rate': (
                self.load_statistics['cached_loads'] / self.load_statistics['total_attempts']
                if self.load_statistics['total_attempts'] > 0 else 0.0
            ),
            'success_rate': (
                self.load_statistics['successful_loads'] / self.load_statistics['total_attempts']
                if self.load_statistics['total_attempts'] > 0 else 0.0
            )
        }
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        self.loaded_models.clear()
        logger.info("GLB model cache cleared")
    
    def validate_glb_file(self, glb_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a GLB file without fully loading it.
        
        Args:
            glb_path: Path to GLB file
            
        Returns:
            Dictionary containing validation results
        """
        glb_path = Path(glb_path)
        validation_result = {
            'path': str(glb_path),
            'exists': glb_path.exists(),
            'is_file': glb_path.is_file() if glb_path.exists() else False,
            'size_bytes': glb_path.stat().st_size if glb_path.exists() else 0,
            'is_glb': glb_path.suffix.lower() == '.glb',
            'can_load': False,
            'error': None
        }
        
        if not validation_result['exists']:
            validation_result['error'] = "File does not exist"
            return validation_result
        
        if not validation_result['is_file']:
            validation_result['error'] = "Path is not a file"
            return validation_result
        
        if not validation_result['is_glb']:
            validation_result['error'] = "File does not have .glb extension"
            return validation_result
        
        if validation_result['size_bytes'] == 0:
            validation_result['error'] = "File is empty"
            return validation_result
        
        # Try to load with trimesh to check validity
        try:
            scene = trimesh.load(str(glb_path), force='scene')
            validation_result['can_load'] = True
            
            # Add basic scene information
            if hasattr(scene, 'geometry'):
                validation_result['geometry_count'] = len(scene.geometry)
            else:
                validation_result['geometry_count'] = 1 if hasattr(scene, 'vertices') else 0
                
        except Exception as e:
            validation_result['error'] = str(e)
        
        return validation_result


# Convenience functions for common use cases

def load_abo_model(model_id: str, abo_dataset_path: Optional[Path] = None) -> GLBModel:
    """Convenience function to load an ABO model.
    
    Args:
        model_id: ABO model identifier
        abo_dataset_path: Optional path to ABO dataset
        
    Returns:
        Loaded GLB model
    """
    loader = GLBLoader()
    return loader.load_abo_model(model_id, abo_dataset_path)


def validate_glb_file(glb_path: Union[str, Path]) -> Dict[str, Any]:
    """Convenience function to validate a GLB file.
    
    Args:
        glb_path: Path to GLB file
        
    Returns:
        Validation results
    """
    loader = GLBLoader()
    return loader.validate_glb_file(glb_path)