import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from typing import List, Tuple
import folder_paths

# TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    
    HAS_TENSORRT = True
    
    # Initialize CUDA and create a context manually
    try:
        cuda.init()
        device = cuda.Device(0)
        CUDA_CONTEXT = device.make_context()
        
        # Test basic CUDA operations
        test_mem = cuda.mem_alloc(1024)
        test_mem.free()
        
        # Pop the context so it's available for later use
        cuda.Context.pop()
        
    except Exception as ctx_e:
        # Fallback to autoinit
        try:
            import pycuda.autoinit
            CUDA_CONTEXT = cuda.Context.get_current()
        except Exception as auto_e:
            CUDA_CONTEXT = None
    
except ImportError as e:
    HAS_TENSORRT = False
    CUDA_CONTEXT = None
except Exception as e:
    HAS_TENSORRT = False
    CUDA_CONTEXT = None

# PIL for image processing
from PIL import Image

# Setup logging
logger = logging.getLogger('ComfyUI.TRTUpscaler')
TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if HAS_TENSORRT else None


class TRTUpscaler:
    """
    TensorRT-accelerated image upscaler node for ComfyUI.
    Supports batch processing and various upscaling models.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get available model files
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        model_files = []
        
        if os.path.exists(models_dir):
            # Look for .trt engine files
            for f in os.listdir(models_dir):
                if f.endswith('.trt'):
                    # Remove extension and precision suffix
                    model_name = f.replace('.trt', '')
                    if model_name.endswith('_fp16') or model_name.endswith('_fp32'):
                        model_name = model_name[:-5]
                    if model_name not in model_files:
                        model_files.append(model_name)
        
        if not model_files:
            model_files = ["No TRT models found"]
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_files,),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Size of tiles for processing large images"
                }),
                "tile_overlap": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 8,
                    "tooltip": "Overlap between tiles to reduce seams"
                }),
                "use_fp16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use FP16 precision (faster but may have slight quality loss)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    
    def __init__(self):
        self.engine_cache = {}
        self.context_cache = {}
    
    def upscale(self, image: torch.Tensor, model_name: str, tile_size: int = 512, 
                tile_overlap: int = 32, use_fp16: bool = True) -> Tuple[torch.Tensor]:
        """
        Upscale images using TensorRT acceleration.
        
        Args:
            image: Input images tensor [B, H, W, C]
            model_name: Name of the TRT model to use
            tile_size: Size of tiles for processing
            tile_overlap: Overlap between tiles
            use_fp16: Whether to use FP16 precision
            
        Returns:
            Tuple containing upscaled images tensor
        """
        if not HAS_TENSORRT:
            raise RuntimeError("TensorRT is not available. Please install TensorRT and pycuda.")
        
        if model_name == "No TRT models found":
            raise ValueError("No TRT model files found in upscale_models directory.")
        
        # Get engine path
        models_dir = folder_paths.get_folder_paths("upscale_models")[0]
        precision_suffix = "_fp16" if use_fp16 else "_fp32"
        engine_path = os.path.join(models_dir, f"{model_name}{precision_suffix}.trt")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TRT engine file not found: {engine_path}")
        
        # Ensure consistent CUDA context throughout the entire process
        active_context = None
        context_needs_cleanup = False
        
        try:
            # Check if we already have an active context
            active_context = cuda.Context.get_current()
        except Exception as e:
            active_context = None
            
        # If no active context, we need to establish one
        if active_context is None:
            # Try to use the saved context or create a new one
            if CUDA_CONTEXT is not None:
                CUDA_CONTEXT.push()
                context_needs_cleanup = True
                active_context = CUDA_CONTEXT
            else:
                try:
                    cuda.init()
                    device = cuda.Device(0)
                    new_context = device.make_context()
                    context_needs_cleanup = True
                    active_context = new_context
                except Exception as e:
                    raise RuntimeError(f"Failed to create CUDA context: {e}")
        
        try:
            # Load engine and create context in the active CUDA context
            engine_key = engine_path
            if engine_key not in self.engine_cache:
                engine = self._load_engine(engine_path)
                if engine is None:
                    raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")
                self.engine_cache[engine_key] = engine
                
                context = engine.create_execution_context()
                if context is None:
                    raise RuntimeError("Failed to create TensorRT execution context. Likely out of GPU memory.")
                self.context_cache[engine_key] = context
            
            engine = self.engine_cache[engine_key]
            context = self.context_cache[engine_key]
            
            # Get scale factor from model name
            scale = self._get_scale_from_model_name(model_name)
            
            # Process batch
            batch_size = image.shape[0]
            results = []
            
            for i in range(batch_size):
                # Extract single image [H, W, C]
                single_image = image[i]
                
                # Convert to numpy and ensure contiguous
                img_np = single_image.cpu().numpy()
                img_np = np.ascontiguousarray(img_np)
                
                # Ensure proper format (0-1 range, RGB)
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
                
                # Process with tiling
                upscaled = self._process_image_with_tiles(
                    img_np, engine, context, scale, tile_size, tile_overlap
                )
                
                # Convert back to torch tensor
                upscaled_tensor = torch.from_numpy(upscaled).to(image.device)
                results.append(upscaled_tensor)
            
            # Stack results back into batch
            output = torch.stack(results, dim=0)
            
        except Exception as e:
            raise
        finally:
            # Clean up CUDA context if we created one
            if context_needs_cleanup:
                try:
                    cuda.Context.pop()
                except Exception as e:
                    pass
        
        return (output,)
    
    def _load_engine(self, engine_path: str):
        """Load TensorRT engine from file"""
        runtime = trt.Runtime(TRT_LOGGER)
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            return runtime.deserialize_cuda_engine(engine_data)
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def _get_scale_from_model_name(self, model_name: str) -> int:
        """Extract scale factor from model name"""
        model_name_lower = model_name.lower()
        if "4x" in model_name_lower:
            return 4
        elif "2x" in model_name_lower:
            return 2
        elif "8x" in model_name_lower:
            return 8
        else:
            return 4
    
    def _process_image_with_tiles(self, img_np: np.ndarray, engine, context, 
                                  scale: int, tile_size: int, overlap: int) -> np.ndarray:
        """Process image using tiled approach for large images"""
        height, width = img_np.shape[:2]
        
        # Calculate output dimensions
        out_height = height * scale
        out_width = width * scale
        
        # Initialize output buffer
        output = np.zeros((out_height, out_width, 3), dtype=np.float32)
        weight_map = np.zeros((out_height, out_width, 3), dtype=np.float32)
        
        # Process tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                # Extract tile
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                
                tile = img_np[y:y_end, x:x_end]
                tile_h, tile_w = tile.shape[:2]
                
                # Handle small tiles with padding
                min_size = 64
                if tile_h < min_size or tile_w < min_size:
                    pad_h = max(0, min_size - tile_h)
                    pad_w = max(0, min_size - tile_w)
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
                
                # Preprocess tile
                tile_input = self._preprocess(tile)
                
                # Run inference
                tile_output = self._infer_tile(tile_input, engine, context, scale)
                
                # Remove padding if applied
                if tile_h < min_size or tile_w < min_size:
                    original_output_h = tile_h * scale
                    original_output_w = tile_w * scale
                    tile_output = tile_output[:original_output_h, :original_output_w]
                
                # Calculate output position
                out_y = y * scale
                out_y_end = y_end * scale
                out_x = x * scale
                out_x_end = x_end * scale
                
                # Create feather mask for blending
                mask = self._create_feather_mask(
                    tile_output.shape[:2],
                    x > 0,
                    x_end < width,
                    y > 0,
                    y_end < height,
                    overlap * scale // 2
                )
                
                # Accumulate output with blending
                output[out_y:out_y_end, out_x:out_x_end] += tile_output * mask[:, :, np.newaxis]
                weight_map[out_y:out_y_end, out_x:out_x_end] += mask[:, :, np.newaxis]
        
        # Normalize by weights
        output = output / np.maximum(weight_map, 1e-8)
        
        # Ensure output is in valid range
        output = np.clip(output, 0, 1)
        
        return output
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TensorRT (HWC to CHW, add batch dimension)"""
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        # Ensure float32 and contiguous
        image = np.ascontiguousarray(image, dtype=np.float32)
        return image
    
    def _infer_tile(self, tile_input: np.ndarray, engine, context, scale: int) -> np.ndarray:
        """Run inference on a single tile"""
        batch_size, channels, h, w = tile_input.shape
        
        # Get input/output names
        input_name = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        
        # Set input shape
        context.set_input_shape(input_name, tile_input.shape)
        
        # Calculate output shape
        out_h = h * scale
        out_w = w * scale
        output_shape = (batch_size, channels, out_h, out_w)
        
        # Allocate GPU memory
        input_size = tile_input.nbytes
        output_size = int(np.prod(output_shape) * 4)  # float32
        
        try:
            d_input = cuda.mem_alloc(input_size)
            d_output = cuda.mem_alloc(output_size)
        except cuda.MemoryError as e:
            raise RuntimeError(f"GPU memory allocation failed. Try reducing tile size. Error: {e}")
        
        try:
            # Create bindings
            bindings = [int(d_input), int(d_output)]
            
            # Copy input to GPU
            cuda.memcpy_htod(d_input, tile_input)
            
            # Run inference with newer TensorRT API for better context handling
            try:
                # Set tensor addresses for newer API
                context.set_tensor_address(input_name, int(d_input))
                context.set_tensor_address(output_name, int(d_output))
                
                # Create CUDA stream for execution
                stream = cuda.Stream()
                success = context.execute_v3(stream)
                stream.synchronize()
                
            except AttributeError:
                # Fallback to execute_v2 if execute_v3 not available
                success = context.execute_v2(bindings)
            except Exception as e:
                success = context.execute_v2(bindings)
            
            if not success:
                # Get detailed error information
                error_details = []
                try:
                    error_recorder = context.get_error_recorder()
                    if error_recorder:
                        num_errors = error_recorder.get_num_errors()
                        for i in range(num_errors):
                            error = error_recorder.get_error(i)
                            error_details.append(f"Error {i}: {error}")
                except:
                    pass
                
                # Provide helpful error message
                error_msg = "TensorRT inference failed"
                if error_details:
                    error_msg += f": {'; '.join(error_details)}"
                else:
                    error_msg += " - possible causes: GPU memory insufficient, context inconsistency, or model incompatibility"
                
                # Add memory information if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    error_msg += f" (GPU memory: {mem_info.used//1024**2}MB used / {mem_info.total//1024**2}MB total)"
                except:
                    pass
                
                raise RuntimeError(error_msg)
            
            # Copy output from GPU
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh(output, d_output)
            
            # Remove batch dimension and convert CHW to HWC
            output = output[0].transpose(1, 2, 0)
            
            return output
            
        finally:
            # Always free GPU memory
            try:
                d_input.free()
            except Exception as e:
                pass
            try:
                d_output.free()
            except Exception as e:
                pass
    
    def _create_feather_mask(self, shape: tuple, has_left: bool, has_right: bool,
                             has_top: bool, has_bottom: bool, feather_size: int) -> np.ndarray:
        """Create feathering mask for tile blending"""
        h, w = shape
        mask = np.ones((h, w), dtype=np.float32)
        
        if feather_size > 0:
            # Left edge
            if has_left:
                for i in range(min(feather_size, w)):
                    mask[:, i] *= i / feather_size
            
            # Right edge
            if has_right:
                for i in range(min(feather_size, w)):
                    mask[:, -(i+1)] *= i / feather_size
            
            # Top edge
            if has_top:
                for i in range(min(feather_size, h)):
                    mask[i, :] *= i / feather_size
            
            # Bottom edge
            if has_bottom:
                for i in range(min(feather_size, h)):
                    mask[-(i+1), :] *= i / feather_size
        
        return mask
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution if model changes
        return float("nan")


NODE_CLASS_MAPPINGS = {
    "TRTUpscaler": TRTUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TRTUpscaler": "TRT Upscaler"
}