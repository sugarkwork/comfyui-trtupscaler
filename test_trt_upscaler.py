#!/usr/bin/env python3
"""
Unit tests for TensorRT Upscaler ComfyUI node
"""

import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Mock ComfyUI dependencies
sys.modules['folder_paths'] = MagicMock()
sys.modules['folder_paths'].get_folder_paths = MagicMock(return_value=[str(Path(__file__).parent / "test_models")])

# Mock TensorRT if not available
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_TENSORRT = True
except ImportError:
    # Create mock modules
    sys.modules['tensorrt'] = MagicMock()
    sys.modules['pycuda'] = MagicMock()
    sys.modules['pycuda.driver'] = MagicMock()
    sys.modules['pycuda.autoinit'] = MagicMock()
    HAS_TENSORRT = False

# Now import our module
from nodes import TRTUpscaler, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


class TestTRTUpscaler(unittest.TestCase):
    """Test cases for TRTUpscaler node"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = Path(__file__).parent / "test_models"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create dummy TRT engine files for testing
        self.create_dummy_engine_files()
        
        # Initialize node
        self.node = TRTUpscaler()
    
    def tearDown(self):
        """Clean up test files"""
        # Remove test files
        if self.test_dir.exists():
            for file in self.test_dir.glob("*.trt"):
                file.unlink()
    
    def create_dummy_engine_files(self):
        """Create dummy .trt files for testing"""
        dummy_models = [
            "test_model_2x_fp16.trt",
            "test_model_2x_fp32.trt",
            "test_model_4x_fp16.trt",
            "test_model_4x_fp32.trt",
        ]
        
        for model in dummy_models:
            (self.test_dir / model).write_bytes(b"dummy engine data")
    
    def test_node_registration(self):
        """Test that node is properly registered"""
        self.assertIn("TRTUpscaler", NODE_CLASS_MAPPINGS)
        self.assertIn("TRTUpscaler", NODE_DISPLAY_NAME_MAPPINGS)
        self.assertEqual(NODE_CLASS_MAPPINGS["TRTUpscaler"], TRTUpscaler)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TRTUpscaler"], "TRT Upscaler")
    
    def test_input_types(self):
        """Test INPUT_TYPES class method"""
        with patch('folder_paths.get_folder_paths', return_value=[str(self.test_dir)]):
            input_types = TRTUpscaler.INPUT_TYPES()
            
            # Check structure
            self.assertIn("required", input_types)
            required = input_types["required"]
            
            # Check required inputs
            self.assertIn("image", required)
            self.assertIn("model_name", required)
            self.assertIn("tile_size", required)
            self.assertIn("tile_overlap", required)
            self.assertIn("use_fp16", required)
            
            # Check model names are parsed correctly
            model_names = required["model_name"][0]
            self.assertIn("test_model_2x", model_names)
            self.assertIn("test_model_4x", model_names)
    
    def test_return_types(self):
        """Test return types and function name"""
        self.assertEqual(TRTUpscaler.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(TRTUpscaler.RETURN_NAMES, ("upscaled_image",))
        self.assertEqual(TRTUpscaler.FUNCTION, "upscale")
        self.assertEqual(TRTUpscaler.CATEGORY, "image/upscaling")
    
    def test_scale_detection(self):
        """Test scale factor detection from model names"""
        test_cases = [
            ("test_2x_model", 2),
            ("4x_ultrasharp", 4),
            ("model_8x_super", 8),
            ("unknown_model", 4),  # default
        ]
        
        for model_name, expected_scale in test_cases:
            scale = self.node._get_scale_from_model_name(model_name)
            self.assertEqual(scale, expected_scale)
    
    def test_preprocess(self):
        """Test image preprocessing"""
        # Create test image (HWC format)
        test_image = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Preprocess
        processed = self.node._preprocess(test_image)
        
        # Check shape (should be BCHW)
        self.assertEqual(processed.shape, (1, 3, 256, 256))
        self.assertEqual(processed.dtype, np.float32)
        
        # Check memory layout
        self.assertTrue(processed.flags['C_CONTIGUOUS'])
    
    def test_create_feather_mask(self):
        """Test feather mask creation"""
        # Test without feathering
        mask = self.node._create_feather_mask((100, 100), False, False, False, False, 10)
        self.assertTrue(np.all(mask == 1.0))
        
        # Test with left feathering
        mask = self.node._create_feather_mask((100, 100), True, False, False, False, 10)
        self.assertLess(mask[50, 0], 1.0)  # Left edge should be feathered
        self.assertEqual(mask[50, 50], 1.0)  # Center should be 1.0
        
        # Test with all edges feathered
        mask = self.node._create_feather_mask((100, 100), True, True, True, True, 10)
        self.assertLess(mask[0, 0], 1.0)  # Corners should be feathered most
        self.assertEqual(mask[50, 50], 1.0)  # Center should still be 1.0
    
    @patch('nodes.HAS_TENSORRT', False)
    def test_no_tensorrt_error(self):
        """Test error when TensorRT is not available"""
        dummy_image = torch.rand(1, 256, 256, 3)
        
        with self.assertRaises(RuntimeError) as context:
            self.node.upscale(dummy_image, "test_model", 512, 32, True)
        
        self.assertIn("TensorRT is not available", str(context.exception))
    
    @patch('nodes.HAS_TENSORRT', True)
    def test_no_models_error(self):
        """Test error when no models are found"""
        dummy_image = torch.rand(1, 256, 256, 3)
        
        with self.assertRaises(ValueError) as context:
            self.node.upscale(dummy_image, "No TRT models found", 512, 32, True)
        
        self.assertIn("No TRT model files found", str(context.exception))
    
    @patch('nodes.HAS_TENSORRT', True)
    def test_batch_processing_mock(self):
        """Test batch processing with mocked TensorRT"""
        with patch('folder_paths.get_folder_paths', return_value=[str(self.test_dir)]):
            # Create mock engine and context
            mock_engine = MagicMock()
            mock_engine.get_tensor_name.side_effect = ["input", "output"]
            mock_context = MagicMock()
            mock_context.execute_v2.return_value = True
            
            # Patch the loading and inference methods
            with patch.object(self.node, '_load_engine', return_value=mock_engine):
                with patch.object(self.node, '_infer_tile') as mock_infer:
                    # Mock inference to return upscaled tile
                    def mock_infer_func(tile_input, engine, context, scale):
                        b, c, h, w = tile_input.shape
                        return np.random.rand(h * scale, w * scale, c).astype(np.float32)
                    
                    mock_infer.side_effect = mock_infer_func
                    
                    # Create batch of test images
                    batch_size = 3
                    test_images = torch.rand(batch_size, 128, 128, 3)
                    
                    # Mock cuda operations
                    with patch('pycuda.driver.mem_alloc'), \
                         patch('pycuda.driver.memcpy_htod'), \
                         patch('pycuda.driver.memcpy_dtoh'):
                        
                        # Process batch
                        result = self.node.upscale(test_images, "test_model_2x", 
                                                 tile_size=64, tile_overlap=8, use_fp16=True)
                    
                    # Check result
                    self.assertIsInstance(result, tuple)
                    self.assertEqual(len(result), 1)
                    output = result[0]
                    
                    # Check output shape (2x upscale)
                    self.assertEqual(output.shape[0], batch_size)
                    self.assertEqual(output.shape[1], 256)  # 128 * 2
                    self.assertEqual(output.shape[2], 256)  # 128 * 2
                    self.assertEqual(output.shape[3], 3)
    
    def test_process_image_with_tiles_dimensions(self):
        """Test tile processing maintains correct dimensions"""
        # Create test data
        test_image = np.random.rand(100, 100, 3).astype(np.float32)
        scale = 2
        
        # Mock engine and context
        mock_engine = MagicMock()
        mock_context = MagicMock()
        
        # Mock the inference to return properly scaled output
        with patch.object(self.node, '_infer_tile') as mock_infer:
            def mock_infer_func(tile_input, engine, context, scale):
                b, c, h, w = tile_input.shape
                return np.random.rand(h * scale, w * scale, c).astype(np.float32)
            
            mock_infer.side_effect = mock_infer_func
            
            # Process with tiles
            output = self.node._process_image_with_tiles(
                test_image, mock_engine, mock_context, scale, 
                tile_size=50, overlap=10
            )
            
            # Check output dimensions
            self.assertEqual(output.shape, (200, 200, 3))
            self.assertTrue(np.all(output >= 0))
            self.assertTrue(np.all(output <= 1))


class TestIntegration(unittest.TestCase):
    """Integration tests with sample images"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = Path(__file__).parent / "test_models"
        self.test_dir.mkdir(exist_ok=True)
        self.output_dir = Path(__file__).parent / "test_output"
        self.output_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test files"""
        # Clean up output files
        if self.output_dir.exists():
            for file in self.output_dir.glob("*.png"):
                file.unlink()
    
    def test_node_workflow_simulation(self):
        """Test node in a simulated ComfyUI workflow"""
        with patch('folder_paths.get_folder_paths', return_value=[str(self.test_dir)]):
            # Create dummy engine file
            (self.test_dir / "test_4x_fp16.trt").write_bytes(b"dummy")
            
            # Get input types
            input_types = TRTUpscaler.INPUT_TYPES()
            
            # Verify we can get model list
            model_list = input_types["required"]["model_name"][0]
            self.assertIsInstance(model_list, list)
            self.assertGreater(len(model_list), 0)
            
            # Verify parameter ranges
            tile_size_config = input_types["required"]["tile_size"][1]
            self.assertEqual(tile_size_config["min"], 64)
            self.assertEqual(tile_size_config["max"], 2048)
            
            tile_overlap_config = input_types["required"]["tile_overlap"][1]
            self.assertEqual(tile_overlap_config["min"], 0)
            self.assertEqual(tile_overlap_config["max"], 256)


if __name__ == "__main__":
    unittest.main()