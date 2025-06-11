#!/usr/bin/env python3
"""
Model conversion utilities for TensorRT Upscaler
Converts PyTorch .pth models to TensorRT engines
"""

import os
import sys
import torch
import onnx
import subprocess
from pathlib import Path
import logging

# Try to import spandrel
try:
    from spandrel import ModelLoader, ImageModelDescriptor
    try:
        from spandrel_extra_arches import EXTRA_REGISTRY
        from spandrel import MAIN_REGISTRY
        # Safely add EXTRA_REGISTRY (ignore duplicate errors)
        try:
            MAIN_REGISTRY.add(*EXTRA_REGISTRY)
        except Exception:
            pass
    except ImportError:
        pass
except ImportError as e:
    print(f"Warning: Could not import spandrel: {e}")
    print("Please install spandrel: pip install spandrel")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def export_to_onnx(pth_path: str, onnx_path: str):
    """Export PyTorch model to ONNX format"""
    logger.info(f"Exporting {pth_path} to ONNX...")
    
    # Load state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(pth_path, map_location="cpu")
    
    # Remove module prefix if present
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[7:]] = v
            else:
                new_sd[k] = v
        sd = new_sd
    
    # Load model using spandrel
    model = ModelLoader().load_from_state_dict(sd)
    model = model.model
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )
    
    logger.info(f"ONNX export completed: {onnx_path}")


def convert_onnx_to_trt(onnx_path: str, engine_path: str, use_fp16: bool = True):
    """Convert ONNX model to TensorRT engine"""
    logger.info(f"Converting ONNX to TensorRT engine (FP{'16' if use_fp16 else '32'})...")
    
    # Build trtexec command
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--minShapes=input:1x3x64x64",
        "--optShapes=input:1x3x256x256",
        "--maxShapes=input:1x3x512x512",
        "--memPoolSize=workspace:1024M",
        "--builderOptimizationLevel=3",
        "--tacticSources=-CUDNN,+CUBLAS",
    ]
    
    if use_fp16:
        cmd.append("--fp16")
    
    # Run conversion
    try:
        logger.info("Running trtexec...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("TensorRT conversion completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.warning("Initial conversion failed, trying with reduced settings...")
        
        # Fallback with more conservative settings
        cmd_fallback = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--minShapes=input:1x3x64x64",
            "--optShapes=input:1x3x128x128",
            "--maxShapes=input:1x3x256x256",
            "--memPoolSize=workspace:512M",
            "--tacticSources=+CUBLAS",
        ]
        
        if use_fp16:
            cmd_fallback.append("--fp16")
        
        try:
            result = subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)
            logger.info("TensorRT conversion completed with fallback settings")
        except subprocess.CalledProcessError as e:
            logger.error(f"TensorRT conversion failed: {e}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise


def convert_pth_to_trt(model_name: str, models_dir: str, use_fp16: bool = True):
    """Convert a .pth model to TensorRT engine"""
    models_dir = Path(models_dir)
    
    # File paths
    pth_path = models_dir / f"{model_name}.pth"
    onnx_path = models_dir / f"{model_name}.onnx"
    precision_suffix = "_fp16" if use_fp16 else "_fp32"
    engine_path = models_dir / f"{model_name}{precision_suffix}.trt"
    
    # Check if .pth exists
    if not pth_path.exists():
        raise FileNotFoundError(f"Model file not found: {pth_path}")
    
    # Export to ONNX if needed
    if not onnx_path.exists():
        export_to_onnx(str(pth_path), str(onnx_path))
    else:
        logger.info(f"ONNX file already exists: {onnx_path}")
    
    # Convert to TensorRT
    if engine_path.exists():
        logger.info(f"TensorRT engine already exists: {engine_path}")
        overwrite = input("Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            return
    
    convert_onnx_to_trt(str(onnx_path), str(engine_path), use_fp16)
    logger.info(f"Conversion complete: {engine_path}")


def main():
    """Command line interface for model conversion"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert upscaling models to TensorRT")
    parser.add_argument("model_name", help="Model name (without extension)")
    parser.add_argument("--models-dir", default="../../models/upscale_models",
                        help="Models directory path")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 precision instead of FP16")
    
    args = parser.parse_args()
    
    # Convert model
    convert_pth_to_trt(args.model_name, args.models_dir, use_fp16=not args.fp32)


if __name__ == "__main__":
    main()