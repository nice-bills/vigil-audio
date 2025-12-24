import os
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from optimum.onnxruntime import ORTModelForAudioClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from optimum.onnxruntime import ORTQuantizer
import shutil

# --- CONFIG ---
MODEL_PATH = "models/wav2vec2-finetuned"
ONNX_PATH = "models/onnx"
QUANTIZED_PATH = "models/onnx_quantized"

def export_to_onnx():
    print(f"Exporting PyTorch model to ONNX...")
    
    # Load PyTorch model
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    
    # Export to ONNX using Optimum (handles complex configs automatically)
    model = ORTModelForAudioClassification.from_pretrained(MODEL_PATH, export=True)
    
    # Save ONNX model
    if os.path.exists(ONNX_PATH):
        shutil.rmtree(ONNX_PATH)
    model.save_pretrained(ONNX_PATH)
    feature_extractor.save_pretrained(ONNX_PATH)
    
    print(f"ONNX model saved to: {ONNX_PATH}")

def quantize_onnx():
    print(f"Quantizing ONNX model to INT8...")
    
    # Load ONNX model for quantization
    quantizer = ORTQuantizer.from_pretrained(ONNX_PATH, file_name="model.onnx")
    
    # Define quantization config (INT8 dynamic quantization)
    # Exclude Conv layers to prevent 'initializer' errors in Wav2Vec2
    qconfig = AutoQuantizationConfig.arm64(
        is_static=False, 
        per_channel=False,
        operators_to_quantize=["MatMul", "Attention", "LSTM", "Gather", "Transpose", "EmbedLayerNormalization"]
    )
    
    # Apply quantization
    if os.path.exists(QUANTIZED_PATH):
        shutil.rmtree(QUANTIZED_PATH)
        
    quantizer.quantize(save_dir=QUANTIZED_PATH, quantization_config=qconfig)
    
    # Copy feature extractor config to quantized folder so it's self-contained
    feature_extractor = AutoFeatureExtractor.from_pretrained(ONNX_PATH)
    feature_extractor.save_pretrained(QUANTIZED_PATH)
    
    print(f"INT8 Quantized model saved to: {QUANTIZED_PATH}")

if __name__ == "__main__":
    export_to_onnx()
    quantize_onnx()
