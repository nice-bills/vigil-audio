import torch
import librosa
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path

class AudioFeatureExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", cache_dir="models/hub"):
        """
        Initializes the Wav2Vec2 extractor with local caching.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading model: {model_name}...")
        print(f"Cache directory: {self.cache_dir.absolute()}")
        
        # Load processor and model with explicit cache_dir
        self.processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.model = Wav2Vec2Model.from_pretrained(model_name, cache_dir=self.cache_dir)
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")

    def extract(self, audio_path):
        """
        Extracts a single 768-dim embedding for an audio file.
        """
        try:
            # 1. Load audio (Wav2Vec2 expects 16kHz)
            speech, sr = librosa.load(audio_path, sr=16000)
            
            # 2. Preprocess
            inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(self.device)
            
            # 3. Inference (No gradient needed)
            with torch.no_grad():
                outputs = self.model(input_values)
                # We use the last_hidden_state and mean-pool over the time dimension
                # Resulting shape: (1, 768)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None

if __name__ == "__main__":
    # Test on a single file
    import pandas as pd
    
    metadata_path = "data/processed/metadata.csv"
    if Path(metadata_path).exists():
        df = pd.read_csv(metadata_path)
        sample_path = df.iloc[0]['path']
        
        extractor = AudioFeatureExtractor()
        embedding = extractor.extract(sample_path)
        
        if embedding is not None:
            print(f"\nSuccess!")
            print(f"File: {sample_path}")
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 5 values: {embedding[:5]}")
    else:
        print("Metadata not found. Please run harmonization first.")
