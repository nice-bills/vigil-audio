import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from extractor import AudioFeatureExtractor
import os

def build_all_features(metadata_path, output_dir):
    # 1. Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(metadata_path)
    extractor = AudioFeatureExtractor()
    
    print(f"Starting bulk extraction for {len(df)} files...")
    
    # 2. Loop with progress bar
    # We use a custom naming scheme: {split}_{original_filename}.npy
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        # Create a unique filename for the embedding
        # We replace .wav with .npy
        rel_path = Path(row['path']).stem
        embedding_path = output_dir / f"{row['split']}_{row['emotion']}_{rel_path}.npy"
        
        # 3. Skip if already exists (Resume capability)
        if embedding_path.exists():
            continue
            
        # 4. Extract and Save
        embedding = extractor.extract(row['path'])
        if embedding is not None:
            np.save(embedding_path, embedding)

    print(f"\nBulk Extraction Complete!")
    print(f"Embeddings saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    METADATA = "data/processed/metadata.csv"
    OUTPUT = "data/embeddings/wav2vec2"
    
    if os.path.exists(METADATA):
        build_all_features(METADATA, OUTPUT)
    else:
        print("Metadata not found. Run harmonize.py first.")
