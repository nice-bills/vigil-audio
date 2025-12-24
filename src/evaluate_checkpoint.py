import os
import torch
import pandas as pd
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification, AutoConfig
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

def evaluate_model(model_path="models/wav2vec2-finetuned", metadata_path="data/processed/metadata.csv"):
    print(f"Evaluating model at: {model_path}")
    
    # 1. Load Config to get TRUE labels
    try:
        config = AutoConfig.from_pretrained(model_path)
        id2label = config.id2label
        label2id = config.label2id
        print(f"Loaded Label Map from Config: {label2id}")
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # 2. Load Model & Extractor
    device = torch.device("cpu")
    try:
        extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 3. Load Metadata
    if not os.path.exists(metadata_path):
        print(f"Metadata not found at {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    # Test on 100 samples
    test_df = df[df['split'] == 'test'].sample(min(100, len(df[df['split'] == 'test'])), random_state=42)
    
    print(f"Testing on {len(test_df)} samples...")

    y_true = []
    y_pred = []

    # 4. Inference Loop
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
        # Fix: Ensure we match the exact string expected by the model
        true_label_str = row['emotion'] 
        
        # Handle "Suprised" misspelling if model expects "surprised"
        if "surprised" in label2id and true_label_str == "suprised":
            true_label_str = "surprised"
            
        if true_label_str not in label2id:
            print(f"Warning: Label '{true_label_str}' not in model config. Skipping.")
            continue
            
        target_id = label2id[true_label_str]
        
        # Path handling
        audio_path = row['path']
        if not os.path.exists(audio_path):
            audio_path = os.path.join("C:/dev/archive/Emotions", row['emotion'].capitalize(), row['filename'])
            if not os.path.exists(audio_path):
                continue

        try:
            # Load and Preprocess
            speech, sr = librosa.load(audio_path, sr=16000)
            # Crop to 5s to match training logic
            if len(speech) > 16000 * 5:
                speech = speech[:16000 * 5]
                
            inputs = extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Predict
            with torch.no_grad():
                logits = model(inputs.input_values).logits
                pred_id = torch.argmax(logits, dim=-1).item()
            
            y_true.append(target_id)
            y_pred.append(pred_id)
            
        except Exception as e:
            continue

    # 5. Results
    if not y_true:
        print("No files processed.")
        return

    acc = accuracy_score(y_true, y_pred)
    print(f"\nFINAL ACCURACY: {acc:.2%}")
    
    # Map IDs back to names for the report
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, labels=sorted(id2label.keys())))
    
    # Print a small confusion matrix snippet
    print("\nConfusion Matrix (True vs Pred):")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()