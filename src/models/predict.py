import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import argparse
import os

class EmotionPredictor:
    def __init__(self, model_path="models/wav2vec2-finetuned"):
        """
        Initializes the predictor with the fine-tuned model.
        """
        print(f"Loading model from: {model_path}...")
        
        # Load config and model
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
            self.model.eval() # Set to inference mode
            
            # Extract labels from config
            self.id2label = self.model.config.id2label
            print(f"Model loaded. Labels: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise

    def predict(self, audio_path):
        """
        Predicts emotion from a single audio file.
        """
        # 1. Load Audio
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            return {"error": f"Could not load audio: {e}"}

        # 2. Preprocess
        inputs = self.feature_extractor(
            speech, 
            sampling_rate=16000, 
            padding=True, 
            return_tensors="pt"
        )

        # 3. Inference
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # 4. Post-processing
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(logits, dim=-1).item()
        confidence = probabilities[0][predicted_id].item()
        predicted_label = self.id2label[predicted_id]

        return {
            "emotion": predicted_label,
            "confidence": f"{confidence:.2%}",
            "probabilities": {
                self.id2label[i]: f"{probabilities[0][i].item():.2%}" 
                for i in range(len(self.id2label))
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from audio file")
    parser.add_argument("--file", type=str, help="Path to .wav file")
    args = parser.parse_args()

    if args.file:
        predictor = EmotionPredictor()
        result = predictor.predict(args.file)
        print("\nPrediction Result:")
        print(f"Emotion: {result.get('emotion', 'Error')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(result)
    else:
        print("Please provide a file path using --file")
