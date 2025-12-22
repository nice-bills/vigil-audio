import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import librosa

def harmonize_data(raw_data_path, output_path):
    print(f"Scanning directory: {raw_data_path}")
    
    data = []
    # Folder names are our labels
    emotion_folders = [f for f in os.listdir(raw_data_path) if os.path.isdir(os.path.join(raw_data_path, f))]
    
    # Map folder names to standard labels
    # Note: 'Suprised' is misspelled in the source, we'll keep it for mapping but label it 'surprised'
    label_map = {folder: folder.lower() for folder in emotion_folders}
    
    for folder in emotion_folders:
        folder_path = Path(raw_data_path) / folder
        files = list(folder_path.glob("*.wav"))
        
        print(f"Processing {folder}: {len(files)} files")
        
        for file_path in tqdm(files, desc=f"Processing {folder}"):
            try:
                # Basic validation: can librosa load it?
                # We don't load the whole file here to save time, just check existence
                if file_path.exists():
                    data.append({
                        "filename": file_path.name,
                        "emotion": label_map[folder],
                        "path": str(file_path.absolute())
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data found! Please check the raw_data_path.")
        return

    # --- Stratified Splitting (80/10/10) ---
    print("\nCreating stratified splits...")
    
    # First split: Train vs Temp (20%)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df['emotion'], random_state=42
    )
    
    # Second split: Val (10%) vs Test (10%) from the Temp (20%)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['emotion'], random_state=42
    )
    
    # Mark splits
    train_df = train_df.assign(split='train')
    val_df = val_df.assign(split='val')
    test_df = test_df.assign(split='test')
    
    # Combine back
    final_df = pd.concat([train_df, val_df, test_df])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"\nHarmonization Complete!")
    print(f"Total files: {len(final_df)}")
    print(f"Metadata saved to: {output_path}")
    print("\nSplit Statistics:")
    print(final_df.groupby(['split', 'emotion']).size().unstack(fill_value=0))

if __name__ == "__main__":
    RAW_PATH = r"C:\dev\archive\Emotions"
    OUTPUT_PATH = "data/processed/metadata.csv"
    harmonize_data(RAW_PATH, OUTPUT_PATH)
