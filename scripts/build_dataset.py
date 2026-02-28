import os
import pandas as pd
from datasets import load_dataset

# ─── Configuration ────────────────────────────────────────────────────────────

OUTPUT_FILE = "merged_text.csv"
ORIGINAL_CSV = "text.csv"

# ─── Mapping GoEmotions ───────────────────────────────────────────────────────
# Map all 28 GoEmotions string labels to 7 core integer IDs [0-6].
# 0: Sadness, 1: Joy, 2: Love, 3: Anger, 4: Fear, 5: Surprise, 6: Neutral
GOEMOTIONS_MAP = {
    # Sadness cluster (0)
    "sadness": 0, "disappointment": 0, "embarrassment": 0, "grief": 0, "remorse": 0,
    # Joy cluster (1)
    "joy": 1, "amusement": 1, "excitement": 1, "gratitude": 1, "optimism": 1, "relief": 1, "pride": 1, "admiration": 1,
    # Love cluster (2)
    "love": 2, "caring": 2, "desire": 2,
    # Anger cluster (3)
    "anger": 3, "annoyance": 3, "disapproval": 3, "disgust": 3,
    # Fear cluster (4)
    "fear": 4, "nervousness": 4,
    # Surprise cluster (5)
    "surprise": 5, "realization": 5, "confusion": 5, "curiosity": 5,
    # Neutral cluster (6)
    "neutral": 6, "approval": 6
}

def load_and_map_goemotions():
    print("Fetching GoEmotions...")
    dataset = load_dataset("go_emotions", "simplified", split="train")
    
    # We need the feature names to map the integer IDs to strings first
    label_names = dataset.features["labels"].feature.names
    
    rows = []
    for item in dataset:
        labels = item["labels"]
        # GoEmotions is multi-label. To keep data clean, we only take single-label unambiguous examples.
        if len(labels) == 1:
            label_idx = labels[0]
            label_str = label_names[label_idx]
            
            if label_str in GOEMOTIONS_MAP:
                rows.append({
                    "text": item["text"].strip(),
                    "label": GOEMOTIONS_MAP[label_str]
                })
    
    df = pd.DataFrame(rows)
    print(f"  -> Extracted {len(df)} mapped single-label samples from GoEmotions.")
    return df

def load_dair_ai_emotion():
    print("Fetching dair-ai/emotion...")
    # This dataset's labels 0-5 perfectly match our 0-5 core labels.
    dataset = load_dataset("dair-ai/emotion", split="train", trust_remote_code=True)
    
    rows = []
    for item in dataset:
        rows.append({
            "text": item["text"].strip(),
            "label": item["label"]
        })
        
    df = pd.DataFrame(rows)
    print(f"  -> Extracted {len(df)} samples from dair-ai/emotion.")
    return df

def main():
    print("=== Emolyzer Dataset Expander (28 Classes) ===")
    
    dfs = []
    
    # 1. Base Twitter Dataset
    if os.path.exists(ORIGINAL_CSV):
        print(f"Loading base dataset: {ORIGINAL_CSV}")
        df_base = pd.read_csv(ORIGINAL_CSV)
        
        # Ensure correct columns and rename if needed
        if 'Unnamed: 0' in df_base.columns:
            df_base = df_base.drop(columns=['Unnamed: 0'])
            
        dfs.append(df_base)
        print(f"  -> Loaded {len(df_base)} samples from Twitter.")
    else:
        print(f"Warning: {ORIGINAL_CSV} not found! Building only from HF datasets.")
        
    # 2. dair-ai/emotion
    try:
        df_emotion = load_dair_ai_emotion()
        dfs.append(df_emotion)
    except Exception as e:
        print(f"Failed to load dair-ai/emotion: {e}")
        
    # 3. GoEmotions
    df_go = load_and_map_goemotions()
    dfs.append(df_go)
    
    # 4. Merge and Serialize
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Drop completely empty texts or NaN labels
    df_final = df_final.dropna(subset=['text', 'label'])
    df_final = df_final[df_final['text'].str.strip() != ""]
    
    # Enforce integer types for validation
    df_final['label'] = df_final['label'].astype(int)
    
    print("\n--- Final Dataset Distribution ---")
    print(df_final['label'].value_counts().sort_index())
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccessfully wrote {len(df_final)} rows to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
