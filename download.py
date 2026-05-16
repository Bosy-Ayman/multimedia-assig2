import kagglehub
import os
import sys

def main():
    print("Connecting to Kaggle...")
    try:
        # Download the smaller, high-quality Pneumonia dataset (~1.1 GB)
        path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"\n✅ Download complete!")
        print(f"Dataset location: {path}")
        
        # Trigger ingestion
        print("\nStarting ingestion into Knowledge Base...")
        os.system(f'python ingest_mimic.py "{path}"')
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure you have an internet connection and 'kagglehub' is installed (pip install kagglehub).")

if __name__ == "__main__":
    main()
