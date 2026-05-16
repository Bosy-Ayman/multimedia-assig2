import os
import json
import pandas as pd
from pathlib import Path

def ingest_mimic_dataset(dataset_dir):
    dataset_path = Path(dataset_dir)
    kb_path = Path("data/knowledge_base.json")
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        print("Please download the dataset from Kaggle and extract it to this folder.")
        return

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.dicom', '*.dcm']:
        image_files.extend(dataset_path.rglob(ext))
    
    if not image_files:
        print(f"No images found in the directory {dataset_dir}.")
        return
        
    print(f"Found {len(image_files)} images.")
    
    # Try to find a CSV file with metadata/reports
    csv_files = list(dataset_path.rglob("*.csv"))
    df = None
    if csv_files:
        print(f"Found metadata file: {csv_files[0]}")
        try:
            df = pd.read_csv(csv_files[0])
            print("Columns available:", df.columns.tolist())
        except Exception as e:
            print(f"Could not read CSV: {e}")
            
    records = []
    
    # For the assignment demo, we'll limit to a subset of images so indexing doesn't take hours
    max_records = 50 
    print(f"Processing up to {max_records} records for the Knowledge Base...")
    
    for img_path in image_files[:max_records]:
        img_id = img_path.stem
        
        report_text = "Standard chest X-ray view. Normal findings."
        impression_text = "No acute cardiopulmonary abnormality."
        
        # Try to match with CSV if available
        if df is not None:
            # Look for rows where any column contains the image_id
            match = df[df.apply(lambda row: row.astype(str).str.contains(img_id).any(), axis=1)]
            if not match.empty:
                row = match.iloc[0]
                # Try to find report-like columns
                for col in df.columns:
                    col_lower = col.lower()
                    if 'report' in col_lower or 'text' in col_lower or 'findings' in col_lower:
                        report_text = str(row[col])
                    if 'impression' in col_lower:
                        impression_text = str(row[col])
        
        record = {
            "image_id": img_id,
            "image_path": str(img_path.absolute()),
            "impression": impression_text,
            "report": report_text,
            "embeddings": {}
        }
        records.append(record)
        
    # Save to knowledge base
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge with existing KB if it exists
    if kb_path.exists():
        with open(kb_path, 'r') as f:
            try:
                existing_records = json.load(f)
                # Deduplication by image_id
                existing_ids = {r["image_id"] for r in existing_records}
                for r in records:
                    if r["image_id"] not in existing_ids:
                        existing_records.append(r)
                records = existing_records
            except json.JSONDecodeError:
                pass
                
    with open(kb_path, 'w') as f:
        json.dump(records, f, indent=2)
        
    print(f"Successfully added MIMIC records to {kb_path}.")
    print(f"Total records in Knowledge Base: {len(records)}")
    print("\nNext step: Run 'python assig.py', go to the 'Knowledge Base' tab, and click 'Index All Records'!")

if __name__ == "__main__":
    import sys
    # Use command line arg if provided, otherwise assume 'mimic-cxr-dataset' folder
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "mimic-cxr-dataset"
    ingest_mimic_dataset(dataset_dir)
