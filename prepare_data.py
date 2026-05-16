import os
import json
from pathlib import Path
from PIL import Image, ImageDraw

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SAMPLES = [
    {
        "image_id": "CXR_001_NORMAL",
        "impression": "No acute cardiopulmonary process.",
        "report": "Technical Quality: Adequate.\nFindings:\nLungs: Bilateral lung fields are clear. No focal consolidation, pleural effusion, or pneumothorax.\nHeart: The cardiomediastinal silhouette is within normal limits.\nBones: No acute osseous abnormalities.\nImpression: Normal chest X-ray. No acute cardiopulmonary findings."
    },
    {
        "image_id": "CXR_002_PNEUMONIA",
        "impression": "Right lower lobe consolidation concerning for pneumonia.",
        "report": "Technical Quality: Adequate.\nFindings:\nLungs: There is a focal area of consolidation in the right lower lobe. The left lung is clear. No pleural effusion or pneumothorax.\nHeart: The heart size is normal.\nBones: Intact.\nImpression: Right lower lobe airspace opacity. Given the clinical presentation, this is highly suggestive of pneumonia. Recommend clinical correlation and follow-up."
    }
]

def create_dummy_image(path, label):
    img = Image.new('RGB', (512, 512), color = (73, 109, 137) if "NORMAL" in label else (137, 73, 109))
    d = ImageDraw.Draw(img)
    d.text((10,10), label, fill=(255,255,0))
    img.save(path)

def prepare_data():
    kb_records = []
    
    for sample in SAMPLES:
        img_filename = f"{sample['image_id']}.png"
        img_path = DATA_DIR / img_filename
        
        print(f"Creating {sample['image_id']}...")
        try:
            create_dummy_image(img_path, sample['image_id'])
            
            qas = []
            if "NORMAL" in sample["image_id"]:
                qas = [
                    {"question": "Are there any opacities in the lungs?", "answer": "No, the bilateral lung fields are clear without any focal consolidation."},
                    {"question": "Is the heart size normal?", "answer": "Yes, the cardiomediastinal silhouette is within normal limits."},
                    {"question": "What is the overall impression?", "answer": "The impression is a normal chest X-ray with no acute cardiopulmonary process."}
                ]
            else:
                qas = [
                    {"question": "Are there any opacities in the lungs?", "answer": "Yes, there is a focal area of consolidation in the right lower lobe."},
                    {"question": "Is there evidence of pneumonia?", "answer": "Yes, the right lower lobe airspace opacity is highly suggestive of pneumonia."},
                    {"question": "What is the recommended follow-up?", "answer": "Clinical correlation and follow-up are recommended."}
                ]

            record = {
                "image_id": sample["image_id"],
                "image_path": str(img_path),
                "impression": sample["impression"],
                "report": sample["report"],
                "qa_pairs": qas,
                "embeddings": {}
            }
            kb_records.append(record)
            print(f"Successfully processed {sample['image_id']}.")
        except Exception as e:
            print(f"Failed to process {sample['image_id']}: {e}")
            
    kb_path = DATA_DIR / "knowledge_base.json"
    with open(kb_path, 'w') as f:
        json.dump(kb_records, f, indent=2)
    print(f"Knowledge base saved to {kb_path} with {len(kb_records)} records.")

if __name__ == "__main__":
    prepare_data()
