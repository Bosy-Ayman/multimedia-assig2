
import torch
import os
from PIL import Image
from pathlib import Path

# Use environment token
if not os.environ.get("HF_TOKEN"):
    from dotenv import load_dotenv
    load_dotenv()

try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    print("Loading colpali-engine model...")
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.float32,
        device_map="cpu",
        token=os.environ["HF_TOKEN"],
    )
    processor = ColPaliProcessor.from_pretrained(
        "vidore/colpali-v1.2",
        token=os.environ["HF_TOKEN"],
    )
    
    print(f"Model vocab size: {model.config.vocab_size}")
    if hasattr(model.config, "text_config"):
        print(f"Text config vocab size: {model.config.text_config.vocab_size}")
    
    # Check embedding layer
    embed_weight = model.get_input_embeddings().weight
    print(f"Embedding weight shape: {embed_weight.shape}")
    
    # Test processing
    img = Image.new('RGB', (224, 224), color='gray')
    inputs = processor.process_images([img])
    max_id = inputs['input_ids'].max().item()
    print(f"Max input ID from processor: {max_id}")
    
    if max_id >= embed_weight.shape[0]:
        print(f"CRITICAL: Max ID {max_id} is >= embedding size {embed_weight.shape[0]}")
    
    # Try forward pass
    print("Running forward pass...")
    with torch.no_grad():
        out = model(**inputs)
    print("Success!")

except Exception as e:
    import traceback
    traceback.print_exc()
