"""
Standalone test: Can ColPali load on this machine?
Run outside of Streamlit to minimize RAM usage.
"""
import gc
import os
import torch

# Use environment token
if not os.environ.get("HF_TOKEN"):
    from dotenv import load_dotenv
    load_dotenv()

print(f"Python process starting...")
print(f"Available RAM: checking...")

import psutil
mem = psutil.virtual_memory()
print(f"Total RAM: {mem.total / 1024**3:.1f} GB")
print(f"Available RAM: {mem.available / 1024**3:.1f} GB")
print(f"Used RAM: {mem.used / 1024**3:.1f} GB")
print()

# Force garbage collection to free as much RAM as possible
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("=== Attempting ColPali load on CPU with float16 ===")
try:
    from colpali_engine.models import ColPali, ColPaliProcessor
    
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=os.environ["HF_TOKEN"]
    )
    processor = ColPaliProcessor.from_pretrained(
        "vidore/colpali-v1.2",
        token=os.environ["HF_TOKEN"]
    )
    
    print("\n=== SUCCESS! ColPali loaded! ===")
    mem2 = psutil.virtual_memory()
    print(f"RAM after load: {mem2.used / 1024**3:.1f} GB used / {mem2.available / 1024**3:.1f} GB available")
    
    # Quick test
    from PIL import Image
    import numpy as np
    dummy = Image.new("RGB", (224, 224), color="gray")
    inputs = processor.process_images([dummy]).to("cpu")
    with torch.no_grad():
        emb = model(**inputs)
    print(f"Embedding shape: {emb.shape}")
    print("ColPali is fully functional on this machine!")
    
except Exception as e:
    print(f"\n=== FAILED: {e} ===")
    mem2 = psutil.virtual_memory()
    print(f"RAM at failure: {mem2.used / 1024**3:.1f} GB used / {mem2.available / 1024**3:.1f} GB available")
