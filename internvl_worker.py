import os
import sys

# ENABLE PROGRESS FOR USER
os.environ["TQDM_DISABLE"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

if not os.environ.get("HF_TOKEN"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

import torch
import json
import gc
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import threading
import psutil
import time

def monitor_ram():
    while True:
        mem = psutil.virtual_memory()
        print(f"AI Status: RAM Usage: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)", file=sys.stderr)
        time.sleep(10)

def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_model(model_name):
    # Start RAM monitor in background
    monitor_thread = threading.Thread(target=monitor_ram, daemon=True)
    monitor_thread.start()
    
    print(f"AI Status: Stage 1/2 - Initializing {model_name}...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    print("AI Status: Stage 1/2 - Weights loaded on CPU.", file=sys.stderr)
    return model, tokenizer

def run_worker():
    line = sys.stdin.readline()
    if not line: return
    
    try:
        request = json.loads(line)
        model_name = request.get("model_name", "OpenGVLab/InternVL2-1B")
        image_path = request.get("image_path")
        prompt_text = request.get("prompt", "Describe this chest X-ray in detail for a clinical report.")
        
        # Clean up the prompt if it's coming from the MedGemma format
        if prompt_text.startswith("caption en\n"):
            prompt_text = prompt_text.replace("caption en\n", "")
            
        model, tokenizer = load_model(model_name)
        
        image = Image.open(image_path).convert('RGB')
        transform = build_transform(input_size=448)
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16)
        
        print("AI Status: Analyzing image and thinking (Stage 2/2)...", file=sys.stderr)
        
        generation_config = dict(max_new_tokens=256, do_sample=True, temperature=0.4)
        
        question = f'<image>\n{prompt_text}'
        
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            
        print(json.dumps({"status": "ok", "report": response}))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
    finally:
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        gc.collect()

if __name__ == "__main__":
    run_worker()
