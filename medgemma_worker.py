import os
import sys

# MASTER SILENCE
# ENABLE PROGRESS FOR USER
os.environ["TQDM_DISABLE"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
# The token should be inherited from the parent process or loaded from environment
if not os.environ.get("HF_TOKEN"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

import torch
import json
import gc
import warnings
from PIL import Image

import psutil
import threading
import time

def monitor_ram():
    while True:
        mem = psutil.virtual_memory()
        print(f"AI Status: RAM Usage: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)", file=sys.stderr)
        time.sleep(10)

# Total Silence Mode
warnings.filterwarnings('ignore')
import transformers
# transformers.utils.logging.set_verbosity_info()

from transformers import AutoProcessor, AutoModelForImageTextToText

def load_model(model_name):
    # Start RAM monitor in background
    monitor_thread = threading.Thread(target=monitor_ram, daemon=True)
    monitor_thread.start()
    
    print("AI Status: Stage 1/2 - Initializing (Checking cache/downloading)...", file=sys.stderr)
    # Use bfloat16 for 4B version to save RAM (uses ~8GB instead of ~16GB)
    token = os.environ.get("HF_TOKEN")
    
    # We load the processor first as it's small and confirms the model ID/token is correct
    try:
        processor = AutoProcessor.from_pretrained(model_name, token=token)
        print("AI Status: Stage 1/2 - Processor ready. Loading weights (this may take 2-5 mins)...", file=sys.stderr)
    except Exception as e:
        print(f"AI Status: Error loading processor: {str(e)}", file=sys.stderr)
        raise e

    # Create offload directory
    offload_dir = os.path.join(os.getcwd(), "offload_medgemma")
    os.makedirs(offload_dir, exist_ok=True)

    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Use 4-bit quantization for GPU efficiency (mandatory for 6GB VRAM)
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    # Load model strictly on GPU to prevent bitsandbytes CPU offload crash
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        token=token
    )
    print("AI Status: Stage 1/2 - Weights loaded in 4-bit on GPU.", file=sys.stderr)
    print("AI Status: Stage 1/2 - Weights loaded. Ready for generation.", file=sys.stderr)
    return model, processor

def run_worker():
    line = sys.stdin.readline()
    if not line: return
        
    try:
        request = json.loads(line)
        model_name = request.get("model_name", "google/medgemma-1.5-4b-it")
        image_path = request.get("image_path")
        prompt_text = request.get("prompt", "Describe this chest X-ray in detail for a clinical report.")
        
        model, processor = load_model(model_name)
        image = Image.open(image_path).convert("RGB")
        
        print("AI Status: Analyzing image and thinking (Stage 2/2)...", file=sys.stderr)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(model.device)
        
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                do_sample=True,
                temperature=0.4
            )
            
        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        report = processor.decode(new_tokens, skip_special_tokens=True)
            
        print(json.dumps({"status": "ok", "report": report}))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))
    finally:
        if 'model' in locals(): del model
        if 'processor' in locals(): del processor
        gc.collect()

if __name__ == "__main__":
    run_worker()
