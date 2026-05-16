"""
ColPali subprocess worker — uses transformers native ColPali for better
memory-mapped loading. Runs in a separate process.

Memory-optimised: uses disk offloading + aggressive GC to avoid
Windows paging-file errors (OS error 1455).
"""
import os
import sys

# MASTER SILENCE: Disable all progress bars to prevent subprocess deadlocks
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# The token should be inherited from the parent process or loaded from environment
if not os.environ.get("HF_TOKEN"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

import json
import gc
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Limit PyTorch CPU threads
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_grad_enabled(False)

OFFLOAD_DIR = str(Path(__file__).parent / "offload" / "colpali")

def mean_pool(embeddings):
    if hasattr(embeddings, 'embeddings'):
        embeddings = embeddings.embeddings
    if isinstance(embeddings, torch.Tensor):
        if embeddings.dim() == 3:
            return embeddings.mean(dim=1).cpu().float().numpy()
        return embeddings.cpu().float().numpy()
    return np.array(embeddings)

def extract_embeddings(outputs):
    """Robustly extract embeddings from model outputs."""
    if hasattr(outputs, "embeddings"):
        return outputs.embeddings
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs

def main():
    # FORCE VIRTUAL ENVIRONMENT PRIORITY for the worker subprocess
    import sys
    import os
    _project_root = os.path.dirname(os.path.abspath(__file__))
    _venv_site = os.path.join(_project_root, "venv", "Lib", "site-packages")
    if os.path.exists(_venv_site) and _venv_site not in sys.path:
        sys.path.insert(0, _venv_site)

    request = json.loads(sys.stdin.read())
    action = request["action"]

    # Aggressive memory cleanup before anything heavy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Prefer colpali-engine for correct projection dimensions (1024 -> 256).
        # We use attn_implementation="eager" to avoid crashes with optimized kernels on CPU.
        try:
            from colpali_engine.models import ColPali as ModelClass, ColPaliProcessor as ProcClass
            print("Using colpali-engine ColPali", file=sys.stderr)
        except ImportError:
            from transformers import ColPaliForRetrieval as ModelClass, ColPaliProcessor as ProcClass
            print("Using native transformers ColPali (fallback)", file=sys.stderr)

        # Ensure offload directory exists and is clean
        offload_path = Path(OFFLOAD_DIR)
        if offload_path.exists():
            import shutil
            # Only clean if it's our specific offload dir
            if "offload" in str(offload_path):
                shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
        offload_path.mkdir(parents=True, exist_ok=True)

        print("Loading model (low-mem / disk offload)...", file=sys.stderr)
        gc.collect()
        
        # Use bfloat16 to save 50% RAM compared to float32
        # ~6GB instead of ~12GB for ColPali v1.2
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        model = ModelClass.from_pretrained(
            "vidore/colpali-v1.2",
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True
        )
        model.eval()
        # Free any cached weight buffers
        gc.collect()

        processor = ProcClass.from_pretrained(
            "vidore/colpali-v1.2",
            token=os.environ["HF_TOKEN"],
        )
        model.eval()
        print("Model loaded successfully!", file=sys.stderr)

        # Safety check: Ensure embeddings can handle the image token (often 257152)
        # Some PaliGemma/ColPali checkpoints have a vocab_size of 257152 but the 
        # processor uses 257152 as an index, which is out of bounds (0-indexed).
        if hasattr(model, "get_input_embeddings"):
            embed_weight = model.get_input_embeddings().weight
            current_size = embed_weight.shape[0]
            # 257152 is the common image token ID for PaliGemma/ColPali
            required_size = max(getattr(model.config, "vocab_size", 0), 257153)
            # Resizing embeddings in 4-bit can be unstable; disabled for GPU safety
            # if current_size < required_size:
            #     print(f"Resizing embeddings from {current_size} to {required_size} to prevent IndexError", file=sys.stderr)
            #     model.resize_token_embeddings(required_size)
        
        print(f"Final vocab size: {model.get_input_embeddings().weight.shape[0]}", file=sys.stderr)

        if action == "embed_image":
            image = Image.open(request["image_path"]).convert("RGB")
            inputs = processor.process_images([image]).to(model.device)
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
            
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask")
                )
            emb = extract_embeddings(outputs)
            result = mean_pool(emb).tolist()
            del emb, inputs, outputs; gc.collect()
            print(json.dumps({"status": "ok", "embedding": result}))

        elif action == "embed_text":
            inputs = processor.process_queries([request["text"]]).to(model.device)
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].long()
                
            with torch.inference_mode():
                outputs = model(
                    input_ids=inputs.get("input_ids"),
                    pixel_values=inputs.get("pixel_values"),
                    attention_mask=inputs.get("attention_mask")
                )
            emb = extract_embeddings(outputs)
            result = mean_pool(emb).tolist()
            del emb, inputs, outputs; gc.collect()
            print(json.dumps({"status": "ok", "embedding": result}))

        elif action == "similarity":
            image = Image.open(request["image_path"]).convert("RGB")
            text = request["text"]

            img_inputs = processor.process_images([image]).to(model.device)
            if "pixel_values" in img_inputs:
                img_inputs["pixel_values"] = img_inputs["pixel_values"].to(model.dtype)
                
            with torch.inference_mode():
                img_outputs = model(
                    input_ids=img_inputs.get("input_ids"),
                    pixel_values=img_inputs.get("pixel_values"),
                    attention_mask=img_inputs.get("attention_mask")
                )
            img_emb = mean_pool(extract_embeddings(img_outputs))
            del img_inputs, img_outputs; 
            torch.cuda.empty_cache()
            gc.collect()

            txt_inputs = processor.process_queries([text]).to(model.device)
            if "input_ids" in txt_inputs:
                txt_inputs["input_ids"] = txt_inputs["input_ids"].long()
                
            with torch.inference_mode():
                txt_outputs = model(
                    input_ids=txt_inputs.get("input_ids"),
                    pixel_values=txt_inputs.get("pixel_values"),
                    attention_mask=txt_inputs.get("attention_mask")
                )
            txt_emb = mean_pool(extract_embeddings(txt_outputs))
            del txt_inputs, txt_outputs; 
            torch.cuda.empty_cache()
            gc.collect()

            sim = float((img_emb @ txt_emb.T)[0][0])
            print(json.dumps({"status": "ok", "similarity": sim}))
        elif action == "batch_embed_images":
            image_paths = request["image_paths"]
            embeddings = []
            total = len(image_paths)
            print(f"Batch processing {total} images...", file=sys.stderr)
            for i, path in enumerate(image_paths):
                try:
                    image = Image.open(path).convert("RGB")
                    inputs = processor.process_images([image]).to(model.device)
                    
                    # Force correct data types to prevent CUDA "Int" errors
                    if "input_ids" in inputs:
                        inputs["input_ids"] = inputs["input_ids"].long()
                    if "pixel_values" in inputs:
                        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

                    with torch.inference_mode():
                        outputs = model(
                            input_ids=inputs.get("input_ids"),
                            pixel_values=inputs.get("pixel_values"),
                            attention_mask=inputs.get("attention_mask")
                        )
                    emb = extract_embeddings(outputs)
                    embeddings.append({"path": path, "status": "ok", "embedding": mean_pool(emb).tolist()})
                    del emb, inputs, outputs; 
                    torch.cuda.empty_cache()
                    gc.collect()
                    if (i + 1) % 5 == 0 or i == total - 1:
                        print(f"  [Progress] {i + 1}/{total} images processed", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing {path}: {e}", file=sys.stderr)
                    embeddings.append({"path": path, "status": "error", "message": str(e)})
            print(json.dumps({"status": "ok", "embeddings": embeddings}))
        else:
            print(json.dumps({"status": "error", "message": f"Unknown action: {action}"}))

    except Exception as e:
        import traceback
        print(f"Worker error: {traceback.format_exc()}", file=sys.stderr)
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    main()

