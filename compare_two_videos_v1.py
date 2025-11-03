# Simplified version of your mitochondrial video comparison pipeline

import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Literal
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM

# ---------------------------- Load 5D Volume ----------------------------
def load_5d_npy(path: str) -> np.ndarray:
    vol = np.load(path)
    assert vol.ndim == 5, f"Expected shape (T,C,Z,X,Y), got {vol.shape}"
    return vol.astype(np.float32)

# ---------------------------- Sample Time ----------------------------
def sample_time(vol: np.ndarray, max_frames: int = 100) -> np.ndarray:
    T = vol.shape[0]
    idx = np.linspace(0, T - 1, num=min(max_frames, T)).astype(int)
    return vol[idx]

# ---------------------------- Z Collapse ----------------------------
def z_collapse(vol: np.ndarray, mode: str = "max") -> np.ndarray:
    if mode == "max": return vol.max(axis=2)
    if mode == "mean": return vol.mean(axis=2)
    if mode == "mid": return vol[:, :, vol.shape[2] // 2]
    raise ValueError(f"Invalid collapse mode: {mode}")

# ---------------------------- Normalize and Convert ----------------------------
def normalize_uint8(frames: np.ndarray) -> np.ndarray:
    normed = np.zeros_like(frames)
    for c in range(frames.shape[1]):
        ch = frames[:, c]
        mn, mx = ch.min(), ch.max()
        normed[:, c] = 255 * (ch - mn) / (mx - mn + 1e-5)
    return normed.astype(np.uint8)

def to_rgb(frames: np.ndarray) -> np.ndarray:
    C = frames.shape[1]
    # r, g, b = [frames[:, min(i, C - 1)] for i in range(3)]
    r,g,b = [frames[:, 1] for i in range(3)]
    return np.stack([r, g, b], axis=-1)  # (T, X, Y, 3)

# ---------------------------- Resize and Export ----------------------------
def resize_and_save(frames: np.ndarray, size: Tuple[int, int], path: str):
    frames_resized = np.stack([cv2.resize(f, size[::-1]) for f in frames])
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for f in frames_resized: out.write(f[..., ::-1])
    out.release()

# ---------------------------- Compare Videos Using Qwen ----------------------------
# class QwenVideoComparator:
#     def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda"):
#         self.processor = AutoProcessor.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name, 
#             torch_dtype=torch.float16, 
#             device_map=device
#         ).eval()

#     def describe(self, video_a: str, video_b: str) -> str:
#         def get_sample_frames(path, num_frames=5):
#             from PIL import Image
            
#             cap = cv2.VideoCapture(path)
#             frames = []
#             while cap.isOpened():
#                 ret, f = cap.read()
#                 if not ret: break
#                 rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
#                 # Convert to PIL Image
#                 frames.append(Image.fromarray(rgb))
#             cap.release()
            
#             # Sample representative frames
#             if len(frames) <= num_frames:
#                 return frames
#             idx = np.linspace(0, len(frames)-1, num_frames).astype(int)
#             return [frames[i] for i in idx]

#         # Get sample frames from both videos
#         frames_a = get_sample_frames(video_a)
#         frames_b = get_sample_frames(video_b)
        
#         # Combine all images
#         all_images = frames_a + frames_b
        
#         # Create text prompt
#         text = "Here are frames from two mitochondrial videos (first 5 frames from Video A, next 5 from Video B). Describe the differences in motion, intensity, and morphology."
        
#         # Use chat format for Qwen2.5-VL
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": img} for img in all_images
#                 ] + [
#                     {"type": "text", "text": text}
#                 ]
#             }
#         ]
        
#         # Apply chat template and process
#         text_processed = self.processor.apply_chat_template(
#             messages, 
#             tokenize=False, 
#             add_generation_prompt=True
#         )
        
#         images = [{"type": "image", "image": img} for img in all_images]
#         inputs = self.processor(
#             text=[text_processed], 
#             images=images, 
#             padding=True, 
#             return_tensors="pt"
#         ).to(self.model.device)
        
#         # Use the model's forward pass for generation
#         with torch.no_grad():
#             input_ids = inputs.input_ids
#             generated_ids = input_ids
#             max_new_tokens = 512
#             eos_token_id = self.processor.tokenizer.eos_token_id
            
#             for _ in range(max_new_tokens):
#                 # Forward pass
#                 outputs = self.model(
#                     input_ids=generated_ids,
#                     attention_mask=torch.ones_like(generated_ids),
#                     pixel_values=inputs.get("pixel_values"),
#                     image_grid_thw=inputs.get("image_grid_thw")
#                 )
                
#                 # Get next token
#                 next_token_logits = outputs.logits[:, -1, :]
#                 next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
#                 # Append new token
#                 generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
#                 # Check for end token
#                 if next_token_id.item() == eos_token_id:
#                     break
        
#         # Extract and decode only the generated tokens
#         generated_only = generated_ids[0][input_ids.shape[1]:]
#         generated_text = self.processor.tokenizer.decode(generated_only, skip_special_tokens=True)
        
#         return generated_text


from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

class QwenVideoComparator:
    def __init__(self, model_name="Qwen/Qwen3-VL-4B-Instruct", device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            # force_download=True
        ).eval()

    def describe(self, video_a: str, video_b: str) -> str:
        
        def get_sample_frames(path, num_frames=5):
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened():
                ret, f = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cap.release()

            if len(frames) <= num_frames:
                return frames
            idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
            return [frames[i] for i in idx]

        frames_a = get_sample_frames(video_a)
        frames_b = get_sample_frames(video_b)
        all_images = frames_a + frames_b
        # print(len(all_images), np.array(frames_a[0]).min(), np.array(frames_a[0]).max())
        text = (
            "Compare these two mitochondrial videos (A first, B second). "
            "Describe differences in morphology, motion, and intensity over time."
            # "Link the differences across these factors to mechanistic pathways in the cell."
        )

        messages = [
            {
                "role": "user",
                # "content": [{"type": "video", "video": video_a} ] + [{"type": "video", "video": video_b} ] + [{"type": "text", "text": text}],
                "content": [{"type": "image", "image": img} for img in all_images] + [{"type": "text", "text": text}],
            }
        ]

        text_in = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_in],
            # images=[img for img in all_images],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return output
# ---------------------------- Main Pipeline ----------------------------
if __name__ == "__main__":
    path_a = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240815/000004.npy"
    path_b = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000001.npy"

    vol_a = sample_time(load_5d_npy(path_a))
    vol_b = sample_time(load_5d_npy(path_b))

    a_rgb = to_rgb(normalize_uint8(z_collapse(vol_a)))
    b_rgb = to_rgb(normalize_uint8(z_collapse(vol_b)))

    size = tuple(min(s1, s2) for s1, s2 in zip(a_rgb.shape[1:3], b_rgb.shape[1:3]))

    resize_and_save(a_rgb, size, "video_cccp_a.mp4")
    resize_and_save(b_rgb, size, "video_cccp_b.mp4")

    comparator = QwenVideoComparator()
    report = comparator.describe("video_cccp_a.mp4", "video_cccp_b.mp4")
    print("\n=== DIFFERENCE REPORT ===\n", report)
