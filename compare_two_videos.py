import os
import numpy as np
import cv2
import torch
from typing import Tuple, List, Optional, Literal, Callable
# from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
# odel = QWenVLModel.from_pretrained(model_name)model = QWenVLModel.from_pretrained(model_name)

########################################
# 1. READ THE IMAGES
########################################

def load_5d_npy(path: str) -> np.ndarray:
    """
    Load a 5D array saved as .npy.

    Expected shape: (T, C, Z, X, Y)

    Returns
    -------
    vol : np.ndarray
        Float32 array with shape (T, C, Z, X, Y).
        We coerce to float32 for consistency downstream.
    """
    vol = np.load(path)
    if vol.ndim != 5:
        raise ValueError(f"Expected 5D array (T,C,Z,X,Y), got {vol.shape}")
    return vol.astype(np.float32, copy=False)


########################################
# 2. SAMPLE FRAMES FROM THOSE IMAGES
########################################

def sample_time(
    vol: np.ndarray,
    max_frames: Optional[int] = None,
    stride: Optional[int] = None
) -> np.ndarray:
    """
    Subsample along time.

    You can choose either:
    - max_frames: keep ~that many frames, spaced uniformly
    - stride: keep every `stride`-th frame

    If both are given, stride wins.

    Returns
    -------QWenVLModel
    vol_sub : np.ndarray
        Subsampled volume, still shape (T_sub, C, Z, X, Y)
    """
    T = vol.shape[0]

    if stride is not None:
        idx = np.arange(0, T, stride)
    elif max_frames is not None and max_frames < T:
        idx = np.linspace(0, T - 1, num=max_frames)
        idx = np.round(idx).astype(int)
        idx = np.unique(idx)
    else:
        idx = np.arange(T)

    return vol[idx]


########################################
# Helper: collapse 3D (Z,X,Y) -> 2D (X,Y)
########################################

def z_collapse(
    vol_tczxy: np.ndarray,
    mode: Literal["max", "mean", "mid"] = "max"
) -> np.ndarray:
    """
    Collapse Z dimension to get 2D frames for visualization / video export.

    vol_tczxy: (T, C, Z, X, Y)

    Returns
    -------
    vol_tcxy : (T, C, X, Y)
    """
    T, C, Z, X, Y = vol_tczxy.shape

    if mode == "max":
        collapsed = vol_tczxy.max(axis=2)         # (T,C,X,Y)
    elif mode == "mean":
        collapsed = vol_tczxy.mean(axis=2)        # (T,C,X,Y)
    elif mode == "mid":
        z_mid = Z // 2
        collapsed = vol_tczxy[:, :, z_mid, ...]   # (T,C,X,Y)
    else:
        raise ValueError(f"Unknown collapse mode {mode}")

    return collapsed


########################################
# 3. VIEW THE 4D IMAGES IN NAPARI
########################################

def view_in_napari(
    vol: np.ndarray,
    name: str = "volume",
    contrast_limits: Optional[Tuple[float, float]] = None,
    multiscale: bool = False
):
    """
    Launches a Napari viewer for interactive inspection.

    vol should be (T, C, Z, X, Y).

    We add it as a napari layer with dims ('T','C','Z','Y','X') order expected by napari for 5D.
    Napari expects the last 2 axes to be spatial (Y,X), so we swap (X,Y)->(Y,X).
from transformers import QWenVLModel
model = QWenVLModel.from_pretrained(model_name)
    NOTE: This will block / open GUI in a desktop session.
    """
    import napari

    # swap X,Y -> Y,X
    # (T,C,Z,X,Y) -> (T,C,Z,Y,X)
    vol_for_napari = np.swapaxes(vol, -1, -2)

    viewer = napari.Viewer()
    viewer.add_image(
        vol_for_napari,
        name=name,
        channel_axis=1,          # interpret axis 1 as channels
        contrast_limits=contrast_limits,
        multiscale=multiscale,
        blending='additive'
    )
    napari.run()


########################################
# 4. EXPORT MATCHED VIDEOS
########################################

def normalize_to_uint8(
    frames: np.ndarray,
    per_channel: bool = True
) -> np.ndarray:
    """
    Normalize float data to uint8 [0,255].

    frames: (T, C, X, Y)

    Returns (T, C, X, Y) uint8
    """
    arr = frames.copy()

    if per_channel:
        # scale each channel independently (min->0 max->255)
        for c in range(arr.shape[1]):
            ch = arr[:, c]
            mn = ch.min()
            mx = ch.max()
            if mx == mn:
                arr[:, c] = 0
            else:
                arr[:, c] = (ch - mn) / (mx - mn) * 255.0
    else:
        mn = arr.min()
        mx = arr.max()
        if mx == mn:
            arr[...] = 0
        else:
            arr = (arr - mn) / (mx - mn) * 255.0

    return arr.clip(0, 255).astype(np.uint8)


def stack_channels_for_rgb(
    frames_uint8: np.ndarray,
    channel_map: Optional[Tuple[int, int, int]] = None
) -> np.ndarray:
    """
    Convert multi-channel scientific data into RGB frames for video export.

    frames_uint8: (T, C, X, Y) uint8
    channel_map: which channels become R,G,B (defaults below)

    Returns
    -------
    rgb_frames : (T, X, Y, 3) uint8  (OpenCV uses BGR, we'll flip later)
    """
    T, C, X, Y = frames_uint8.shape

    if channel_map is None:
        # heuristics:
        # - if C>=3, pick first 3
        # - if C==2, map [0]->R, [1]->G, B=0
        # - if C==1, replicate
        if C >= 3:
            channel_map = (0, 1, 2)
        elif C == 2:
            channel_map = (0, 1, 1)  # fake blue using channel 1
        elif C == 1:
            channel_map = (0, 0, 0)
        else:
            raise ValueError("No channels?")

    r = frames_uint8[:, channel_map[0]]
    g = frames_uint8[:, channel_map[1]]
    b = frames_uint8[:, channel_map[2]]

    # stack as (T,X,Y,3) in RGB order
    rgb = np.stack([r, g, b], axis=-1)  # (T,X,Y,3)
    return rgb


def resize_frames(
    frames: np.ndarray,
    target_hw: Tuple[int, int]
) -> np.ndarray:
    """
    Resize frames to target height,width using OpenCV.

    frames: (T, H, W, 3) uint8
    target_hw: (target_h, target_w)

    Returns (T, target_h, target_w, 3)
    """
    T, H, W, _ = frames.shape
    tgt_h, tgt_w = target_hw
    out = []
    for t in range(T):
        out.append(cv2.resize(frames[t], (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0)


def write_mp4(
    frames_bgr: np.ndarray,
    out_path: str,
    fps: int = 10,
    codec: str = "mp4v"
):
    """
    Save frames to MP4 using OpenCV.

    frames_bgr: (T, H, W, 3) uint8 in BGR order
    """
    T, H, W, _ = frames_bgr.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    for i in range(T):
        writer.write(frames_bgr[i])
    writer.release()


def make_matched_videos(
    vol_a: np.ndarray,
    vol_b: np.ndarray,
    collapse_mode: Literal["max", "mean", "mid"] = "max",
    max_frames: Optional[int] = None,
    stride: Optional[int] = None,
    out_a: str = "video_a.mp4",
    out_b: str = "video_b.mp4",
    fps: int = 10,
    codec: str = "mp4v"
) -> Tuple[str, str]:
    """
    1. Time-sample both volumes with SAME policy
    2. Collapse Z to 2D
    3. Normalize to uint8
    4. Map channels -> RGB
    5. Resize so both videos share same spatial size
    6. Trim to same T
    7. Write MP4s with same fps

    Returns (out_a, out_b)
    """
    # --- step 1: sample time
    vol_a_s = sample_time(vol_a, max_frames=max_frames, stride=stride)
    vol_b_s = sample_time(vol_b, max_frames=max_frames, stride=stride)

    # --- step 2: collapse Z
    a_tcxy = z_collapse(vol_a_s, mode=collapse_mode)  # (Ta,C,X,Y)
    b_tcxy = z_collapse(vol_b_s, mode=collapse_mode)  # (Tb,C,X,Y)
    with torch.no_grad():
        
        

    output_tokens = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,
    )
    # --- step 3: normalize
    a_u8 = normalize_to_uint8(a_tcxy, per_channel=True)  # (Ta,C,X,Y)
    b_u8 = normalize_to_uint8(b_tcxy, per_channel=True)  # (Tb,C,X,Y)

    # --- step 4: channel map -> RGB frames
    a_rgb = stack_channels_for_rgb(a_u8)  # (Ta,X,Y,3) RGB
    b_rgb = stack_channels_for_rgb(b_u8)  # (Tb,X,Y,3) RGB

    # --- step 5: resize
    # choose target spatial size as min of both (so we downsample larger)
    Ha, Wa = a_rgb.shape[1:3]
    Hb, Wb = b_rgb.shape[1:3]
    tgt_h = min(Ha, Hb)
    tgt_w = min(Wa, Wb)

    a_rgb_rs = resize_frames(a_rgb, (tgt_h, tgt_w))
    b_rgb_rs = resize_frames(b_rgb, (tgt_h, tgt_w))

    # --- step 6: trim to same T
    Ta = a_rgb_rs.shape[0]
    Tb = b_rgb_rs.shape[0]
    Tmatch = min(Ta, Tb)
    a_rgb_rs = a_rgb_rs[:Tmatch]
    b_rgb_rs = b_rgb_rs[:Tmatch]

    # --- OpenCV expects BGR, so swap channels
    a_bgr = a_rgb_rs[..., ::-1]
    b_bgr = b_rgb_rs[..., ::-1]

    # --- step 7: write
    write_mp4(a_bgr, out_a, fps=fps, codec=codec)
    write_mp4(b_bgr, out_b, fps=fps, codec=codec)

    return out_a, out_b


########################################
# 5. INITIALISE Qwen3.0
########################################

class QwenVideoComparator:
    """
    Thin wrapper around a Qwen3.0 video-language model.

    This class:
    - loads model + tokenizer
    - builds a prompt comparing video A vs video B
    - runs generation
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-0.6B",   # keep configurable
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        load_fn: Optional[Callable] = None,
        tokenizer_fn: Optional[Callable] = None
    ):
        """
        model_name:
            HuggingFace-style model id (example placeholder).
            Adjust to the actual Qwen3.0 video model you have.

        device:
            "cuda", "cuda:0", "cpu", etc.

        dtype:
            torch.float16 or torch.bfloat16, etc.

        load_fn:
            Optional injector to override how the model is loaded (for custom checkpoints / API).
            Signature: load_fn(model_name, torch_dtype, device) -> model

        tokenizer_fn:
            Optional injector to override tokenizer init.
            Signature: tokenizer_fn(model_name) -> tokenizer
        """

        # You may need to install / import appropriate Qwen libs, e.g.:
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # or the Qwen-provided video model class.

        if load_fn is None or tokenizer_fn is None:
            # default HuggingFace-like path
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                # from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    "You need transformers installed and a Qwen3.0-compatible model class. "
                    "pip install transformers accelerate safetensors"
                ) from e

        # tokenizer
        if tokenizer_fn is None:
            tokenizer_fn = lambda name : AutoTokenizer.from_pretrained(name)
            # tokenizer_fn = lambda name: AutoTokenizer.from_pretrained(name)

        # model
        if load_fn is None:
            

            def _default_load(model_name, torch_dtype, device):
                # model = AutoModel.from_pretrained(
                #     name,
                #     torch_dtype=torch_dtype,
                #     device_map=device,
                # )
                # model.eval()

                
                
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                model.eval()

            load_fn = _default_load

        self.tokenizer = tokenizer_fn(model_name)
        self.model = load_fn(model_name, dtype, device)

        # NOTE:
        # For true video understanding, Qwen VL models usually expect
        # a multimodal forward pass with video frames (as tensors),
        # not just text tokens. We'll stub that next.


########################################
# Helper: decode video into frames for Qwen
########################################

def read_video_frames_for_model(
    video_path: str,
    max_frames: int = 16,
    frame_sample: Literal["uniform", "head", "tail"] = "uniform"
) -> List[np.ndarray]:
    """
    Extract a handful of frames from MP4 so we can feed them to the VLM.

    Returns list of RGB frames as np.uint8 (H,W,3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_sample == "uniform":
        indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames))
        indices = np.round(indices).astype(int)
        indices = np.unique(indices)
    elif frame_sample == "head":
        indices = np.arange(0, min(max_frames, total_frames))
    elif frame_sample == "tail":
        start = max(0, total_frames - max_frames)
        indices = np.arange(start, total_frames)
    else:
        raise ValueError("bad frame_sample")

    grabbed_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = frame_bgr[..., ::-1]  # BGR->RGB
        grabbed_frames.append(frame_rgb)

    cap.release()
    return grabbed_frames


########################################
# 6. GENERATE TEXT DESCRIPTION OF DIFFERENCES
########################################

def describe_differences_qwen(
    comparator: QwenVideoComparator,
    video_a_path: str,
    video_b_path: str,
    prompt_template: str = (
        "You are comparing two microscopy time-lapse videos.\n"
        "Video A shows {A_summary}.\n"
        "Video B shows {B_summary}.\n"
        "Describe key differences in motion, intensity, morphology, and any events over time.\n"
        "Be specific and concise."
    ),
    max_model_frames: int = 16
) -> str:
    """
    High-level pipeline:
    1. Sample frames from both videos.
    2. Prepare a multimodal prompt.
    3. Run Qwen3.0 to get natural language diff.

    NOTE:
    Actual Qwen3.0 video-language models generally want inputs like:
      model.generate( vision_inputs=[framesA, framesB], text_inputs=prompt )
    The exact API varies across releases.

    Since we can't guarantee exact Qwen API here, we:
    - Extract sparse RGB frames.
    - Create naive textual summaries (mean brightness, motion estimate).
    - Feed that text into the LLM as a stand-in.
    - This keeps the interface clean so you can later swap in the true multimodal call.
    """

    # 1. grab representative frames
    frames_a = read_video_frames_for_model(video_a_path, max_frames=max_model_frames)
    frames_b = read_video_frames_for_model(video_b_path, max_frames=max_model_frames)

    def quick_textual_summary(frames: List[np.ndarray]) -> str:
        """
        Cheap numeric descriptors we can hand to the LLM.
        You can later replace this with actual vision embedding.
        """
        if len(frames) == 0:
            return "no visible content"

        # mean brightness and rough motion
        means = []
        motion_scores = []
        prev = None
        for f in frames:
            means.append(float(f.mean()))
            if prev is not None:
                # L2 diff / pixel as motion proxy
                diff = (f.astype(np.float32) - prev.astype(np.float32)) ** 2
                motion_scores.append(float(np.sqrt(diff.mean())))
            prev = f
        mean_brightness = np.mean(means)
        motion_level = np.mean(motion_scores) if len(motion_scores) else 0.0

        H, W, _ = frames[0].shape
        return (
            f"~{len(frames)} sampled frames, size {H}x{W} px, "
            f"avg brightness {mean_brightness:.1f}, "
            f"avg inter-frame motion score {motion_level:.2f}."
        )

    A_summary = quick_textual_summary(frames_a)
    B_summary = quick_textual_summary(frames_b)

    full_prompt = prompt_template.format(
        A_summary=A_summary,
        B_summary=B_summary
    )

    # 3. run language generation on Qwen text-only as placeholder
    # Actual Qwen-VL call may look different; adapt here.
    tok = comparator.tokenizer
    model = comparator.model

    inputs = tok(full_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        
        
        # Use the correct model class for generation
        if isinstance(model, Qwen2_5_VLModel):
            # Qwen2.5-VL models need to use the conditional generation wrapper
            generation_model = Qwen2VLForConditionalGeneration.from_pretrained(
                comparator.model_name,
                torch_dtype=comparator.dtype,
                device_map=comparator.device
            )
            output_tokens = generation_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
            )
        else:
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
            )
    out_text = tok.decode(output_tokens[0], skip_special_tokens=True)

    # Post-process: return only the newly generated part if you like
    # We'll just return the full decode for clarity.
    return out_text


########################################
# EXAMPLE MAIN USAGE
########################################

if __name__ == "__main__":
    # --- user inputs ---
    path_a = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240807/000001.npy"
    path_b = "/media/mayunagupta/easystore/MitoSpace4D/data/2024_data/processed_data/20240729/000001.npy"

    # 1. read
    vol_a = load_5d_npy(path_a)  # shape (T,C,Z,X,Y)
    vol_b = load_5d_npy(path_b)

    # 2. (optional) sample
    vol_a_sub = sample_time(vol_a, max_frames=100)
    vol_b_sub = sample_time(vol_b, max_frames=100)

    # 3. view in napari (uncomment if running locally with GUI)
    # view_in_napari(vol_a_sub, name="A")
    # view_in_napari(vol_b_sub, name="B")

    # 4. make synchronized videos
    out_a, out_b = make_matched_videos(
        vol_a_sub,
        vol_b_sub,
        collapse_mode="max",
        max_frames=100,
        fps=10,
        out_a="video_a.mp4",
        out_b="video_b.mp4"
    )
    print("Wrote:", out_a, out_b)

    # 5. init Qwen3.0
    comparator = QwenVideoComparator(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",  # <-- change to your actual checkpoint
        device="cuda",
        dtype=torch.float16,
    )

    # 6. describe differences
    diff_text = describe_differences_qwen(
        comparator,
        video_a_path=out_a,
        video_b_path=out_b,
        prompt_template=(
            "You are an expert in biomedical time-lapse imaging.\n"
            "We have Video A ({A_summary}) and Video B ({B_summary}).\n"
            "List the 5 most important differences between A and B.\n"
            "Focus on morphology, signal intensity patterns, and dynamic events.\n"
            "Answer in bullet points."
        ),
        max_model_frames=16
    )

    print("\n=== DIFFERENCE REPORT ===\n")
    print(diff_text)
