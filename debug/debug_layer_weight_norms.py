import torch
import numpy as np

def main():
    # Hardcoded checkpoint path
    # ckpt_path="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_decoupled-tmrm_r20251115_epoch=145-step=25988-val_loss=0.00.ckpt"
    # ckpt_path="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_2024v2_ablated-tmrm_r20251115_epoch=161-step=28836-val_loss=0.00.ckpt"
    
    # ckpt_path="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_decoupled-tmrm_r20251115_epoch=256-step=41120-val_loss=0.00.ckpt"
    ckpt_path="/home/earkfeld/Projects/MitoSpace4D/checkpoints/models_r202511/resnetbilstm_encoded_kinetics_ablated-tmrm_r20251115_epoch=291-step=46720-val_loss=0.00.ckpt"

    # Load checkpoint (map to CPU to avoid GPU requirements)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Extract state_dict
    state_dict = ckpt.get("state_dict", ckpt)

    print(f"Loaded checkpoint: {ckpt_path}")
    print("Parameter norms:\n")

    for name, tensor in state_dict.items():
        # Skip the decoder
        if name.startswith("model.decoder."):
            continue

        if not isinstance(tensor, torch.Tensor):
            continue

        if not tensor.is_floating_point():
            continue

        # Compute L2 norm
        norm = tensor.norm(p=2).item()
        print(f"{name} norm = {norm:.6f}")

if __name__ == "__main__":
    main()
