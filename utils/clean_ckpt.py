import argparse
import os.path as osp
import sys
import types

import torch

for _mod_name in (
    "autoencoder",
    "autoencoder.autoencoder_runner",
    "autoencoder.autoencoder_models_resnet",
):
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _mod_name == "autoencoder.autoencoder_runner":
            _stub.AutoEncoderRunner = object
        if _mod_name == "autoencoder.autoencoder_models_resnet":
            _stub.MitoSpace3DAutoencoder = object
        sys.modules[_mod_name] = _stub

from simclr.model import MitoSpace4D  # noqa: E402
from simclr.simclr import SimCLRRunner  # noqa: E402
from utils.utils import load_config  # noqa: E402


def clean_checkpoint(ckpt_in, ckpt_out, prefixes=("model.decoder.",)):
    ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)

    state_dict = ckpt["state_dict"]
    removed = [k for k in state_dict if any(k.startswith(p) for p in prefixes)]
    for k in removed:
        del state_dict[k]

    for opt_state in ckpt.get("optimizer_states", []) or []:
        opt_state.pop("state", None)
        opt_state.pop("param_groups", None)
    ckpt["optimizer_states"] = []
    ckpt["lr_schedulers"] = []

    torch.save(ckpt, ckpt_out)

    print(f"Loaded:  {ckpt_in}")
    print(f"Removed {len(removed)} keys with prefixes {prefixes}")
    print(f"Remaining keys: {len(state_dict)}")
    print(f"Saved:   {ckpt_out}")


def verify_checkpoint(ckpt_path, config_path, dropped_prefixes=("model.decoder.",)):
    """Load the cleaned checkpoint into the SimCLR runner, check key matches, and run a forward pass."""
    print(f"\nVerifying {ckpt_path} loads into SimCLRRunner...")
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MitoSpace4D(
        embedding_size=2048,
        cfg=cfg,
        apply_aug=False,
        decoder_checkpoint_path=None,
    )

    runner = SimCLRRunner.load_from_checkpoint(
        ckpt_path, model=model, cfg=cfg, strict=False, map_location=device
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_keys = set(ckpt["state_dict"].keys())
    runner_keys = set(runner.state_dict().keys())

    missing = sorted(runner_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - runner_keys)

    unexpected_real = [
        k for k in unexpected if not any(k.startswith(p) for p in dropped_prefixes)
    ]
    missing_real = [
        k for k in missing if not any(k.startswith(p) for p in dropped_prefixes)
    ]

    print(f"  Checkpoint keys:        {len(ckpt_keys)}")
    print(f"  Runner state_dict keys: {len(runner_keys)}")
    print(
        f"  Unexpected keys: {len(unexpected)} (excluding dropped prefixes: {len(unexpected_real)})"
    )
    print(
        f"  Missing keys:    {len(missing)} (excluding dropped prefixes: {len(missing_real)})"
    )

    if unexpected_real:
        print("  !! Unexpected keys not accounted for by dropped prefixes:")
        for k in unexpected_real[:10]:
            print(f"     {k}")
    if missing_real:
        print("  !! Missing keys not accounted for by dropped prefixes:")
        for k in missing_real[:10]:
            print(f"     {k}")

    keys_ok = not unexpected_real and not missing_real

    print("\n  Running dummy forward pass...")
    runner_model = runner.model.eval().to(device)
    for p in runner_model.parameters():
        p.requires_grad = False

    timesteps = cfg["model_params"].get("timesteps", 20)
    num_z = cfg["model_params"].get("num_z", 60)
    in_channels = cfg["model_params"].get("in_channels", 1)
    dummy_shape = (1, timesteps, in_channels, num_z, 256, 256)
    print(f"    input shape: {dummy_shape}")

    dummy = torch.rand(*dummy_shape, dtype=torch.float32, device=device)
    try:
        with torch.no_grad():
            features, resnet_feats, proj = runner_model(dummy, get_resnet_feats=True)
        print(f"    features shape: {tuple(features.shape)}  (expected (B, T, 2048))")
        print(
            f"    resnet  shape:  {tuple(resnet_feats.shape)}  (expected (B, T, 512))"
        )
        print(f"    proj    shape:  {tuple(proj.shape)}  (expected (B*T, 512))")

        finite = (
            torch.isfinite(features).all().item()
            and torch.isfinite(resnet_feats).all().item()
            and torch.isfinite(proj).all().item()
        )
        shapes_ok = (
            features.shape == (1, timesteps, 2048)
            and resnet_feats.shape == (1, timesteps, 512)
            and proj.shape == (timesteps, 512)
        )
        forward_ok = finite and shapes_ok
        if not finite:
            print("    !! Output contains non-finite values.")
        if not shapes_ok:
            print("    !! Output shapes do not match expectations.")
    except Exception as e:
        print(f"    !! Forward pass raised: {type(e).__name__}: {e}")
        forward_ok = False

    ok = keys_ok and forward_ok
    print(f"\n  Verification: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove autoencoder weights from a checkpoint."
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/home/earkfeld/Projects/MitoSpace4D/ms4d+ae.ckpt",
        help="Path to the input checkpoint containing the autoencoder.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Path to write the cleaned checkpoint. Defaults to ms4d_cleaned.ckpt alongside the input.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="State-dict key prefix(es) to drop. Repeatable. Defaults to 'model.decoder.'.",
    )
    parser.add_argument(
        "--config",
        default="/home/earkfeld/Projects/MitoSpace4D/simclr/config.yaml",
        help="SimCLR config used to instantiate the model for verification.",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip loading the cleaned checkpoint into SimCLRRunner to verify it.",
    )
    args = parser.parse_args()

    out_path = args.output_path or osp.join(
        osp.dirname(args.checkpoint_path), "ms4d_cleaned.ckpt"
    )
    prefixes = tuple(args.prefix) if args.prefix else ("model.decoder.",)

    clean_checkpoint(args.checkpoint_path, out_path, prefixes=prefixes)

    if not args.skip_verify:
        ok = verify_checkpoint(out_path, args.config, dropped_prefixes=prefixes)
        if not ok:
            raise SystemExit(1)
