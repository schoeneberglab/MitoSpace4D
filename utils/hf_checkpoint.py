"""Publish + fetch MitoSpace4D checkpoints via the Hugging Face Hub.

Auth: set `HF_TOKEN` (an access token with write scope for upload, read for
download) in the environment, or run `huggingface-cli login` once. Private
repos under an organization require the user be a member of that org.

Default target: schoeneberglab/mitospace (private model repo).

CLI:
    python utils/hf_checkpoint.py upload   --ckpt ms4d.ckpt
    python utils/hf_checkpoint.py download --filename model.safetensors
    python utils/hf_checkpoint.py release  --ckpt manuscript_v2/cleaned_ckpts/ms4d.ckpt
"""

import argparse
import json
import os
import os.path as osp
import shutil

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

DEFAULT_REPO_ID = "schoeneberglab/mitospace"
DEFAULT_REPO_TYPE = "model"

PROJECT_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_CKPT = osp.join(PROJECT_ROOT, "manuscript_v2", "cleaned_ckpts", "ms4d.ckpt")
DEFAULT_OUTPUT_DIR = osp.join(PROJECT_ROOT, "manuscript_v2", "cleaned_ckpts", "hf_release")
DEFAULT_LICENSE = osp.join(PROJECT_ROOT, "LICENSE")
DEFAULT_MODEL_CARD = osp.join(PROJECT_ROOT, "MODEL_CARD.md")
DEFAULT_CONFIG_YAML = osp.join(PROJECT_ROOT, "simclr", "config.yaml")


def upload_checkpoint(ckpt_path, repo_id=DEFAULT_REPO_ID, private=True,
                      path_in_repo=None, commit_message=None, token=None):
    """Create the repo if missing and upload a single checkpoint file."""
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE,
                    private=private, exist_ok=True)

    target_name = path_in_repo or osp.basename(ckpt_path)
    commit_message = commit_message or f"Upload {target_name}"

    url = api.upload_file(
        path_or_fileobj=ckpt_path,
        path_in_repo=target_name,
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        commit_message=commit_message,
        token=token,
    )
    print(f"Uploaded {ckpt_path} → {repo_id}/{target_name}")
    print(f"  URL: {url}")
    return url


def download_checkpoint(filename, repo_id=DEFAULT_REPO_ID, revision="main",
                        cache_dir=None, token=None):
    """Download a single file from the model repo and return its local path."""
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        repo_type=DEFAULT_REPO_TYPE,
        cache_dir=cache_dir,
        token=token,
    )
    print(f"Downloaded {repo_id}/{filename} → {local_path}")
    return local_path


def convert_ckpt_to_safetensors(ckpt_path, out_path, strip_prefix="model."):
    """Convert a Lightning .ckpt into a flat safetensors state_dict file.

    Strips the leading `model.` prefix from every state-dict key so the result
    loads directly into `MitoSpace4D` without going through `SimCLRRunner`.
    """
    if not osp.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    cleaned = {}
    skipped = 0
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            skipped += 1
            continue
        new_k = k[len(strip_prefix):] if k.startswith(strip_prefix) else k
        # safetensors requires contiguous, non-shared storage.
        cleaned[new_k] = v.detach().contiguous().clone()

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    save_file(cleaned, out_path)

    size_mb = osp.getsize(out_path) / 1024 ** 2
    print(f"Converted {len(cleaned)} tensors → {out_path} ({size_mb:.1f} MB)"
          + (f"; skipped {skipped} non-tensor entries" if skipped else ""))
    return cleaned


def build_config_json(cfg_yaml_path, out_path, embedding_size=2048,
                      architecture="MitoSpace4D"):
    """Emit a minimal config.json describing the model's input contract."""
    with open(cfg_yaml_path) as f:
        cfg = yaml.safe_load(f)
    mp = cfg["model_params"]

    config = {
        "architecture": architecture,
        "framework": "pytorch",
        "embedding_size": embedding_size,
        "in_channels": mp["in_channels"],
        "channels": mp["channels"],
        "num_z": mp["num_z"],
        "timesteps": mp["timesteps"],
        "bidirectional": mp["bidirectional"],
        "expected_input_shape": [
            "batch", mp["timesteps"], mp["in_channels"], mp["num_z"], 256, 256
        ],
    }

    os.makedirs(osp.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote {out_path}")
    return config


def preflight_checks(license_path=DEFAULT_LICENSE,
                     model_card_path=DEFAULT_MODEL_CARD,
                     cfg_yaml_path=DEFAULT_CONFIG_YAML):
    """Validate release prerequisites; raise RuntimeError listing all failures."""
    errors = []

    if not osp.exists(license_path):
        errors.append(f"LICENSE not found: {license_path}")
    elif osp.getsize(license_path) == 0:
        errors.append(f"LICENSE is empty (0 bytes): {license_path}")

    if not osp.exists(model_card_path):
        errors.append(f"MODEL_CARD.md not found: {model_card_path}")
    else:
        with open(model_card_path) as f:
            content = f.read()
        if not content.startswith("---\n"):
            errors.append(f"MODEL_CARD.md missing YAML frontmatter (must start with `---`): {model_card_path}")
        else:
            try:
                _, fm, _ = content.split("---\n", 2)
                if "pipeline_tag:" not in fm:
                    errors.append("MODEL_CARD.md YAML frontmatter missing `pipeline_tag`")
                if "license:" not in fm:
                    errors.append("MODEL_CARD.md YAML frontmatter missing `license`")
            except ValueError:
                errors.append("MODEL_CARD.md YAML frontmatter not closed (need a second `---` line)")

    if not osp.exists(cfg_yaml_path):
        errors.append(f"config.yaml not found: {cfg_yaml_path}")
    else:
        with open(cfg_yaml_path) as f:
            cfg = yaml.safe_load(f)
        in_channels = cfg.get("model_params", {}).get("in_channels")
        if in_channels != 1:
            errors.append(f"config.yaml model_params.in_channels != 1 (got {in_channels})")

    if errors:
        raise RuntimeError("Preflight checks failed:\n  - " + "\n  - ".join(errors))
    print("Preflight checks passed.")


def verify_safetensors(safetensors_path, cfg_yaml_path=DEFAULT_CONFIG_YAML):
    """Load the safetensors file into a fresh `MitoSpace4D` and run a dummy forward.

    Requires CUDA: `MitoSpace4D.__init__` unconditionally moves the augmentation
    pipeline to cuda, so CPU-only verification isn't supported.
    """
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from simclr.model import MitoSpace4D
    from utils.utils import load_config

    if not torch.cuda.is_available():
        raise RuntimeError("verify_safetensors requires CUDA (model __init__ moves "
                           "augmentation pipeline to cuda unconditionally).")

    cfg = load_config(cfg_yaml_path)
    model = MitoSpace4D(
        embedding_size=2048,
        cfg=cfg,
        apply_aug=False,
        decoder_checkpoint_path=None,
    ).cuda().eval()

    state_dict = load_file(safetensors_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys: {missing[:5]} (and {max(0, len(missing) - 5)} more)")
    if unexpected:
        raise RuntimeError(f"Unexpected keys: {unexpected[:5]} (and {max(0, len(unexpected) - 5)} more)")
    print(f"Loaded {len(state_dict)} tensors into MitoSpace4D; no missing/unexpected keys.")

    # Move the int32 channel-index tensor onto cuda so fancy-indexing works.
    model._channels = model._channels.cuda()

    dummy = torch.rand(1, 20, 1, 60, 256, 256, device="cuda")
    with torch.no_grad():
        feats, resnet_feats, proj = model(dummy, get_resnet_feats=True)

    assert feats.shape == (1, 20, 2048), f"features shape {feats.shape} ≠ (1, 20, 2048)"
    assert resnet_feats.shape == (1, 20, 512), f"resnet shape {resnet_feats.shape} ≠ (1, 20, 512)"
    assert proj.shape == (20, 512), f"proj shape {proj.shape} ≠ (20, 512)"
    assert torch.isfinite(feats).all(), "features contain non-finite values"
    print("Verification PASSED: shapes (1, 20, 2048) / (1, 20, 512) / (20, 512); all finite.")


def build_release_bundle(ckpt_path=DEFAULT_CKPT,
                         output_dir=DEFAULT_OUTPUT_DIR,
                         cfg_yaml_path=DEFAULT_CONFIG_YAML,
                         license_path=DEFAULT_LICENSE,
                         model_card_path=DEFAULT_MODEL_CARD,
                         verify=True):
    """Stage every HF release artifact under `output_dir`."""
    preflight_checks(license_path=license_path,
                     model_card_path=model_card_path,
                     cfg_yaml_path=cfg_yaml_path)

    os.makedirs(output_dir, exist_ok=True)
    safetensors_path = osp.join(output_dir, "model.safetensors")
    config_path = osp.join(output_dir, "config.json")
    readme_path = osp.join(output_dir, "README.md")
    license_dest = osp.join(output_dir, "LICENSE")
    gitattr_path = osp.join(output_dir, ".gitattributes")

    convert_ckpt_to_safetensors(ckpt_path, safetensors_path)
    build_config_json(cfg_yaml_path, config_path)
    shutil.copy(model_card_path, readme_path)
    shutil.copy(license_path, license_dest)

    with open(gitattr_path, "w") as f:
        f.write("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    print(f"Wrote {gitattr_path}")

    if verify:
        verify_safetensors(safetensors_path, cfg_yaml_path)

    print(f"\nRelease bundle staged at {output_dir}:")
    for fname in sorted(os.listdir(output_dir)):
        p = osp.join(output_dir, fname)
        print(f"  {fname:<24} {osp.getsize(p) / 1024:>10.1f} KB")
    return output_dir


def release_to_hf(bundle_dir, repo_id=DEFAULT_REPO_ID, token=None,
                  delete_legacy_ckpt="ms4d.ckpt",
                  commit_message=None):
    """Upload the bundle and delete the legacy `.ckpt` from the repo."""
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=DEFAULT_REPO_TYPE,
                    private=True, exist_ok=True)

    commit_message = commit_message or "Release MitoSpace4D bundle (safetensors + model card)"
    api.upload_folder(
        folder_path=bundle_dir,
        repo_id=repo_id,
        repo_type=DEFAULT_REPO_TYPE,
        commit_message=commit_message,
        token=token,
    )
    print(f"\nUploaded {bundle_dir} → {repo_id}")

    if delete_legacy_ckpt:
        try:
            api.delete_file(
                path_in_repo=delete_legacy_ckpt,
                repo_id=repo_id,
                repo_type=DEFAULT_REPO_TYPE,
                commit_message=f"Remove legacy {delete_legacy_ckpt} (replaced by model.safetensors)",
                token=token,
            )
            print(f"Deleted legacy {delete_legacy_ckpt} from {repo_id}.")
        except Exception as e:
            print(f"(Legacy {delete_legacy_ckpt} not deleted — already absent? {type(e).__name__}: {e})")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Publish a single file to the HF hub.")
    up.add_argument("--ckpt", required=True, help="Local path to the file to upload.")
    up.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    up.add_argument("--public", action="store_true",
                    help="Create the repo public (default: private).")
    up.add_argument("--path-in-repo", default=None,
                    help="Filename in the repo (default: basename of --ckpt).")
    up.add_argument("--commit-message", default=None)

    dn = sub.add_parser("download", help="Fetch a single file from the HF hub.")
    dn.add_argument("--filename", required=True, help="File path inside the repo.")
    dn.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    dn.add_argument("--revision", default="main")
    dn.add_argument("--cache-dir", default=None)

    rel = sub.add_parser("release",
                         help="Build the full release bundle and upload it.")
    rel.add_argument("--ckpt", default=DEFAULT_CKPT)
    rel.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    rel.add_argument("--license", default=DEFAULT_LICENSE)
    rel.add_argument("--model-card", default=DEFAULT_MODEL_CARD)
    rel.add_argument("--config", default=DEFAULT_CONFIG_YAML)
    rel.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    rel.add_argument("--skip-verify", action="store_true",
                     help="Skip the dummy-forward verification.")
    rel.add_argument("--skip-upload", action="store_true",
                     help="Build the bundle but don't push to HF.")
    rel.add_argument("--commit-message", default=None)

    return parser.parse_args()


def main():
    args = _parse_args()
    token = os.environ.get("HF_TOKEN")

    if args.cmd == "upload":
        upload_checkpoint(
            ckpt_path=args.ckpt,
            repo_id=args.repo_id,
            private=not args.public,
            path_in_repo=args.path_in_repo,
            commit_message=args.commit_message,
            token=token,
        )
    elif args.cmd == "download":
        download_checkpoint(
            filename=args.filename,
            repo_id=args.repo_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            token=token,
        )
    elif args.cmd == "release":
        bundle_dir = build_release_bundle(
            ckpt_path=args.ckpt,
            output_dir=args.output_dir,
            cfg_yaml_path=args.config,
            license_path=args.license,
            model_card_path=args.model_card,
            verify=not args.skip_verify,
        )
        if not args.skip_upload:
            release_to_hf(
                bundle_dir=bundle_dir,
                repo_id=args.repo_id,
                token=token,
                commit_message=args.commit_message,
            )


if __name__ == "__main__":
    main()
