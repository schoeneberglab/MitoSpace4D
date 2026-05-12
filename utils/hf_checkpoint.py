"""Publish + fetch MitoSpace4D checkpoints via the Hugging Face Hub.

Auth: set `HF_TOKEN` (an access token with write scope for upload, read for
download) in the environment, or run `huggingface-cli login` once. Private
repos under an organization require the user be a member of that org.

Default target: schoeneberglab/mitospace4d (private model repo).

CLI:
    python utils/hf_checkpoint.py upload --ckpt ms4d_cleaned.ckpt
    python utils/hf_checkpoint.py download --filename ms4d_cleaned.ckpt
"""

import argparse
import os
import os.path as osp

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO_ID = "schoeneberglab/mitospace"
DEFAULT_REPO_TYPE = "model"


def upload_checkpoint(ckpt_path, repo_id=DEFAULT_REPO_ID, private=True,
                      path_in_repo=None, commit_message=None, token=None):
    """Create the repo if missing and upload a single checkpoint file.

    The repo is created with `private=True` by default. If it already exists,
    its visibility is left unchanged (HF won't downgrade a public repo to
    private just because we pass `private=True`).
    """
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
    """Download a single file from the model repo and return its local path.

    The file is cached under `~/.cache/huggingface/hub` (or `cache_dir`); a
    subsequent call returns the cached path without re-downloading.
    """
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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upload", help="Publish a checkpoint to the HF hub.")
    up.add_argument("--ckpt", required=True, help="Local path to the .ckpt file.")
    up.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    up.add_argument("--public", action="store_true",
                    help="Create the repo public (default: private).")
    up.add_argument("--path-in-repo", default=None,
                    help="Filename in the repo (default: basename of --ckpt).")
    up.add_argument("--commit-message", default=None)

    dn = sub.add_parser("download", help="Fetch a checkpoint from the HF hub.")
    dn.add_argument("--filename", required=True, help="File path inside the repo.")
    dn.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    dn.add_argument("--revision", default="main")
    dn.add_argument("--cache-dir", default=None)

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


if __name__ == "__main__":
    main()
