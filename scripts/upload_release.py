"""Publish a complete checkpoint to a private Hugging Face model repository."""

import argparse
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from safetensors import safe_open


def upload_release(directory, repo):
    root = Path(directory)
    required = ["model.safetensors", "config.json", "processor_config.json", "README.md", "LICENSE"]
    for file in required:
        if not (root / file).is_file():
            raise ValueError(f"Release is incomplete: missing {file}")
    allowed = set(required) | {"NOTICE", "COLIPRI_LICENSE"}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    paths = {p.relative_to(root).as_posix() for p in files}
    for path in paths:
        if path not in allowed and not (
            path.startswith("assets/") and Path(path).suffix in {".png", ".jpg", ".svg", ".pdf"}
        ):
            raise ValueError(f"Hub export must contain weights and metadata only; move source to GitHub: {path}")
    config = json.loads((root / "config.json").read_text())
    if not config.get("encoder_bundled") or not config.get("encoder_model_config"):
        raise ValueError("Publish a complete bundled model; head-only test fixtures must never replace Hub weights")
    with safe_open(root / "model.safetensors", framework="numpy") as checkpoint:
        keys = set(checkpoint.keys())
        encoder_keys = {key for key in keys if key.startswith("encoder.")}
        if len(encoder_keys) != 380 or not set(config.get("tensor_specs", {})) <= keys:
            raise ValueError("The checkpoint must contain the complete pinned encoder and Rev head tensors")
    api = HfApi()
    identity = api.whoami()
    namespaces = {identity["name"]} | {
        org["name"]
        for org in identity.get("orgs", [])
        if org.get("roleInOrg") in {"write", "admin"}
    }
    if repo.split("/")[0] not in namespaces:
        raise ValueError("Choose your account or an organisation where you have write access")
    try:
        info = api.model_info(repo)
    except RepositoryNotFoundError:
        api.create_repo(repo, repo_type="model", private=True)
        info = api.model_info(repo)
    if not info.private:
        raise ValueError("Choose a private repository for this upload")
    obsolete = sorted(set(api.list_repo_files(repo, revision=info.sha)) - paths - {".gitattributes"})
    operations = [CommitOperationAdd(path_in_repo=p.relative_to(root).as_posix(), path_or_fileobj=p) for p in files]
    operations += [CommitOperationDelete(path_in_repo=p) for p in obsolete]
    commit = api.create_commit(
        repo_id=repo,
        repo_type="model",
        operations=operations,
        commit_message="Release Rev CT weights, decision-model metadata and model card; code on GitHub",
        parent_commit=info.sha,
    )
    print(commit.commit_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--repo", default="hcltech-robotics/rev-ct")
    args = parser.parse_args()
    upload_release(args.directory, args.repo)
