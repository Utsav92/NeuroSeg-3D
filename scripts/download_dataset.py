#!/usr/bin/env python
"""Verify endpoints, download (resumable), integrity-check, record provenance and extract Task01_BrainTumour.

Only official AWS Open Data Registry endpoints listed in configs/datasets.yaml are used. BraTS (Synapse/TCIA)
datasets require registration and are never downloaded automatically.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import stat
import sys
import tarfile
from pathlib import Path

import _bootstrap  # noqa: F401
import requests
from _bootstrap import ROOT

from neuroseg.utils.logging import load_yaml


def probe(url):
    r = requests.head(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return {
        "url": url,
        "status": r.status_code,
        "content_length": int(r.headers.get("Content-Length", -1)),
        "accept_ranges": r.headers.get("Accept-Ranges"),
        "etag": r.headers.get("ETag"),
        "last_modified": r.headers.get("Last-Modified"),
    }


def download(url, dest: Path, total: int):
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have > total:
        raise SystemExit(f"{dest} is larger ({have}) than expected {total}; delete it and retry")
    if have == total:
        print(f"already complete: {dest} ({have} bytes)")
        return
    print(f"downloading {url}\n  resuming at {have / 2**20:.1f} MiB of {total / 2**20:.1f} MiB")
    with requests.get(url, headers={"Range": f"bytes={have}-"} if have else {}, stream=True, timeout=60) as r:
        if have and r.status_code != 206:
            raise SystemExit("host did not honour Range request; delete the partial file to restart")
        r.raise_for_status()
        done = have
        with open(dest, "ab" if have else "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                print(f"\r  {done / total * 100:5.1f}%  {done / 2**20:8.1f} MiB", end="", flush=True)
    print()


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def safe_extract(tar_path: Path, out: Path):
    out = out.resolve()
    with tarfile.open(tar_path) as tf:
        members = [m for m in tf.getmembers() if not Path(m.name).name.startswith("._")]
        for m in members:
            target = (out / m.name).resolve()
            if not str(target).startswith(str(out)):
                raise SystemExit(f"unsafe path in archive: {m.name}")
        tf.extractall(out, members=members, filter="data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="msd_brain")
    ap.add_argument("--config", default=str(ROOT / "configs/datasets.yaml"))
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--skip-sha256", action="store_true")
    a = ap.parse_args()
    cfg = load_yaml(a.config)["datasets"][a.dataset]
    if cfg.get("access") == "manual":
        sys.exit(
            f"'{a.dataset}' requires registration / a data-use agreement and is not downloaded automatically. See docs/DATASETS.md."
        )
    dl = cfg["download"]
    probes, chosen = [], None
    for u in dl["urls"]:
        try:
            p = probe(u)
            probes.append(p)
            ok = p["content_length"] == dl["expected_size_bytes"]
            print(
                f"endpoint OK: {u}  size={p['content_length']}  ranges={p['accept_ranges']}  matches expected size: {ok}"
            )
            if chosen is None and ok:
                chosen = p
        except Exception as e:  # noqa: BLE001
            print(f"endpoint FAILED: {u}: {e}")
    if chosen is None:
        sys.exit("no endpoint with the expected size is reachable; refusing to substitute another source")
    archive = Path(dl["archive"])
    download(chosen["url"], archive, dl["expected_size_bytes"])
    size = archive.stat().st_size
    if size != dl["expected_size_bytes"]:
        sys.exit(f"size mismatch: {size} != {dl['expected_size_bytes']}")
    digest = None if a.skip_sha256 else sha256(archive)
    print("sha256:", digest)
    with tarfile.open(archive) as tf:  # full read of the index validates the container structure
        names = tf.getnames()
    print(f"archive index OK: {len(names)} members")
    root = Path(cfg["root"])
    if not a.no_extract and not (root / "dataset.json").exists():
        print("extracting (original archive is kept unchanged)...")
        safe_extract(archive, root.parent)
        for p in root.rglob("*"):  # make the extracted original read-only
            if p.is_file():
                os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    manifest = {
        "dataset": a.dataset,
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_url": chosen["url"],
        "registry": "https://registry.opendata.aws/msd/",
        "endpoint_probes": probes,
        "archive": str(archive),
        "size_bytes": size,
        "sha256": digest,
        "license": cfg["license"],
        "tar_members": len(names),
    }
    Path("data/manifests").mkdir(parents=True, exist_ok=True)
    Path(f"data/manifests/{a.dataset}_download.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote data/manifests/{a.dataset}_download.json")


if __name__ == "__main__":
    main()
