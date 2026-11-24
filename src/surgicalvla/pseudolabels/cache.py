"""Columnar feature and label cache (Parquet shards), keyed by (dataset, episode, frame).

Layout::

    <root>/<dataset-slug>/<cache-name>/
        manifest.json                    provenance, labeler settings, config hash, shard index
        shards/frames-00012.parquet      one row per cached frame (inputs and/or targets)
        shards/episodes-00012.parquet    one row per episode (instruction, phrase, phases)
        shards/cot-00003.parquet         reasoning traces (scripts/make_cot_traces.py)

Only derived data is stored, never frames. Rows carry the LeRobot ``episode_index`` and
``frame_index``, so the cache lines up with the dataset whether it is read from disk or
streamed. Shards are written atomically and recorded in the manifest with a SHA-256, so an
interrupted run resumes after the last complete shard and a damaged shard is detected on read.

Disk use is bounded (``max_gb``): with ``upload_repo`` set, every finished shard is uploaded
to a Hub dataset, and uploaded shards are evicted locally, least recently used first. Readers
download evicted shards again on demand.

A reader states what it expects (``expect``). Any difference from the manifest raises an
error naming each differing field, so a model is never trained on a cache made with other
settings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 3


# --------------------------------------------------------------------------- masks
def rle_encode(mask: np.ndarray) -> np.ndarray:
    """Run lengths of a boolean mask in row-major order, starting with a run of False."""
    flat = np.asarray(mask, bool).reshape(-1)
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    edges = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(edges)
    if flat.size and flat[0]:
        runs = np.concatenate([[0], runs])
    return runs.astype(np.int32)


def rle_decode(runs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    vals = np.zeros(len(runs), bool)
    vals[1::2] = True
    return np.repeat(vals, np.asarray(runs, np.int64)).reshape(shape)


# --------------------------------------------------------------------------- provenance
def file_sha256(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def hub_weights(repo_id: str) -> dict:
    """Revision and weight-file hashes of a Hub model from the local cache (no download).

    LFS files are stored under their SHA-256 in the Hugging Face cache, so the hash is read
    from the blob name instead of re-hashing gigabytes.
    """
    from huggingface_hub import snapshot_download

    try:
        path = Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return {"id": repo_id, "revision": None, "weights": {}}
    weights = {}
    for f in sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin")) + sorted(path.glob("*.pt")):
        target = f.resolve()
        weights[f.name] = target.name if re.fullmatch(r"[0-9a-f]{64}", target.name) else file_sha256(target)
    return {"id": repo_id, "revision": path.name, "weights": weights}


def config_hash(labeler: dict, modules: dict) -> str:
    """Hash of everything that changes the cached values (not of timestamps or hosts)."""
    payload = {"schema": SCHEMA_VERSION, "labeler": labeler,
               "modules": {k: {kk: v.get(kk) for kk in ("id", "revision", "weights", "sha256")} for k, v in modules.items()}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


class CacheMismatch(ValueError):
    pass


def check_manifest(manifest: dict, expect: dict) -> None:
    """Raise :class:`CacheMismatch` naming every field where ``manifest`` differs from ``expect``.

    ``expect`` keys are dotted paths into the manifest (``labeler.stride``,
    ``labeler.object_inputs.num_candidates``, ``config_hash``, ``schema_version``).
    """
    flat = flatten(manifest)
    diffs = []
    for key, want in flatten(expect).items():
        have = flat.get(key, "<missing>")
        if isinstance(want, (list, tuple)) or isinstance(have, (list, tuple)):
            same = list(want or []) == list(have) if have != "<missing>" else False
        else:
            same = want == have
        if not same:
            diffs.append(f"{key}: training config expects {want!r}, cache has {have!r}")
    if diffs:
        raise CacheMismatch("feature cache does not match the training config:\n  " + "\n  ".join(diffs))


# --------------------------------------------------------------------------- columns
def _fixed(arr: np.ndarray):
    import pyarrow as pa

    arr = np.ascontiguousarray(arr)
    width = int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1
    flat = pa.array(arr.reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, width) if arr.ndim > 1 else flat


def _ragged(rows: list[np.ndarray]):
    import pyarrow as pa

    return pa.array([np.asarray(r).tolist() for r in rows], type=pa.list_(pa.int32()))


def _np(col, shape_tail: tuple[int, ...] = ()) -> np.ndarray:
    """Arrow column -> numpy, restoring the per-row shape."""
    import pyarrow as pa

    col = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    if pa.types.is_fixed_size_list(col.type):
        vals = col.flatten().to_numpy(zero_copy_only=False)
        return vals.reshape(len(col), *shape_tail) if shape_tail else vals.reshape(len(col), -1)
    return col.to_numpy(zero_copy_only=False)


# --------------------------------------------------------------------------- store
class FeatureCache:
    """Writer and reader of one cache directory."""

    def __init__(self, root: str | Path, dataset_id: str, name: str, *, manifest: dict | None = None,
                 max_gb: float = 20.0, upload_repo: str | None = None, shard_episodes: int = 8,
                 lru_shards: int = 16, expect: dict | None = None):
        self.dir = Path(root) / re.sub(r"[^A-Za-z0-9._-]+", "__", dataset_id) / name
        self.shard_dir = self.dir / "shards"
        self.manifest_path = self.dir / "manifest.json"
        self.dataset_id = dataset_id
        self.name = name
        self.max_bytes = int(max_gb * 1024**3)
        self.upload_repo = upload_repo
        self.shard_episodes = int(shard_episodes)
        self._pending: list[tuple[dict, dict]] = []
        self._lru: OrderedDict[int, dict] = OrderedDict()
        self._lru_size = int(lru_shards)
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if self.manifest.get("schema_version") != SCHEMA_VERSION:
                raise CacheMismatch(f"{self.manifest_path}: schema_version {self.manifest.get('schema_version')} != {SCHEMA_VERSION}")
            if manifest is not None and manifest.get("config_hash") != self.manifest.get("config_hash"):
                check_manifest(self.manifest, {"labeler": manifest.get("labeler", {}), "config_hash": manifest.get("config_hash")})
        elif manifest is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.shard_dir.mkdir(exist_ok=True)
            self.manifest = {**manifest, "schema_version": SCHEMA_VERSION, "dataset": dataset_id,
                             "created": manifest.get("created") or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "shards": [], "episodes": {}, "cot_shards": []}
            self._save_manifest()
        else:
            raise FileNotFoundError(f"no feature cache at {self.dir}")
        if expect:
            check_manifest(self.manifest, expect)

    # ------------------------------------------------------------------ manifest
    def _save_manifest(self) -> None:
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=1, default=str))
        os.replace(tmp, self.manifest_path)

    def done(self, episode: int) -> bool:
        return str(int(episode)) in self.manifest["episodes"] or any(int(e["episode"]) == int(episode) for e, _ in self._pending)

    def episodes(self) -> list[int]:
        return sorted(int(e) for e in self.manifest["episodes"])

    # ------------------------------------------------------------------ write
    def add(self, episode: dict, frames: dict) -> None:
        """Queue one episode. ``episode``: scalars and lists; ``frames``: column -> per-row array.

        ``frames`` must hold ``frame`` (dataset frame index) and the same number of rows in
        every column. A shard is written once ``shard_episodes`` episodes are queued.
        """
        n = len(frames["frame"])
        for k, v in frames.items():
            if len(v) != n:
                raise ValueError(f"column {k} has {len(v)} rows, expected {n}")
        self._pending.append((episode, frames))
        if len(self._pending) >= self.shard_episodes:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        idx = len(self.manifest["shards"])
        cols: dict[str, list] = {}
        for ep, fr in self._pending:
            n = len(fr["frame"])
            fr = {"episode": np.full(n, int(ep["episode"]), np.int32), **fr}
            for k, v in fr.items():
                cols.setdefault(k, []).append(v)
        table = {}
        shapes = self.manifest.setdefault("columns", {})
        for k, parts in cols.items():
            if isinstance(parts[0], list):  # ragged (RLE masks)
                table[k] = _ragged([r for p in parts for r in p])
            else:
                arr = np.concatenate(parts)
                shapes[k] = list(arr.shape[1:])
                table[k] = _fixed(arr)
        ep_table = pa.Table.from_pylist([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in ep.items()}
                                         for ep, _ in self._pending])
        written = []
        for kind, t in (("frames", pa.table(table)), ("episodes", ep_table)):
            path = self.shard_dir / f"{kind}-{idx:05d}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            pq.write_table(t, tmp, compression="zstd")
            os.replace(tmp, path)
            written.append(path)
        shard = {"index": idx, "files": {p.name: file_sha256(p) for p in written},
                 "bytes": sum(p.stat().st_size for p in written), "episodes": [int(ep["episode"]) for ep, _ in self._pending],
                 "uploaded": False}
        self.manifest["shards"].append(shard)
        for ep, _ in self._pending:
            self.manifest["episodes"][str(int(ep["episode"]))] = idx
        self._pending = []
        self._save_manifest()
        if self.upload_repo:
            self._upload(shard)
        self._enforce_cap()

    def _upload(self, shard: dict) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(self.upload_repo, repo_type="dataset", private=True, exist_ok=True)
        prefix = f"{self.dir.parent.name}/{self.name}"
        for fname in shard["files"]:
            api.upload_file(path_or_fileobj=str(self.shard_dir / fname), path_in_repo=f"{prefix}/shards/{fname}",
                            repo_id=self.upload_repo, repo_type="dataset")
        shard["uploaded"] = True
        self._save_manifest()
        api.upload_file(path_or_fileobj=str(self.manifest_path), path_in_repo=f"{prefix}/manifest.json",
                        repo_id=self.upload_repo, repo_type="dataset")

    def disk_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.shard_dir.glob("*.parquet"))

    def _enforce_cap(self) -> None:
        used = self.disk_bytes()
        if used <= self.max_bytes:
            return
        local = [s for s in self.manifest["shards"] if s["uploaded"] and all((self.shard_dir / f).exists() for f in s["files"])]
        local.sort(key=lambda s: max((self.shard_dir / f).stat().st_atime for f in s["files"]))
        for s in local:
            if used <= self.max_bytes:
                return
            for f in s["files"]:
                used -= (self.shard_dir / f).stat().st_size
                (self.shard_dir / f).unlink()
        if used > self.max_bytes:
            raise RuntimeError(f"feature cache {self.dir} exceeds max_gb and nothing uploaded can be evicted; "
                               "set upload_repo or raise max_gb")

    # ------------------------------------------------------------------ read
    def _shard_file(self, shard: dict, fname: str) -> Path:
        path = self.shard_dir / fname
        if not path.exists():
            if not shard.get("uploaded") or not self.upload_repo:
                raise FileNotFoundError(f"{path} missing and not uploaded")
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(self.upload_repo, f"{self.dir.parent.name}/{self.name}/shards/{fname}", repo_type="dataset")
            self.shard_dir.mkdir(parents=True, exist_ok=True)
            os.replace(got, path) if Path(got).parent == path.parent else path.write_bytes(Path(got).read_bytes())
        return path

    def _load_shard(self, idx: int) -> dict:
        if idx in self._lru:
            self._lru.move_to_end(idx)
            return self._lru[idx]
        import pyarrow.parquet as pq

        shard = self.manifest["shards"][idx]
        tables = {}
        for fname, digest in shard["files"].items():
            path = self._shard_file(shard, fname)
            if file_sha256(path) != digest:
                raise CacheMismatch(f"{path}: checksum differs from the manifest (damaged or replaced shard)")
            tables[fname.split("-")[0]] = pq.read_table(path)
        self._lru[idx] = tables
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)
        return tables

    def read(self, episode: int) -> dict | None:
        """All cached arrays of one episode (see ``scripts/make_pseudolabels.py`` for columns)."""
        idx = self.manifest["episodes"].get(str(int(episode)))
        if idx is None:
            return None
        t = self._load_shard(int(idx))
        frames = t["frames"]
        ep_col = frames.column("episode").to_numpy()
        rows = np.flatnonzero(ep_col == int(episode))
        sub = frames.take(rows)
        shapes = self.manifest.get("columns", {})
        out = {}
        for name in sub.column_names:
            col = sub.column(name)
            import pyarrow as pa

            if pa.types.is_list(col.type):
                out[name] = [np.asarray(v, np.int32) for v in col.to_pylist()]
            else:
                out[name] = _np(col, tuple(shapes.get(name, ())))
        eps = t["episodes"].to_pylist()
        meta = next(e for e in eps if int(e["episode"]) == int(episode))
        for k, v in meta.items():
            out[f"ep.{k}"] = json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v
        return out


# --------------------------------------------------------------------------- reasoning traces
def write_traces(cache: FeatureCache, rows: list[dict], tokenizer_id: str | None = None) -> None:
    """Append reasoning-trace rows (episode, segment, start, end, trace, gain, kept, token_ids)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    idx = len(cache.manifest["cot_shards"])
    path = cache.shard_dir / f"cot-{idx:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    os.replace(tmp, path)
    cache.manifest["cot_shards"].append({"file": path.name, "sha256": file_sha256(path),
                                         "episodes": sorted({int(r["episode"]) for r in rows})})
    cache.manifest["cot_tokenizer"] = tokenizer_id
    cache._save_manifest()


def read_traces(cache: FeatureCache) -> dict[int, list[dict]]:
    """episode -> trace rows from every trace shard (later shards replace earlier rows)."""
    import pyarrow.parquet as pq

    out: dict[int, list[dict]] = {}
    for s in cache.manifest.get("cot_shards", []):
        path = cache.shard_dir / s["file"]
        if file_sha256(path) != s["sha256"]:
            raise CacheMismatch(f"{path}: checksum differs from the manifest")
        rows = pq.read_table(path).to_pylist()
        for ep in {int(r["episode"]) for r in rows}:
            out[ep] = [r for r in rows if int(r["episode"]) == ep]
    return out


__all__ = ["SCHEMA_VERSION", "CacheMismatch", "FeatureCache", "check_manifest", "config_hash", "file_sha256",
           "hub_weights", "read_traces", "rle_decode", "rle_encode", "write_traces"]
