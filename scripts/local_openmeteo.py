#!/usr/bin/env python3
"""Run a local Open-Meteo API server backed directly by AWS Open Data (s3://openmeteo/data/).

Uses Open-Meteo's official cloud-native REMOTE_DATA_DIRECTORY mode, which fetches only the
needed 64 KB compressed blocks from https://openmeteo.s3.amazonaws.com/data/ via HTTP Range
requests into a bounded local LRU block cache (no AWS account, no API quota, identical JSON
and elevation responses to historical-forecast-api.open-meteo.com).

On Linux x86_64 hosts without Docker (such as Google Colab on Ubuntu 22.04 or 24.04), pulls
the official ghcr.io/open-meteo/open-meteo OCI image layers (~225 MB compressed) into a
self-contained rootfs under /tmp/openmeteo-root and launches the server via its bundled
dynamic linker so host glibc / libnetcdf / tzdata versions never conflict.

    python3 scripts/local_openmeteo.py --start --cache-gb 12
    python3 scripts/local_openmeteo.py --status
    python3 scripts/local_openmeteo.py --probe
    python3 scripts/local_openmeteo.py --stop
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import signal
import socket
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

IMAGE_REPO = "open-meteo/open-meteo"
IMAGE_TAG = "latest"
REMOTE_DATA_DIR = "https://openmeteo.s3.amazonaws.com/data/"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_WORK_DIR = Path("/tmp/openmeteo-local")
DEFAULT_PROBE_PARAMS = {
    "latitude": "46.5000",
    "longitude": "8.0000",
    "start_date": "2024-06-15",
    "end_date": "2024-06-15",
    "hourly": "temperature_2m,temperature_850hPa,geopotential_height_850hPa",
    "models": "icon_global",
    "timezone": "UTC",
}

def _make_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class _CrossOriginSafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Drop Authorization headers when registries redirect blob downloads to a CDN host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            old_host = urllib.parse.urlparse(req.full_url).hostname
            new_host = urllib.parse.urlparse(newurl).hostname
            if old_host != new_host:
                new_req.remove_header("Authorization")
        return new_req


_SSL_CTX = _make_ssl_context()
_OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=_SSL_CTX),
    _CrossOriginSafeRedirectHandler(),
)


def _http_get_bytes(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pre-task-analyzer-local-om/1", **(headers or {})})
    with _OPENER.open(req, timeout=timeout) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding", "").lower() == "gzip":
            data = gzip.decompress(data)
        return data


def _http_get_json(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> Any:
    return json.loads(_http_get_bytes(url, headers=headers, timeout=timeout).decode("utf-8"))


def _download_to_file(url: str, dest: Path, headers: dict[str, str] | None = None, timeout: float = 180.0) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": "pre-task-analyzer-local-om/1", **(headers or {})})
    tmp = dest.with_suffix(dest.suffix + ".part")
    total = 0
    with _OPENER.open(req, timeout=timeout) as resp, tmp.open("wb") as fh:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return total


def _linux_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    raise RuntimeError(f"Unsupported Linux architecture for openmeteo-api: {machine}")


def select_oci_manifest_digest(index_doc: dict[str, Any], arch: str) -> str:
    """Pick the Linux manifest digest matching `arch` from an OCI image index."""
    manifests = index_doc.get("manifests") or []
    for entry in manifests:
        plat = entry.get("platform") or {}
        if plat.get("os") == "linux" and plat.get("architecture") == arch:
            digest = entry.get("digest")
            if digest:
                return str(digest)
    raise RuntimeError(f"No linux/{arch} manifest found in OCI index")


def _safe_rel_path(member_name: str) -> Path | None:
    """Normalize a tar member path and drop unsafe traversal or root entries."""
    norm = member_name.lstrip("/")
    if not norm or norm == ".":
        return None
    parts = [p for p in Path(norm).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return Path(*parts)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def _assert_within_rootfs(path: Path, rootfs_resolved: Path) -> None:
    """Verify `path` resolves inside `rootfs_resolved` so symlinks can never escape onto the host."""
    resolved = path.resolve()
    if os.path.commonpath([str(rootfs_resolved), str(resolved)]) != str(rootfs_resolved):
        raise RuntimeError(f"Refusing to write outside rootfs: {path} -> {resolved}")


def extract_oci_layer(tar_path: Path, rootfs: Path) -> None:
    """Extract one OCI tar(.gz) layer into `rootfs`, honoring OCI whiteouts and symlinks."""
    rootfs_resolved = rootfs.resolve()
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        # First pass: apply whiteouts (.wh..wh..opq and .wh.<name>) before writing new files.
        for member in members:
            rel = _safe_rel_path(member.name)
            if rel is None:
                continue
            base = rel.name
            parent = rootfs / rel.parent
            if base == ".wh..wh..opq":
                if parent.is_dir():
                    _assert_within_rootfs(parent, rootfs_resolved)
                    for child in list(parent.iterdir()):
                        _remove_path(child)
            elif base.startswith(".wh."):
                if parent.exists():
                    _assert_within_rootfs(parent, rootfs_resolved)
                target = parent / base[4:]
                _remove_path(target)

        # Second pass: extract directories, files, symlinks, and hardlinks.
        deferred_hardlinks: list[tuple[Path, Path]] = []
        for member in members:
            rel = _safe_rel_path(member.name)
            if rel is None:
                continue
            if rel.name.startswith(".wh."):
                continue
            if member.ischr() or member.isblk() or member.isfifo():
                continue

            dest = rootfs / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            _assert_within_rootfs(dest.parent, rootfs_resolved)
            # Ensure parent directory is writable before overwriting from a previous layer.
            if dest.parent.exists() and not dest.parent.is_symlink():
                try:
                    os.chmod(dest.parent, os.stat(dest.parent).st_mode | 0o700)
                except OSError:
                    pass

            if member.isdir():
                if dest.is_symlink() or dest.is_file():
                    dest.unlink(missing_ok=True)
                dest.mkdir(parents=True, exist_ok=True)
                _assert_within_rootfs(dest, rootfs_resolved)
                try:
                    os.chmod(dest, (member.mode & 0o777) | 0o700)
                except OSError:
                    pass
                continue

            if dest.exists() or dest.is_symlink():
                _remove_path(dest)

            if member.issym():
                link_target = member.linkname
                if link_target.startswith("/"):
                    # Keep every absolute symlink contained inside rootfs before later layers traverse it.
                    in_root = rootfs / link_target.lstrip("/")
                    link_target = os.path.relpath(in_root, start=dest.parent)
                os.symlink(link_target, dest)
            elif member.islnk():
                link_rel = _safe_rel_path(member.linkname)
                if link_rel is not None:
                    deferred_hardlinks.append((rootfs / link_rel, dest))
            elif member.isfile():
                src = tf.extractfile(member)
                if src is None:
                    continue
                with src, dest.open("wb") as out_f:
                    shutil.copyfileobj(src, out_f, length=1024 * 1024)
                try:
                    os.chmod(dest, member.mode & 0o777)
                except OSError:
                    pass

        for src_p, dst_p in deferred_hardlinks:
            if src_p.exists() and not dst_p.exists():
                dst_p.parent.mkdir(parents=True, exist_ok=True)
                _assert_within_rootfs(dst_p.parent, rootfs_resolved)
                try:
                    os.link(src_p, dst_p)
                except OSError:
                    shutil.copy2(src_p, dst_p)


def rewrite_absolute_symlinks(rootfs: Path) -> int:
    """Convert absolute symlinks inside `rootfs` to relative symlinks when their target exists
    inside `rootfs`, so ld-linux never accidentally follows a link into the host filesystem."""
    fixed = 0
    for dirpath, dirnames, filenames in os.walk(rootfs, followlinks=False):
        dp = Path(dirpath)
        for name in (*dirnames, *filenames):
            p = dp / name
            if not p.is_symlink():
                continue
            target = os.readlink(p)
            if not target.startswith("/"):
                continue
            in_root = rootfs / target.lstrip("/")
            if in_root.exists() or in_root.is_symlink():
                rel_target = os.path.relpath(in_root, start=dp)
                p.unlink()
                os.symlink(rel_target, p)
                fixed += 1
    return fixed


def compute_safe_cache_gb(requested_gb: int, cache_dir: Path, min_gb: int = 2) -> int:
    """Clamp requested cache size to at most 25% of free disk space on the target volume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    free_gb = int(shutil.disk_usage(cache_dir).free // (1024 ** 3))
    max_safe = max(min_gb, int(free_gb * 0.25))
    return max(min_gb, min(int(requested_gb), max_safe))


def fetch_oci_rootfs(rootfs: Path, image_repo: str = IMAGE_REPO, tag: str = IMAGE_TAG) -> Path:
    """Pull and unpack ghcr.io/<image_repo>:<tag> for the host Linux architecture into `rootfs`."""
    arch = _linux_arch()
    tok_url = f"https://ghcr.io/token?service=ghcr.io&scope=repository:{image_repo}:pull"
    token = _http_get_json(tok_url)["token"]
    auth_hdr = {"Authorization": f"Bearer {token}"}

    idx = _http_get_json(
        f"https://ghcr.io/v2/{image_repo}/manifests/{tag}",
        headers={
            **auth_hdr,
            "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json",
        },
    )
    if "manifests" in idx:
        digest = select_oci_manifest_digest(idx, arch)
        manifest = _http_get_json(
            f"https://ghcr.io/v2/{image_repo}/manifests/{digest}",
            headers={
                **auth_hdr,
                "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
            },
        )
    else:
        digest = f"{tag}-{arch}"
        manifest = idx

    marker = rootfs / ".openmeteo_ready.json"
    binary = rootfs / "app" / "openmeteo-api"
    if marker.exists() and binary.exists():
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
            if meta.get("digest") == digest:
                print(f"Reusing extracted openmeteo-api rootfs ({digest[:19]}...) at {rootfs}", flush=True)
                return rootfs
        except Exception:
            pass

    layers = manifest.get("layers") or []
    if not layers:
        raise RuntimeError("OCI manifest contains no layers")
    total_mb = sum(int(l.get("size", 0)) for l in layers) / 1e6
    print(f"Pulling ghcr.io/{image_repo}:{tag} (linux/{arch}, {len(layers)} layers, {total_mb:.1f} MB)...", flush=True)

    if rootfs.exists():
        shutil.rmtree(rootfs)
    rootfs.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="om-layers-") as tmp_dir:
        tmp_p = Path(tmp_dir)
        for idx_i, layer in enumerate(layers, 1):
            l_digest = layer["digest"]
            l_mb = int(layer.get("size", 0)) / 1e6
            tar_file = tmp_p / f"layer_{idx_i:02d}.tar.gz"
            blob_url = f"https://ghcr.io/v2/{image_repo}/blobs/{l_digest}"
            _download_to_file(blob_url, tar_file, headers=auth_hdr)
            extract_oci_layer(tar_file, rootfs)
            tar_file.unlink(missing_ok=True)
            if l_mb >= 5.0 or idx_i == len(layers):
                print(f"  layer {idx_i:2d}/{len(layers)} extracted ({l_mb:5.1f} MB)", flush=True)

    n_fixed = rewrite_absolute_symlinks(rootfs)
    if not binary.exists():
        raise RuntimeError(f"Expected binary not found after OCI extract: {binary}")
    marker.write_text(json.dumps({"image": f"{image_repo}:{tag}", "arch": arch, "digest": digest}, indent=2), encoding="utf-8")
    print(f"Prepared self-contained openmeteo-api rootfs at {rootfs} (rewrote {n_fixed} absolute symlinks)", flush=True)
    return rootfs


def _find_rootfs_linker(rootfs: Path) -> Path:
    arch = _linux_arch()
    candidates = (
        [
            rootfs / "usr" / "lib" / "x86_64-linux-gnu" / "ld-linux-x86-64.so.2",
            rootfs / "lib" / "x86_64-linux-gnu" / "ld-linux-x86-64.so.2",
            rootfs / "lib64" / "ld-linux-x86-64.so.2",
        ]
        if arch == "amd64"
        else [
            rootfs / "usr" / "lib" / "aarch64-linux-gnu" / "ld-linux-aarch64.so.1",
            rootfs / "lib" / "aarch64-linux-gnu" / "ld-linux-aarch64.so.1",
            rootfs / "lib" / "ld-linux-aarch64.so.1",
        ]
    )
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise RuntimeError(f"Dynamic linker not found inside {rootfs}")


def _build_rootfs_library_path(rootfs: Path) -> str:
    dirs: list[str] = []
    seen: set[str] = set()
    preferred = [
        rootfs / "app",
        rootfs / "usr" / "lib" / "x86_64-linux-gnu",
        rootfs / "usr" / "lib" / "aarch64-linux-gnu",
        rootfs / "lib" / "x86_64-linux-gnu",
        rootfs / "lib" / "aarch64-linux-gnu",
        rootfs / "usr" / "lib" / "swift" / "linux",
        rootfs / "usr" / "local" / "lib",
        rootfs / "usr" / "lib",
        rootfs / "lib",
    ]
    for p in preferred:
        if p.is_dir():
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                dirs.append(rp)
    # Include any extra subdirectories under usr/lib that carry shared libraries (e.g. liblapack, blas).
    for base in (rootfs / "usr" / "lib", rootfs / "usr" / "local" / "lib"):
        if not base.is_dir():
            continue
        for dp, _, fns in os.walk(base, followlinks=False):
            if any(".so" in fn for fn in fns):
                rp = str(Path(dp).resolve())
                if rp not in seen:
                    seen.add(rp)
                    dirs.append(rp)
    return ":".join(dirs)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _read_pid(pid_file: Path) -> int | None:
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        return None


def probe_server(host: str, port: int, timeout: float = 90.0) -> dict[str, Any]:
    """Run one real historical forecast query (ICON 2024-06-15 in the Alps) against the local server."""
    url = f"http://{host}:{port}/v1/forecast?{urllib.parse.urlencode(DEFAULT_PROBE_PARAMS)}"
    t0 = time.monotonic()
    data = _http_get_json(url, timeout=timeout)
    dt_s = time.monotonic() - t0
    item = data[0] if isinstance(data, list) else data
    hourly = item.get("hourly") or {}
    t2m = [v for v in (hourly.get("temperature_2m") or []) if v is not None]
    t850 = [v for v in (hourly.get("temperature_850hPa") or []) if v is not None]
    z850 = [v for v in (hourly.get("geopotential_height_850hPa") or []) if v is not None]
    if not t2m or not t850 or not z850:
        raise RuntimeError(f"Probe returned empty/null values from {url}: keys={list(hourly)}")
    return {
        "url": f"http://{host}:{port}/v1/forecast",
        "elapsed_s": round(dt_s, 2),
        "elevation_m": item.get("elevation"),
        "hours": len(t2m),
        "t2m_noon_c": t2m[min(12, len(t2m) - 1)],
        "t850_noon_c": t850[min(12, len(t850) - 1)],
        "z850_noon_m": z850[min(12, len(z850) - 1)],
    }


def _tail_log(log_file: Path, lines: int = 40) -> str:
    if not log_file.exists():
        return "(no server log)"
    txt = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(txt[-lines:])


def start_server(
    work_dir: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    cache_gb: int = 12,
    max_concurrent_fetches: int = 24,
    http_connections: int = 32,
    prefetch_look_ahead: int = 6,
    skip_probe: bool = False,
) -> dict[str, Any]:
    """Ensure local openmeteo-api is running against AWS Open Data S3 and passes a live historical probe."""
    work_dir.mkdir(parents=True, exist_ok=True)
    pid_file = work_dir / "openmeteo.pid"
    log_file = work_dir / "openmeteo.log"
    data_dir = work_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if _port_open(host, port):
        pid = _read_pid(pid_file)
        print(f"Local Open-Meteo server already listening on http://{host}:{port} (pid={pid or 'external'})", flush=True)
        info = {"status": "running", "pid": pid, "url": f"http://{host}:{port}/v1/forecast"}
        if not skip_probe:
            info["probe"] = probe_server(host, port)
            p = info["probe"]
            print(
                f"Historical S3 probe OK in {p['elapsed_s']}s: elev={p['elevation_m']} m, "
                f"ICON 2024-06-15 12Z t2m={p['t2m_noon_c']} C, t850={p['t850_noon_c']} C, z850={p['z850_noon_m']} m",
                flush=True,
            )
        return info

    if platform.system() != "Linux":
        raise SystemExit(
            f"Native non-Docker launcher requires Linux (detected {platform.system()}). "
            f"On macOS/Windows, start the official container instead:\n"
            f"  docker run -d --rm --name open-meteo -v open-meteo-data:/app/data \\\n"
            f"    -e REMOTE_DATA_DIRECTORY={REMOTE_DATA_DIR} -e CACHE_SIZE={cache_gb}GB \\\n"
            f"    -e REMOTE_DATA_MAX_CONCURRENT_FETCHES={max_concurrent_fetches} \\\n"
            f"    -e REMOTE_DATA_HTTP_CONNECTIONS={http_connections} \\\n"
            f"    -e REMOTE_DATA_PRELOAD_MAX_AGE_SECONDS=0 \\\n"
            f"    -p {host}:{port}:8080 ghcr.io/{IMAGE_REPO}:{IMAGE_TAG}"
        )

    safe_cache_gb = compute_safe_cache_gb(cache_gb, data_dir)
    if safe_cache_gb != cache_gb:
        print(f"Adjusted CACHE_SIZE from {cache_gb}GB to {safe_cache_gb}GB to keep disk headroom on {data_dir}", flush=True)

    env = os.environ.copy()
    env.update({
        "REMOTE_DATA_DIRECTORY": REMOTE_DATA_DIR,
        "DATA_DIRECTORY": str(data_dir.resolve()) + "/",
        "CACHE_FILE": str((data_dir / "cache.bin").resolve()),
        "CACHE_SIZE": f"{safe_cache_gb}GB",
        "REMOTE_DATA_MAX_CONCURRENT_FETCHES": str(max_concurrent_fetches),
        "REMOTE_DATA_HTTP_CONNECTIONS": str(http_connections),
        "REMOTE_DATA_PRELOAD_MAX_AGE_SECONDS": "0",
        "PREFETCH_LOOK_AHEAD": str(prefetch_look_ahead),
    })

    # Prefer a pre-installed system binary only if already present on PATH; otherwise unpack the official OCI image.
    sys_bin = shutil.which("openmeteo-api")
    if sys_bin:
        cmd = [sys_bin, "serve", "--env", "production", "--hostname", host, "--port", str(port)]
        cwd = str(work_dir)
        print(f"Starting installed {sys_bin} on http://{host}:{port}...", flush=True)
    else:
        rootfs = fetch_oci_rootfs(work_dir / "rootfs")
        linker = _find_rootfs_linker(rootfs)
        lib_path = _build_rootfs_library_path(rootfs)
        binary = rootfs / "app" / "openmeteo-api"
        icu_dir = rootfs / "usr" / "share" / "zoneinfo-icu" / "44" / "le"
        if icu_dir.is_dir():
            env["ICU_TIMEZONE_FILES_DIR"] = str(icu_dir.resolve())
        ecc_defs = rootfs / "usr" / "share" / "eccodes" / "definitions"
        if ecc_defs.is_dir():
            env["ECCODES_DEFINITION_PATH"] = str(ecc_defs.resolve())
        for cert in ("/etc/ssl/certs/ca-certificates.crt", str(rootfs / "etc" / "ssl" / "certs" / "ca-certificates.crt")):
            if Path(cert).is_file():
                env["SSL_CERT_FILE"] = cert
                break
        cmd = [
            str(linker),
            "--inhibit-cache",
            "--library-path",
            lib_path,
            str(binary),
            "serve",
            "--env",
            "production",
            "--hostname",
            host,
            "--port",
            str(port),
        ]
        cwd = str((rootfs / "app").resolve())
        print(f"Starting local openmeteo-api on http://{host}:{port} (CACHE_SIZE={safe_cache_gb}GB, S3={REMOTE_DATA_DIR})...", flush=True)

    # Raise soft open-file limit if permitted so concurrent S3 + cache I/O never runs out of FDs.
    def _preexec() -> None:
        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            target = min(65536, hard if hard != resource.RLIM_INFINITY else 65536)
            if soft < target:
                resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except Exception:
            pass

    log_fh = log_file.open("ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            preexec_fn=_preexec,
        )
    finally:
        log_fh.close()

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"openmeteo-api exited early with code {rc}.\n--- Server log ---\n{_tail_log(log_file)}")
        if _port_open(host, port):
            break
        time.sleep(0.25)
    else:
        raise RuntimeError(f"openmeteo-api did not open {host}:{port} within 30s.\n--- Server log ---\n{_tail_log(log_file)}")

    print(f"Server ready on http://{host}:{port}/v1/forecast (pid={proc.pid}, log={log_file})", flush=True)
    info: dict[str, Any] = {"status": "started", "pid": proc.pid, "url": f"http://{host}:{port}/v1/forecast", "cache_gb": safe_cache_gb}
    if not skip_probe:
        try:
            p = probe_server(host, port)
        except Exception as exc:
            raise RuntimeError(f"Historical S3 probe failed: {exc}\n--- Server log ---\n{_tail_log(log_file)}") from exc
        info["probe"] = p
        print(
            f"Historical S3 probe OK in {p['elapsed_s']}s: elev={p['elevation_m']} m, "
            f"ICON 2024-06-15 12Z t2m={p['t2m_noon_c']} C, t850={p['t850_noon_c']} C, z850={p['z850_noon_m']} m",
            flush=True,
        )
    return info


def stop_server(work_dir: Path) -> None:
    pid_file = work_dir / "openmeteo.pid"
    pid = _read_pid(pid_file)
    if pid is None:
        print("No managed local openmeteo-api process is running.", flush=True)
        pid_file.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
    except ProcessLookupError:
        pass
    pid_file.unlink(missing_ok=True)
    print(f"Stopped local openmeteo-api (pid={pid}).", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true", help="install (if needed), start local server, and probe S3 historical data")
    action.add_argument("--stop", action="store_true", help="stop the managed local server")
    action.add_argument("--status", action="store_true", help="show whether the local server is listening")
    action.add_argument("--probe", action="store_true", help="run a historical ICON query against the local server")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"bind host (default {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"bind port (default {DEFAULT_PORT})")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help=f"working/cache directory (default {DEFAULT_WORK_DIR})")
    parser.add_argument("--cache-gb", type=int, default=12, help="local LRU S3 block cache size in GB (default 12)")
    parser.add_argument("--concurrent-fetches", type=int, default=24, help="REMOTE_DATA_MAX_CONCURRENT_FETCHES (default 24)")
    parser.add_argument("--http-connections", type=int, default=32, help="REMOTE_DATA_HTTP_CONNECTIONS (default 32)")
    parser.add_argument("--skip-probe", action="store_true", help="skip the startup historical S3 probe")
    args = parser.parse_args()

    if args.stop:
        stop_server(args.work_dir)
        return
    if args.status:
        listening = _port_open(args.host, args.port)
        pid = _read_pid(args.work_dir / "openmeteo.pid")
        print(json.dumps({"listening": listening, "pid": pid, "url": f"http://{args.host}:{args.port}/v1/forecast"}, indent=2))
        if not listening:
            sys.exit(1)
        return
    if args.probe:
        print(json.dumps(probe_server(args.host, args.port), indent=2))
        return
    start_server(
        work_dir=args.work_dir,
        host=args.host,
        port=args.port,
        cache_gb=args.cache_gb,
        max_concurrent_fetches=args.concurrent_fetches,
        http_connections=args.http_connections,
        skip_probe=args.skip_probe,
    )


if __name__ == "__main__":
    main()
