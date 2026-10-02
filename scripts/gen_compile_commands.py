#!/usr/bin/env python3
"""Regenerate compile_commands.json for clangd (no full NCCL rebuild required)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BUILD = ROOT / "build"


def read_version() -> tuple[int, int, int, str]:
    major = minor = patch = 0
    suffix = ""
    for line in (ROOT / "makefiles" / "version.mk").read_text().splitlines():
        line = line.strip()
        if line.startswith("NCCL_MAJOR"):
            major = int(line.split(":=")[-1].strip() or line.split("=")[-1].strip())
        elif line.startswith("NCCL_MINOR"):
            minor = int(line.split(":=")[-1].strip() or line.split("=")[-1].strip())
        elif line.startswith("NCCL_PATCH"):
            patch = int(line.split(":=")[-1].strip() or line.split("=")[-1].strip())
        elif line.startswith("NCCL_SUFFIX"):
            suffix = line.split(":=")[-1].strip() or line.split("=")[-1].strip()
    return major, minor, patch, suffix


def find_cuda_include() -> Path:
    candidates = [
        Path("/usr/local/cuda/include"),
        Path("/usr/local/cuda-12.8/include"),
        Path("/usr/lib/cuda/include"),
        Path("/usr/include"),
    ]
    for c in candidates:
        if (c / "cuda_runtime.h").exists():
            return c
    return Path("/usr/local/cuda/include")


def ensure_headers() -> None:
    major, minor, patch, suffix = read_version()
    version = f"{major}{minor:02d}{patch:02d}"
    inc = BUILD / "include"
    obj_inc = BUILD / "obj" / "include"
    inc.mkdir(parents=True, exist_ok=True)
    obj_inc.mkdir(parents=True, exist_ok=True)

    text = (SRC / "nccl.h.in").read_text()
    text = (
        text.replace("${nccl:Major}", str(major))
        .replace("${nccl:Minor}", str(minor))
        .replace("${nccl:Patch}", str(patch))
        .replace("${nccl:Suffix}", suffix)
        .replace("${nccl:Version}", version)
    )
    (inc / "nccl.h").write_text(text)

    device_h = SRC / "include" / "nccl_device.h"
    if device_h.exists():
        (inc / "nccl_device.h").write_text(device_h.read_text())

    src_dev = SRC / "include" / "nccl_device"
    dst_dev = inc / "nccl_device"
    if src_dev.is_dir():
        for path in src_dev.rglob("*"):
            if path.is_file():
                rel = path.relative_to(src_dev)
                out = dst_dev / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(path.read_text(errors="ignore"))

    git_h = obj_inc / "nccl_git_version.h"
    if not git_h.exists():
        git_h.write_text('#pragma once\n#define NCCL_GIT_VERSION "dev"\n')


def main() -> None:
    ensure_headers()
    cuda_inc = find_cuda_include()
    # Detect CUDA major/minor from headers when possible
    cuda_major, cuda_minor = 12, 8
    rt = cuda_inc / "cuda_runtime_api.h"
    if rt.exists():
        for line in rt.read_text(errors="ignore").splitlines():
            if "CUDART_VERSION" in line and line.strip().startswith("#define"):
                # e.g. #define CUDART_VERSION  12080
                parts = line.split()
                if len(parts) >= 3 and parts[2].isdigit():
                    ver = int(parts[2])
                    cuda_major, cuda_minor = ver // 1000, (ver % 1000) // 10
                break

    flags = [
        "-x",
        "c++",
        "-std=c++17",
        "-fPIC",
        "-fvisibility=hidden",
        "-Wall",
        "-Wno-unused-function",
        "-Wno-sign-compare",
        "-Wvla",
        f"-DCUDA_MAJOR={cuda_major}",
        f"-DCUDA_MINOR={cuda_minor}",
        "-DNCCL_OS_LINUX=1",
        f"-I{SRC}",
        f"-I{BUILD / 'include'}",
        f"-I{BUILD / 'obj' / 'include'}",
        f"-I{SRC / 'include'}",
        f"-I{SRC / 'include' / 'plugin'}",
        f"-I{SRC / 'transport' / 'net_ib' / 'gdaki' / 'doca-gpunetio' / 'include'}",
        f"-I{cuda_inc}",
        f"-I{cuda_inc / 'cccl'}",
    ]

    files = sorted({*SRC.rglob("*.cc"), *SRC.rglob("*.cpp")})
    entries = [
        {
            "directory": str(SRC),
            "file": str(f),
            "arguments": ["clang++"] + flags + ["-c", str(f)],
        }
        for f in files
    ]

    # Also index key device headers via a TU that includes them through common.cu-like flags.
    # Map .h/.cuh under device/ to the nearest .cc/.cu companion when possible; otherwise
    # attach device headers to src/device/common.cu if present, else enqueue.cc.
    device_anchor = SRC / "device" / "common.cu"
    if not device_anchor.exists():
        device_anchor = SRC / "enqueue" / "enqueue.cc"
    for f in sorted(SRC.rglob("*.cu")):
        entries.append(
            {
                "directory": str(SRC),
                "file": str(f),
                "arguments": ["clang++"] + flags + ["-c", str(f)],
            }
        )

    out = ROOT / "compile_commands.json"
    out.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {out} ({len(entries)} entries)")
    print(f"CUDA include: {cuda_inc} (CUDART {cuda_major}.{cuda_minor})")


if __name__ == "__main__":
    main()
