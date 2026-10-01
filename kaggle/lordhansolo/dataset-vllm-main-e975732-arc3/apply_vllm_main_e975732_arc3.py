#!/usr/bin/env python3
"""Install the hash-pinned ARC3 exact C14 graph overlay on the e975732 runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath


IDENTITY_FILE = "PATCH_IDENTITY.json"


class PatchError(RuntimeError):
    """The target runtime or shipped overlay does not match its pin."""


def sha256(data: bytes) -> str:

    return hashlib.sha256(data).hexdigest()


def load_identity() -> dict:
    path = Path(__file__).resolve().with_name(IDENTITY_FILE)

    return json.loads(path.read_text())


def checked_relative_path(value: str) -> Path:
    posix = PurePosixPath(value)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise PatchError(f"unsafe overlay target: {value!r}")

    return Path(*posix.parts)


def load_overlay(identity: dict) -> dict[str, bytes]:
    bundle_dir = Path(__file__).resolve().parent
    artifact = bundle_dir / identity["artifact"]
    artifact_bytes = artifact.read_bytes()
    actual_artifact_hash = sha256(artifact_bytes)
    if actual_artifact_hash != identity["overlay_sha256"]:
        raise PatchError(
            f"overlay has sha256 {actual_artifact_hash}, "
            f"expected {identity['overlay_sha256']}"
        )

    expected = {entry["target"]: entry for entry in identity["overlay_files"]}
    payloads: dict[str, bytes] = {}
    with tarfile.open(artifact, "r:") as archive:
        for member in archive:
            if not member.isfile():
                raise PatchError(f"overlay contains non-file member {member.name!r}")
            checked_relative_path(member.name)
            if member.name not in expected:
                raise PatchError(f"overlay contains unexpected member {member.name!r}")
            if member.name in payloads:
                raise PatchError(f"overlay contains duplicate member {member.name!r}")
            stream = archive.extractfile(member)
            if stream is None:
                raise PatchError(f"cannot read overlay member {member.name!r}")
            payload = stream.read()
            expected_hash = expected[member.name]["sha256"]
            actual_hash = sha256(payload)
            if actual_hash != expected_hash:
                raise PatchError(
                    f"{member.name} has sha256 {actual_hash}, expected {expected_hash}"
                )
            payloads[member.name] = payload

    missing = sorted(set(expected) - set(payloads))
    if missing:
        raise PatchError(f"overlay is missing members: {missing}")

    return payloads


def atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.arc3-main-e975732-arc3.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)


def preflight(site_packages: Path, identity: dict) -> tuple[list[dict], int]:
    to_install: list[dict] = []
    already_patched = 0
    for entry in identity["overlay_files"]:
        target = site_packages / checked_relative_path(entry["target"])
        if target.exists():
            actual = sha256(target.read_bytes())
            if actual == entry["sha256"]:
                already_patched += 1
                continue
            if actual not in (
                entry["stock_sha256"],
                entry.get("rc1_sha256"),
                entry.get("rc2_sha256"),
                entry.get("rc3_sha256"),
                entry.get("gdn_sha256"),
            ):
                raise PatchError(
                    f"{entry['target']} has sha256 {actual}, expected pinned stock "
                    f"{entry['stock_sha256']}"
                )
        elif entry["stock_sha256"] is not None:
            raise PatchError(f"pinned stock target is missing: {entry['target']}")
        to_install.append(entry)

    return to_install, already_patched


def install(site_packages: Path, identity: dict, payloads: dict[str, bytes]) -> None:
    to_install, already_patched = preflight(site_packages, identity)
    for entry in to_install:
        target = site_packages / checked_relative_path(entry["target"])
        atomic_write(target, payloads[entry["target"]])
    print(
        f"ARC3 main-e975732 overlay installed={len(to_install)}, "
        f"already-patched={already_patched}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "site_packages",
        type=Path,
        nargs="?",
        help="directory that directly contains the vllm package",
    )
    parser.add_argument("--print-generated-sha256", action="store_true")
    parser.add_argument("--stock-file", type=Path)

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    identity = load_identity()
    payloads = load_overlay(identity)
    if arguments.print_generated_sha256:
        if arguments.stock_file is None:
            raise SystemExit("--stock-file is required with --print-generated-sha256")
        stock_hash = sha256(arguments.stock_file.read_bytes())
        if stock_hash != identity["stock_target_sha256"]:
            raise PatchError(
                f"stock file has sha256 {stock_hash}, "
                f"expected {identity['stock_target_sha256']}"
            )
        primary = payloads[identity["target"]]
        generated_hash = sha256(primary)
        if generated_hash != identity["patched_target_sha256"]:
            raise PatchError(
                f"primary payload has sha256 {generated_hash}, "
                f"expected {identity['patched_target_sha256']}"
            )
        print(generated_hash)

        return

    if arguments.site_packages is None:
        raise SystemExit("site_packages is required")
    install(arguments.site_packages, identity, payloads)


if __name__ == "__main__":
    main()
