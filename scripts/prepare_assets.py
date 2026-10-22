#!/usr/bin/env python3
"""Download only the pinned robot XML and meshes required by the ball scene."""

import argparse
import io
from pathlib import Path, PurePosixPath
import urllib.request
import xml.etree.ElementTree as ET
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVISION = "b02fdea4902c2b93b7d34a83a6f05da4ececb7d3"
SOURCE = "https://github.com/pollen-robotics/microduck_rl"
MODEL_PATH = "src/mjlab_microduck/robot/microduck"


def prepare_assets(accept_model_license: bool) -> None:
    if not accept_model_license:
        raise SystemExit(
            "Robot meshes are CC BY-NC-SA 4.0 (noncommercial). Read LICENSE-MODELS.md "
            "and rerun with --accept-model-license if you agree. "
            "No assets were downloaded."
        )
    target = PROJECT_ROOT / MODEL_PATH
    marker = target / ".revision"
    cache_identity = f"{REVISION}\n{target}\n"
    if marker.is_file() and marker.read_text() == cache_identity:
        print(f"Using cached robot assets in {target}")
        return
    url = f"https://codeload.github.com/pollen-robotics/microduck_rl/zip/{REVISION}"
    print(f"Downloading robot assets from {SOURCE} at {REVISION}")
    with urllib.request.urlopen(url, timeout=120) as response:
        archive_bytes = response.read()
    prefix = f"microduck_rl-{REVISION}/{MODEL_PATH}/"
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        robot_bytes = archive.read(prefix + "robot_groundcontact.xml")
        robot = ET.fromstring(robot_bytes)
        mesh_names = sorted({mesh.attrib["file"] for mesh in robot.findall(".//mesh[@file]")})
        files = {"robot_groundcontact.xml": robot_bytes}
        for name in mesh_names:
            relative = PurePosixPath("assets") / name
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe mesh path in upstream XML: {name}")
            files[str(relative)] = archive.read(prefix + str(relative))
    if robot.find(".//camera[@name='head_camera']") is None:
        for body in robot.iter("body"):
            site = body.find("site[@name='head_camera']")
            if site is not None:
                ET.SubElement(body, "camera", {
                    "name": "head_camera",
                    "pos": site.attrib["pos"],
                    "quat": site.attrib["quat"],
                })
                break
        else:
            raise ValueError("Pinned robot has no head_camera attachment site")
    robot.find("compiler").set("meshdir", str(target / "assets"))
    files["robot_groundcontact.xml"] = ET.tostring(robot, encoding="utf-8")
    for relative, content in files.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    marker.write_text(cache_identity)
    print(f"Prepared robot XML and {len(mesh_names)} meshes in {target}")
    print("Pollen Robotics; meshes: CC BY-NC-SA 4.0. See LICENSE-MODELS.md.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accept-model-license", action="store_true")
    args = parser.parse_args()
    prepare_assets(args.accept_model_license)


if __name__ == "__main__":
    main()