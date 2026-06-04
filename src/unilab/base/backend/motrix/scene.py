from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from unilab.terrains.terrain_generator import TerrainGeneratorCfg

if TYPE_CHECKING:
    from motrixsim import SceneModel
    from motrixsim.msd import Link, World


def _resolve_scene_fragment_path(fragment_file: str, model_file: Path) -> Path:
    path = Path(fragment_file)
    if path.is_absolute():
        return path
    if path.is_file():
        return path.resolve()
    return (model_file.parent / path).resolve()


def _extract_keyframes(fragment_file: Path) -> list[ET.Element]:
    """Return ``<keyframe>`` child elements declared inside ``fragment_file``."""
    root = ET.parse(fragment_file).getroot()
    return list(root.findall("keyframe"))


def _materialize_robot_with_fragment_keyframes(
    robot_path: Path, fragment_paths: Sequence[Path]
) -> Path:
    """Inject fragment ``<keyframe>`` blocks into a temporary copy of ``robot_path``.

    motrix's ``msd.from_file`` validates ``<keyframe>`` qpos against the loaded
    model. fragment XMLs only carry sensors/contacts (no body), so a fragment
    with its own keyframe fails to parse on its own. Mujoco backend already
    merges fragments into the scene XML before parsing; this helper does the
    equivalent for the keyframe block so motrix can load a robot model that
    owns the keyframe declared in a sibling fragment.

    Returns the original ``robot_path`` when no fragment has a keyframe.
    """
    fragment_keyframes: list[ET.Element] = []
    for fragment_path in fragment_paths:
        fragment_keyframes.extend(_extract_keyframes(fragment_path))
    if not fragment_keyframes:
        return robot_path

    tree = ET.parse(robot_path)
    root = tree.getroot()
    existing = root.find("keyframe")
    if existing is None:
        existing = ET.SubElement(root, "keyframe")
    for keyframe in fragment_keyframes:
        existing.extend(list(keyframe))

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{robot_path.name}",
        dir=str(robot_path.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _materialize_fragment_without_keyframes(fragment_file: Path) -> Path:
    """Strip ``<keyframe>`` from a fragment XML; return original if no change."""
    tree = ET.parse(fragment_file)
    root = tree.getroot()
    keyframes = root.findall("keyframe")
    if not keyframes:
        return fragment_file
    for keyframe in keyframes:
        root.remove(keyframe)
    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{fragment_file.name}",
        dir=str(fragment_file.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _cleanup_temp_xml(path: Path, original: Path) -> None:
    if path == original:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _normalize_motrix_site_types(tree: ET.ElementTree) -> bool:
    changed = False
    for site in tree.getroot().iter("site"):
        if site.get("type") != "ellipsoid":
            continue
        site.set("type", "sphere")
        size = site.get("size")
        if size:
            values = [float(item) for item in size.split()]
            if values:
                site.set("size", f"{max(values):.9g}")
        changed = True
    return changed


def _body_contact_geom_names(root: ET.Element) -> dict[str, str]:
    names: dict[str, str] = {}

    def visit(body: ET.Element) -> None:
        body_name = body.get("name")
        if body_name:
            fallback: list[str] = []
            collision: list[str] = []
            for geom in body.findall("geom"):
                geom_name = geom.get("name")
                if not geom_name:
                    continue
                fallback.append(geom_name)
                contype = geom.get("contype")
                conaffinity = geom.get("conaffinity")
                if contype != "0" and conaffinity != "0":
                    collision.append(geom_name)
            if collision:
                names[body_name] = collision[0]
            elif fallback:
                names[body_name] = fallback[0]
        for child in body.findall("body"):
            visit(child)

    for body in root.findall("./worldbody/body"):
        visit(body)
    return names


def _normalize_motrix_contact_sensors(tree: ET.ElementTree) -> bool:
    root = tree.getroot()
    body_geom_names = _body_contact_geom_names(root)
    changed = False
    for contact in root.findall("./sensor/contact"):
        if contact.get("geom2") is not None:
            continue
        body2 = contact.get("body2")
        if body2 is None:
            continue
        geom2 = body_geom_names.get(body2)
        if geom2 is None:
            continue
        contact.set("geom2", geom2)
        del contact.attrib["body2"]
        changed = True
    return changed


def _materialize_motrix_compatible_robot(robot_path: Path, fragment_paths: Sequence[Path]) -> Path:
    fragment_keyframes: list[ET.Element] = []
    for fragment_path in fragment_paths:
        fragment_keyframes.extend(_extract_keyframes(fragment_path))

    tree = ET.parse(robot_path)
    changed = _normalize_motrix_site_types(tree)
    changed = _normalize_motrix_contact_sensors(tree) or changed
    if fragment_keyframes:
        existing = tree.getroot().find("keyframe")
        if existing is None:
            existing = ET.SubElement(tree.getroot(), "keyframe")
        for keyframe in fragment_keyframes:
            existing.extend(list(keyframe))
        changed = True

    if not changed:
        return robot_path

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{robot_path.name}",
        dir=str(robot_path.parent),
        mode="w",
        delete=False,
    )
    tmp.close()
    tree.write(tmp.name)
    return Path(tmp.name)


def _attach_motrix_scene_fragment(world: World, fragment_file: Path) -> None:
    import motrixsim.msd as msd

    sanitized = _materialize_fragment_without_keyframes(fragment_file)
    try:
        fragment = msd.from_file(str(sanitized))
    finally:
        _cleanup_temp_xml(sanitized, fragment_file)
    world.attach(fragment)


def _iter_motrix_links(link: Link):
    yield link
    for child in link.children:
        yield from _iter_motrix_links(child)


def _motrix_world_link_names(world: World) -> list[str]:
    names: list[str] = []
    for body in world.hierarchy.bodies:
        for link in _iter_motrix_links(body.link):
            if link.name:
                names.append(link.name)
    return names


def add_motrix_tracking_frame_sensors(world: World, *, base_name: str) -> None:
    """Add Motrix-native frame sensors matching the legacy tracking sensor contract."""
    import motrixsim.msd as msd

    link_names = _motrix_world_link_names(world)
    if base_name not in link_names:
        raise ValueError(f"Base link '{base_name}' not found in Motrix scene")

    existing = {sensor.name for sensor in world.sensors.frame if sensor.name}
    sensor_specs = (
        ("track_pos_b", msd.FrameSensorType.FramePos),
        ("track_quat_b", msd.FrameSensorType.FrameQuat),
        ("track_linvel_b", msd.FrameSensorType.FrameLinVel),
        ("track_angvel_b", msd.FrameSensorType.FrameAngVel),
    )
    ref_frame = msd.FrameSensorRef.object(msd.ObjectType.link(base_name))
    for link_name in link_names:
        object_type = msd.ObjectType.link(link_name)
        for prefix, sensor_type in sensor_specs:
            sensor_name = f"{prefix}_{link_name}"
            if sensor_name in existing:
                continue
            sensor = msd.FrameSensor()
            sensor.name = sensor_name
            sensor.sensor_type = sensor_type
            sensor.object_type = object_type
            sensor.ref_frame = ref_frame
            world.sensors.frame.append(sensor)


def materialize_motrix_scene(
    *,
    model_file: str,
    fragment_files: Sequence[str] = (),
    add_body_sensors: bool = False,
    base_name: str = "base",
) -> SceneModel:
    """Build a MotrixSim model through MSD scene composition."""
    import motrixsim.msd as msd

    model_path = Path(model_file).resolve()
    fragment_paths = [
        _resolve_scene_fragment_path(fragment_file, model_path) for fragment_file in fragment_files
    ]
    robot_path = _materialize_motrix_compatible_robot(model_path, fragment_paths)
    try:
        world = msd.from_file(str(robot_path))
        for fragment_path in fragment_paths:
            _attach_motrix_scene_fragment(world, fragment_path)
        if add_body_sensors:
            add_motrix_tracking_frame_sensors(world, base_name=base_name)
        return msd.build(world)
    finally:
        _cleanup_temp_xml(robot_path, model_path)


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[False] = False,
) -> tuple[SceneModel, np.ndarray]: ...


@overload
def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: Literal[True],
) -> tuple[SceneModel, np.ndarray, object]: ...


def materialize_motrix_hfield_attached_scene(
    *,
    model_file: str,
    terrain_cfg: TerrainGeneratorCfg,
    fragment_files: Sequence[str] = (),
    hfield_name: str = "terrain_hfield",
    geom_name: str = "floor",
    add_body_sensors: bool = False,
    base_name: str = "base",
    return_surface_sampler: bool = False,
) -> tuple[SceneModel, np.ndarray] | tuple[SceneModel, np.ndarray, object]:
    """Build a MotrixSim model with generated hfield terrain and attached robot."""
    import motrixsim.msd as msd

    from unilab.terrains import TerrainGenerator

    robot_path = Path(model_file).resolve()
    generated = TerrainGenerator(terrain_cfg).generate()

    world = msd.World()
    world.name = "unilab materialized hfield scene"

    hfield = msd.HFieldSource()
    hfield.nrow = int(generated.heights_yx.shape[0])
    hfield.ncol = int(generated.heights_yx.shape[1])
    # MotrixSim's hfield source uses MuJoCo-style X/Y half extents.
    hfield.size = [float(generated.hfield_size[0]), float(generated.hfield_size[1])]
    hfield.height_scale = float(generated.height_extent)
    # MotrixSim buffers use compiled hfield row order: row 0 is the -Y side.
    hfield_data = np.ascontiguousarray(np.flipud(generated.heights_yx).astype(np.float32))
    hfield.source_type = msd.HFieldSourceType.buffer(
        hfield_data.reshape(-1),
        f"{hfield_name}_buffer",
    )
    world.assets.hfields[hfield_name] = hfield

    terrain_geom = msd.Geometry()
    terrain_geom.name = geom_name
    terrain_geom.shape = msd.ShapeType.HField
    terrain_geom.hfield = hfield_name
    terrain_geom.position = np.asarray(generated.geom_pos, dtype=np.float32)
    terrain_geom.collision_mask = msd.CollisionMask.collide_with_all()
    terrain_geom.physics_material.friction = [1.0, 0.005, 0.0001]
    world.hierarchy.geoms.append(terrain_geom)

    fragment_paths = [
        _resolve_scene_fragment_path(fragment_file, robot_path) for fragment_file in fragment_files
    ]
    merged_robot_path = _materialize_robot_with_fragment_keyframes(robot_path, fragment_paths)
    try:
        robot_world = msd.from_file(str(merged_robot_path))
        world.attach(robot_world)
        # TODO(motrixsim): remove this once msd.World.attach carries keyframes.
        world.keyframes.extend(robot_world.keyframes)
    finally:
        _cleanup_temp_xml(merged_robot_path, robot_path)

    for fragment_path in fragment_paths:
        _attach_motrix_scene_fragment(world, fragment_path)
    if add_body_sensors:
        add_motrix_tracking_frame_sensors(world, base_name=base_name)

    if return_surface_sampler:
        return msd.build(world), generated.terrain_origins, generated.surface_sampler()
    return msd.build(world), generated.terrain_origins
