"""Show the K1 locomotion default pose in a Viser browser viewer.

Run: .venv/bin/python src/mjlab_playground/asset_zoo/robots/booster_k1/view_pose_viser.py --port 8080
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

XML = Path(__file__).resolve().parent / "xmls/robot.xml"

# dancer-robocupdemo-release/models/k1_runtime_config.json: loco.default_dof_pos_22
# All joints omitted here are zero. Angles are in radians.
JOINT_POS = {
  "ALeft_Shoulder_Pitch": 0.2,
  "Left_Shoulder_Roll": -1.25,
  "Left_Elbow_Yaw": -0.5,
  "ARight_Shoulder_Pitch": 0.2,
  "Right_Shoulder_Roll": 1.25,
  "Right_Elbow_Yaw": 0.5,
  "Left_Hip_Pitch": -0.15,
  "Left_Knee_Pitch": 0.3,
  "Left_Ankle_Pitch": -0.15,
  "Right_Hip_Pitch": -0.15,
  "Right_Knee_Pitch": 0.3,
  "Right_Ankle_Pitch": -0.15,
}


def build_pose() -> tuple[mujoco.MjModel, mujoco.MjData]:
  model = mujoco.MjModel.from_xml_path(str(XML))
  data = mujoco.MjData(model)
  for name, angle in JOINT_POS.items():
    data.qpos[model.joint(name).qposadr] = angle
  mujoco.mj_forward(model, data)

  # Place the lowest foot collision box face on the z=0 floor.
  foot_bottoms = []
  for name in ("left_foot", "right_foot"):
    geom_id = model.geom(name).id
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    half_height = np.abs(rotation[2]) @ model.geom_size[geom_id]
    foot_bottoms.append(data.geom_xpos[geom_id, 2] - half_height)
  data.qpos[2] -= min(foot_bottoms)
  mujoco.mj_forward(model, data)
  return model, data


def visual_mesh(model: mujoco.MjModel, data: mujoco.MjData) -> trimesh.Trimesh:
  parts = []
  for geom_id in range(model.ngeom):
    if model.geom_group[geom_id] != 2 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    vertices = np.asarray(
      model.mesh_vert[vert_start : vert_start + model.mesh_vertnum[mesh_id]],
      dtype=np.float64,
    )
    faces = np.asarray(
      model.mesh_face[face_start : face_start + model.mesh_facenum[mesh_id]],
      dtype=np.int32,
    )
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
    part = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    material_id = int(model.geom_matid[geom_id])
    rgba = model.mat_rgba[material_id] if material_id >= 0 else model.geom_rgba[geom_id]
    part.visual.vertex_colors = np.tile(
      np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8),
      (len(vertices), 1),
    )
    parts.append(part)
  return trimesh.util.concatenate(parts)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  args = parser.parse_args()

  model, data = build_pose()
  server = viser.ViserServer(host=args.host, port=args.port, label="K1 default pose")
  server.scene.set_up_direction("+z")
  server.scene.add_grid(
    "/ground",
    width=2.0,
    height=2.0,
    cell_size=0.1,
    section_size=0.5,
    plane="xy",
  )
  server.scene.add_mesh_trimesh("/k1", visual_mesh(model, data))
  server.gui.add_markdown(
    "**K1 默认关节角** · `loco.default_dof_pos_22`（rad）\n\n"
    "肩俯仰 `0.2/0.2` · 肩横滚 `−1.25/1.25` · 肘偏航 `−0.5/0.5`\n\n"
    "双腿髋俯仰 `−0.15` · 膝 `0.3` · 踝俯仰 `−0.15`；其余关节 `0`。\n\n"
    f"根部高度 `{data.qpos[2]:.3f} m`，由脚底碰撞盒贴地计算。"
  )

  @server.on_client_connect
  def set_camera(client: viser.ClientHandle) -> None:
    client.camera.position = (1.3, -1.6, 0.95)
    client.camera.look_at = (0.0, 0.0, 0.48)

  print(f"K1 Viser: http://localhost:{args.port} (root z={data.qpos[2]:.3f} m)", flush=True)
  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass


if __name__ == "__main__":
  main()
