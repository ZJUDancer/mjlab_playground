"""Booster K1 constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

K1_XML: Path = Path(__file__).parent / "xmls" / "robot.xml"
assert K1_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(K1_XML))
  # The XML defines three control channels per joint. mjlab adds one position
  # actuator per joint below, so remove the XML channels from this in-memory spec.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec

# PD gains follow the dancer-robocupdemo-release
# armature and effort_limit follow XML

K1_ACTUATOR_NECK = BuiltinPositionActuatorCfg(
  target_names_expr=("AAHead_yaw", "Head_pitch"),
  stiffness=15.0,
  damping=1.0,
  effort_limit=6.0,
  armature=0.02,
)

K1_ACTUATOR_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_Shoulder_Pitch",
    ".*_Shoulder_Roll",
    ".*_Elbow_Pitch",
    ".*_Elbow_Yaw",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=14.0,
  armature=0.01,
)

K1_ACTUATOR_HIP_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Pitch",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=30.0,
  armature=0.0478125,
)

K1_ACTUATOR_HIP_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Roll",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=35.0,
  armature=0.0339552,
)

K1_ACTUATOR_HIP_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Hip_Yaw",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=20.0,
  armature=0.0282528,
)

K1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Knee_Pitch",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=40.0,
  armature=0.095625,
)

K1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_Ankle_Pitch", ".*_Ankle_Roll"),
  stiffness=50.0,
  damping=1.0,
  effort_limit=20.0,
  armature=0.0565,
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.55),
  joint_pos={
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    ".*_Shoulder_Pitch": 0.2,
    "Left_Shoulder_Roll": -1.25,
    "Right_Shoulder_Roll": 1.25,
    ".*_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.5,
    "Right_Elbow_Yaw": 0.5,
    ".*_Hip_Pitch": -0.15,
    ".*_Hip_Roll": 0.0,
    ".*_Hip_Yaw": 0.0,
    ".*_Knee_Pitch": 0.3,
    ".*_Ankle_Pitch": -0.15,
    ".*_Ankle_Roll": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^(left|right)_foot$"
_collision_regex = r"^(torso|pelvis|neck|head|left_.*|right_.*)$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  solref=(0.01, 1),
  condim={_foot_regex: 6, _collision_regex: 3},
  friction={_foot_regex: (1, 5e-3, 5e-4), _collision_regex: (0.6,)},
  priority=1,
)

##
# Final config.
##

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    K1_ACTUATOR_NECK,
    K1_ACTUATOR_ARM,
    K1_ACTUATOR_HIP_PITCH,
    K1_ACTUATOR_HIP_ROLL,
    K1_ACTUATOR_HIP_YAW,
    K1_ACTUATOR_KNEE,
    K1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh K1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    K1_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_k1_robot_cfg())

  viewer.launch(robot.spec.compile())
