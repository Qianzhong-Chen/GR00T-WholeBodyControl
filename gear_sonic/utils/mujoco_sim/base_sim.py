"""MuJoCo simulation environment and loop for the G1 (and H1) humanoid robots.

DefaultEnv owns the MuJoCo model/data, computes PD torques from Unitree SDK
commands, steps physics, and publishes observations back via the SDK bridge.
BaseSimulator wraps DefaultEnv with rate-limiting and viewer/image update loops.
"""

import json
import os
import pathlib
from pathlib import Path
import pickle
import tempfile
from threading import Lock, Thread
import time
from typing import Dict
import xml.etree.ElementTree as ET

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from scipy.spatial.transform import Rotation
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from gear_sonic.utils.mujoco_sim.metric_utils import check_contact, check_height
from gear_sonic.utils.mujoco_sim.sim_utils import get_subtree_body_names
from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import ElasticBand, UnitreeSdk2Bridge
from gear_sonic.utils.mujoco_sim.robot import Robot

from gear_sonic.utils.mujoco_sim.object_pose_publisher import ObjectPosePublisher

GEAR_SONIC_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class DefaultEnv:
    """Base environment class that handles simulation environment setup and step"""

    def __init__(
        self,
        config: Dict[str, any],
        env_name: str = "default",
        camera_configs: Dict[str, any] = {},
        onscreen: bool = False,
        offscreen: bool = False,
        enable_image_publish: bool = False,
    ):
        self.config = config
        self.env_name = env_name
        self.robot = Robot(self.config)
        self.num_body_dof = self.robot.NUM_JOINTS
        self.num_hand_dof = self.robot.NUM_HAND_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.obs = None
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)
        self.camera_configs = camera_configs

        self.reward_lock = Lock()
        self.unitree_bridge = None
        self.onscreen = onscreen

        self.init_scene()
        self.last_reward = 0

        self.offscreen = offscreen
        if self.offscreen:
            self.init_renderers()
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.image_publish_process = None

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        from gear_sonic.utils.mujoco_sim.image_publish_utils import ImagePublishProcess

        if len(self.camera_configs) == 0:
            print(
                "Warning: No camera configs provided, image publishing subprocess will not be started"
            )
            return
        start_method = self.config.get("MP_START_METHOD", "spawn")
        self.image_publish_process = ImagePublishProcess(
            camera_configs=self.camera_configs,
            image_dt=self.image_dt,
            zmq_port=camera_port,
            start_method=start_method,
            verbose=self.config.get("verbose", False),
        )
        self.image_publish_process.start_process()

    def _get_dof_indices_by_class(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            joint_class_map = {}
            for joint_element in root.findall(".//joint[@class]"):
                joint_name = joint_element.get("name")
                joint_class = joint_element.get("class")
                if joint_name and joint_class:
                    joint_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                    )
                    if joint_id != -1:
                        dof_adr = self.mj_model.jnt_dofadr[joint_id]
                        if joint_class not in joint_class_map:
                            joint_class_map[joint_class] = []
                        joint_class_map[joint_class].append(dof_adr)
        finally:
            os.remove(temp_xml_path)

        return joint_class_map

    def _get_default_dof_properties(self):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".xml") as f:
            mujoco.mj_saveLastXML(f.name, self.mj_model)
            temp_xml_path = f.name

        try:
            tree = ET.parse(temp_xml_path)
            root = tree.getroot()

            default_dof_properties = {}
            for default_element in root.findall(".//default/default[@class]"):
                class_name = default_element.get("class")
                joint_element = default_element.find("joint")
                if class_name and joint_element is not None:
                    properties = {}
                    if "damping" in joint_element.attrib:
                        properties["damping"] = float(joint_element.get("damping"))
                    if "armature" in joint_element.attrib:
                        properties["armature"] = float(joint_element.get("armature"))
                    if "frictionloss" in joint_element.attrib:
                        properties["frictionloss"] = float(joint_element.get("frictionloss"))

                    if properties:
                        default_dof_properties[class_name] = properties
        finally:
            os.remove(temp_xml_path)

        return default_dof_properties

    def init_scene(self):
        """Initialize the default robot scene"""
        xml_path = str(pathlib.Path(GEAR_SONIC_ROOT) / self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt
        self.torso_index = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_body = "pelvis"
        self.root_body_id = self.mj_model.body(self.root_body).id
        self._load_randomization_spec(xml_path)

        self.joint_class_map = self._get_dof_indices_by_class()

        self.perform_sysid_search = self.config.get("perform_sysid_search", False)

        # Check for static root link (fixed base)
        self.use_floating_root_link = "floating_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        self.use_constrained_root_link = "constrained_base_joint" in [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]

        # MuJoCo qpos/qvel arrays start with root DOFs before joint DOFs:
        # floating base has 7 qpos (pos + quat) and 6 qvel (lin + ang velocity)
        if self.use_floating_root_link:
            self.qpos_offset = 7
            self.qvel_offset = 6
        else:
            if self.use_constrained_root_link:
                self.qpos_offset = 1
                self.qvel_offset = 1
            else:
                raise ValueError(
                    "No root link found --"
                    "The absolute static root will make the simulation unstable."
                )

        # Enable the elastic band. For a constrained (fixed) base there is no
        # floating root to suspend, so the band stays None and sim_step skips it.
        self.elastic_band = None
        if self.config["ENABLE_ELASTIC_BAND"] and self.use_floating_root_link:
            self.elastic_band = ElasticBand()
            if "g1" in self.config["ROBOT_TYPE"]:
                if self.config["enable_waist"]:
                    self.band_attached_link = self.mj_model.body("pelvis").id
                else:
                    self.band_attached_link = self.mj_model.body("torso_link").id
            elif "h1" in self.config["ROBOT_TYPE"]:
                self.band_attached_link = self.mj_model.body("torso_link").id
            else:
                self.band_attached_link = self.mj_model.body("base_link").id

            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model,
                    self.mj_data,
                    key_callback=self.elastic_band.MujuocoKeyCallback,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None
        else:
            if self.onscreen:
                self.viewer = mujoco.viewer.launch_passive(
                    self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False
                )
            else:
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self.viewer = None

        if self.viewer:
            self.viewer.cam.azimuth = 120
            self.viewer.cam.elevation = -30
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat = np.array([0, 0, 0.5])
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = self.mj_model.body("pelvis").id

        self.body_joint_index = []
        self.left_hand_index = []
        self.right_hand_index = []
        for i in range(self.mj_model.njnt):
            name = self.mj_model.joint(i).name
            if any(
                [
                    part_name in name
                    for part_name in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"]
                ]
            ):
                self.body_joint_index.append(i)
            elif "left_hand" in name:
                self.left_hand_index.append(i)
            elif "right_hand" in name:
                self.right_hand_index.append(i)

        assert len(self.body_joint_index) == self.robot.NUM_JOINTS
        assert len(self.left_hand_index) == self.robot.NUM_HAND_JOINTS
        assert len(self.right_hand_index) == self.robot.NUM_HAND_JOINTS

        self.body_joint_index = np.array(self.body_joint_index)
        self.left_hand_index = np.array(self.left_hand_index)
        self.right_hand_index = np.array(self.right_hand_index)

        if self.use_constrained_root_link:
            self._disable_robot_static_collision()
            self._setup_lower_body_freeze()

    def _setup_lower_body_freeze(self):
        """For the fixed (constrained) base, the legs + waist are cosmetic — the
        robot is a rigidly-mounted arm manipulator. Without a balancing policy
        they sag under gravity (and PD can't reliably recover a collapsed pose,
        nor before Shell C connects). So we HARD-FREEZE the lower-body joints at
        their standing defaults every sim step (qpos held, qvel zeroed), leaving
        only the arms + hands dynamic. Builds the qpos/qvel address lists once.
        """
        default = self.config.get("DEFAULT_DOF_ANGLES") or []
        # Lower body = leg + waist joints (exclude arms/wrists, which the direct
        # planner drives). Match by name; keep DEFAULT_DOF_ANGLES order alignment.
        self._frozen_qadr, self._frozen_vadr, self._frozen_q = [], [], []
        lower_parts = ["hip", "knee", "ankle", "waist"]
        for k, jid in enumerate(self.body_joint_index):
            name = self.mj_model.joint(jid).name
            if any(p in name for p in lower_parts) and k < len(default):
                self._frozen_qadr.append(self.mj_model.jnt_qposadr[jid])
                self._frozen_vadr.append(self.mj_model.jnt_dofadr[jid])
                self._frozen_q.append(float(default[k]))
        self._frozen_qadr = np.array(self._frozen_qadr, dtype=int)
        self._frozen_vadr = np.array(self._frozen_vadr, dtype=int)
        self._frozen_q = np.array(self._frozen_q, dtype=float)
        print(f"[base_sim] constrained base: hard-freezing {len(self._frozen_q)} "
              f"lower-body joints (legs+waist) at standing default")

    def _disable_robot_static_collision(self):
        """For a constrained (fixed) base, the rigidly-mounted robot stands within
        arm's reach of the table, so its lower body overlaps the table footprint
        (the floating-base WBC avoids this by squatting/leaning). With the base
        pinned we don't need leg↔table contact for stability, and direct_manip's
        grasp is kinematic — so we disable ROBOT↔STATIC collisions via
        contype/conaffinity bitmasks while keeping object↔table and object↔hand.

        Bit scheme (collide iff (ctypeA&caffB)|(ctypeB&caffA)):
          robot  -> 0x1/0x1, static (table/marker) -> 0x2/0x2,
          object + floor -> 0x3/0x3 (collide with both). Only applied for the
          constrained-base model, so the SONIC floating-base sim is untouched.
        """
        m = self.mj_model
        pelvis_root = m.body(self.root_body).id
        robot_bodies = set(get_subtree_body_names(m, pelvis_root))
        ROBOT, STATIC, OBJ = 0x1, 0x2, 0x3
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue  # visual-only geom
            bid = m.geom_bodyid[g]
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            jnt_is_free = any(
                m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                for j in range(m.njnt) if m.jnt_bodyid[j] == bid
            )
            if bname in robot_bodies:
                m.geom_contype[g], m.geom_conaffinity[g] = ROBOT, ROBOT
            elif jnt_is_free or bname == "world" or m.geom_bodyid[g] == 0:
                # freejoint manipulables + the floor geom (body 0/world).
                m.geom_contype[g], m.geom_conaffinity[g] = OBJ, OBJ
            else:
                # statics (table, place_marker, lift_target): collide with object only.
                m.geom_contype[g], m.geom_conaffinity[g] = STATIC, STATIC
        print("[base_sim] constrained base: disabled robot↔static collision "
              "(robot=0x1, static=0x2, object/floor=0x3)")

    def init_renderers(self):
        self.renderers = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = mujoco.Renderer(
                self.mj_model, height=camera_config["height"], width=camera_config["width"]
            )
            self.renderers[camera_name] = renderer

    def compute_body_torques(self) -> np.ndarray:
        # PD control: tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
        body_torques = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                if self.unitree_bridge.use_sensor:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (self.unitree_bridge.low_cmd.motor_cmd[i].q - self.mj_data.sensordata[i])
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.sensordata[i + self.unitree_bridge.num_body_motor]
                        )
                    )
                else:
                    body_torques[i] = (
                        self.unitree_bridge.low_cmd.motor_cmd[i].tau
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kp
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].q
                            - self.mj_data.qpos[self.body_joint_index[i] + self.qpos_offset - 1]
                        )
                        + self.unitree_bridge.low_cmd.motor_cmd[i].kd
                        * (
                            self.unitree_bridge.low_cmd.motor_cmd[i].dq
                            - self.mj_data.qvel[self.body_joint_index[i] + self.qvel_offset - 1]
                        )
                    )
        return body_torques

    def get_head_pose(self) -> np.ndarray:
        root_pos = self.mj_data.body("torso_link").xpos.copy()
        # Reorder quaternion from MuJoCo [w,x,y,z] to scipy [x,y,z,w]
        root_quat = self.mj_data.body("torso_link").xquat.copy()[[1, 2, 3, 0]]
        head_pos = root_pos + Rotation.from_quat(root_quat).apply(np.array([0.0, 0.0, -0.044]))
        return np.concatenate((head_pos, root_quat))

    def get_root_vel(self) -> np.ndarray:
        return self.mj_data.qvel[:6]

    def compute_hand_torques(self) -> np.ndarray:
        left_hand_torques = np.zeros(self.num_hand_dof)
        right_hand_torques = np.zeros(self.num_hand_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                left_hand_torques[i] = (
                    self.unitree_bridge.left_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.left_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.left_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.left_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.left_hand_index[i] + self.qvel_offset - 1]
                    )
                )
                right_hand_torques[i] = (
                    self.unitree_bridge.right_hand_cmd.motor_cmd[i].tau
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kp
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
                        - self.mj_data.qpos[self.right_hand_index[i] + self.qpos_offset - 1]
                    )
                    + self.unitree_bridge.right_hand_cmd.motor_cmd[i].kd
                    * (
                        self.unitree_bridge.right_hand_cmd.motor_cmd[i].dq
                        - self.mj_data.qvel[self.right_hand_index[i] + self.qvel_offset - 1]
                    )
                )
        return np.concatenate((left_hand_torques, right_hand_torques))

    def compute_body_qpos(self) -> np.ndarray:
        body_qpos = np.zeros(self.num_body_dof)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_body_motor):
                body_qpos[i] = self.unitree_bridge.low_cmd.motor_cmd[i].q
        return body_qpos

    def compute_hand_qpos(self) -> np.ndarray:
        hand_qpos = np.zeros(self.num_hand_dof * 2)
        if self.unitree_bridge is not None and self.unitree_bridge.low_cmd:
            for i in range(self.unitree_bridge.num_hand_motor):
                hand_qpos[i] = self.unitree_bridge.left_hand_cmd.motor_cmd[i].q
                hand_qpos[i + self.num_hand_dof] = self.unitree_bridge.right_hand_cmd.motor_cmd[i].q
        return hand_qpos

    def prepare_obs(self) -> Dict[str, any]:
        obs = {}
        if self.use_floating_root_link:
            obs["floating_base_pose"] = self.mj_data.qpos[:7]
            obs["floating_base_vel"] = self.mj_data.qvel[:6]
            obs["floating_base_acc"] = self.mj_data.qacc[:6]
        else:
            # Fixed/constrained base: there is no free root in qpos, but the base
            # still has a real WORLD pose (the pelvis body). Publish that so the
            # client's FK + recorder get a valid base transform; a zeros(7) pose
            # would carry a zero-norm quaternion and crash scipy on the client.
            root_id = self.root_body_id
            pose = np.zeros(7)
            pose[:3] = self.mj_data.xpos[root_id]
            pose[3:7] = self.mj_data.xquat[root_id]  # MuJoCo wxyz
            obs["floating_base_pose"] = pose
            obs["floating_base_vel"] = np.zeros(6)
            obs["floating_base_acc"] = np.zeros(6)

        obs["secondary_imu_quat"] = self.mj_data.xquat[self.torso_index]

        pose = np.zeros(13)
        torso_link = self.mj_model.body("torso_link").id
        # mj_objectVelocity returns [ang_vel, lin_vel]; swap to [lin_vel, ang_vel]
        mujoco.mj_objectVelocity(
            self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY, torso_link, pose[7:13], 1
        )
        pose[7:10], pose[10:13] = (
            pose[10:13],
            pose[7:10].copy(),
        )
        obs["secondary_imu_vel"] = pose[7:13]

        # Use qpos_offset/qvel_offset (7/6 for a floating base, 1/1 for the
        # constrained base) instead of hardcoding the floating-base 7/6 — a fixed
        # base has no 6-DOF free root, so the hardcoded +6 would read 6 joints too
        # high and scramble the published body state (garbled torso + arms).
        obs["body_q"] = self.mj_data.qpos[self.body_joint_index + self.qpos_offset - 1]
        obs["body_dq"] = self.mj_data.qvel[self.body_joint_index + self.qvel_offset - 1]
        obs["body_ddq"] = self.mj_data.qacc[self.body_joint_index + self.qvel_offset - 1]
        obs["body_tau_est"] = self.mj_data.actuator_force[self.body_joint_index - 1]
        if self.num_hand_dof > 0:
            obs["left_hand_q"] = self.mj_data.qpos[self.left_hand_index + self.qpos_offset - 1]
            obs["left_hand_dq"] = self.mj_data.qvel[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_ddq"] = self.mj_data.qacc[self.left_hand_index + self.qvel_offset - 1]
            obs["left_hand_tau_est"] = self.mj_data.actuator_force[self.left_hand_index - 1]
            obs["right_hand_q"] = self.mj_data.qpos[self.right_hand_index + self.qpos_offset - 1]
            obs["right_hand_dq"] = self.mj_data.qvel[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_ddq"] = self.mj_data.qacc[self.right_hand_index + self.qvel_offset - 1]
            obs["right_hand_tau_est"] = self.mj_data.actuator_force[self.right_hand_index - 1]
        obs["time"] = self.mj_data.time
        return obs

    def sim_step(self):
        self.obs = self.prepare_obs()
        self.unitree_bridge.PublishLowState(self.obs)
        if self.unitree_bridge.joystick:
            self.unitree_bridge.PublishWirelessController()
        if self.elastic_band:
            # Headless auto-release: a display build lets the operator press key 9
            # to drop the suspension spring AFTER the policy has settled and is
            # balancing. Headless has no viewer, so emulate that timing: hold the
            # band for BAND_HOLD_SEC of policy commands (let the WBC policy ramp up
            # and stabilize), then release once. Releasing immediately on the first
            # command drops the robot before the policy can catch it.
            import time as _t
            if getattr(self.unitree_bridge, "low_cmd_received", False):
                if not hasattr(self, "_band_cmd_t0"):
                    self._band_cmd_t0 = _t.time()
                elif self.elastic_band.enable and (_t.time() - self._band_cmd_t0) >= 8.0:
                    print("[ElasticBand] policy settled — auto-releasing band (headless)")
                    self.elastic_band.enable = False
            if self.elastic_band.enable and self.use_floating_root_link:
                pose = np.concatenate(
                    [
                        self.mj_data.xpos[self.band_attached_link],
                        self.mj_data.xquat[self.band_attached_link],
                        np.zeros(6),
                    ]
                )
                mujoco.mj_objectVelocity(
                    self.mj_model,
                    self.mj_data,
                    mujoco.mjtObj.mjOBJ_BODY,
                    self.band_attached_link,
                    pose[7:13],
                    0,
                )
                pose[7:10], pose[10:13] = pose[10:13], pose[7:10].copy()
                self.mj_data.xfrc_applied[self.band_attached_link] = self.elastic_band.Advance(pose)
            else:
                self.mj_data.xfrc_applied[self.band_attached_link] = np.zeros(6)
        body_torques = self.compute_body_torques()
        hand_torques = self.compute_hand_torques()
        # -1: actuator array is 0-based while joint indices from the model are 1-based
        self.torques[self.body_joint_index - 1] = body_torques
        if self.num_hand_dof > 0:
            self.torques[self.left_hand_index - 1] = hand_torques[: self.num_hand_dof]
            self.torques[self.right_hand_index - 1] = hand_torques[self.num_hand_dof :]

        self.torques = np.clip(self.torques, -self.torque_limit, self.torque_limit)

        if self.config["FREE_BASE"]:
            # Prepend 6 zeros for the floating-base root DOF actuators
            self.mj_data.ctrl = np.concatenate((np.zeros(6), self.torques))
        else:
            self.mj_data.ctrl = self.torques
        mujoco.mj_step(self.mj_model, self.mj_data)

        # Constrained base: hard-freeze the lower body at standing each step so
        # the legs/waist can't sag under gravity (no balancing policy here).
        if getattr(self, "_frozen_qadr", None) is not None and len(self._frozen_qadr):
            self.mj_data.qpos[self._frozen_qadr] = self._frozen_q
            self.mj_data.qvel[self._frozen_vadr] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

        self.check_fall()

    def apply_perturbation(self, key):
        perturbation_x_body = 0.0
        perturbation_y_body = 0.0
        if key == "up":
            perturbation_x_body = 1.0
        elif key == "down":
            perturbation_x_body = -1.0
        elif key == "left":
            perturbation_y_body = 1.0
        elif key == "right":
            perturbation_y_body = -1.0

        vel_body = np.array([perturbation_x_body, perturbation_y_body, 0.0])
        vel_world = np.zeros(3)
        base_quat = self.mj_data.qpos[3:7]
        mujoco.mju_rotVecQuat(vel_world, vel_body, base_quat)

        self.mj_data.qvel[0] += vel_world[0]
        self.mj_data.qvel[1] += vel_world[1]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def update_viewer(self):
        if self.viewer is not None:
            self.viewer.sync()

    def update_viewer_camera(self):
        if self.viewer is not None:
            if self.viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def update_reward(self):
        with self.reward_lock:
            self.last_reward = 0

    def get_reward(self):
        with self.reward_lock:
            return self.last_reward

    def set_unitree_bridge(self, unitree_bridge):
        self.unitree_bridge = unitree_bridge

    def get_privileged_obs(self):
        return {}

    def update_render_caches(self):
        render_caches = {}
        for camera_name, camera_config in self.camera_configs.items():
            renderer = self.renderers[camera_name]
            if "params" in camera_config:
                renderer.update_scene(self.mj_data, camera=camera_config["params"])
            else:
                renderer.update_scene(self.mj_data, camera=camera_name)
            render_caches[camera_name + "_image"] = renderer.render()

        if self.image_publish_process is not None:
            self.image_publish_process.update_shared_memory(render_caches)

        return render_caches

    def handle_keyboard_button(self, key):
        if self.elastic_band:
            self.elastic_band.handle_keyboard_button(key)

        if key == "backspace":
            self.reset()
        if key == "v":
            self.update_viewer_camera()
        if key in ["up", "down", "left", "right"]:
            self.apply_perturbation(key)

    def check_fall(self):
        self.fall = False
        # Only meaningful for a floating base, where qpos[2] is the pelvis world
        # height. For a constrained (fixed) base, qpos[0] is the tiny z-slide
        # value (~0), so this check would false-fire every step — skip it.
        if self.use_floating_root_link and self.mj_data.qpos[2] < 0.2:
            self.fall = True
            print(f"Warning: Robot has fallen, height: {self.mj_data.qpos[2]:.3f} m")

        if self.fall:
            self.reset()

    def check_self_collision(self):
        robot_bodies = get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        self_collision, contact_bodies = check_contact(
            self.mj_model, self.mj_data, robot_bodies, robot_bodies, return_all_contact_bodies=True
        )
        if self_collision:
            print(f"Warning: Self-collision detected: {contact_bodies}")
        return self_collision

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self._seed_default_pose()
        self._randomize_robot_pose()
        # Table-height jitter must run BEFORE object randomization so the dz it
        # records is available to shift resting objects onto the new surface.
        self._randomize_table_height()
        self._randomize_object_pose()

    def reset_object(self):
        """Re-randomize ONLY the table height + manipulable objects, leaving the
        robot exactly as it is (no mj_resetData, no robot re-seed).

        Used by the direct (fixed-base) staged reset: the robot/arm are reset and
        allowed to settle to the idle pose FIRST, then the bottle is respawned so
        it can't collide with a still-retracting arm. The manipulable freejoints
        are re-placed from their model spawn (body_pos / qpos0) + randomization.
        """
        # Reset each freejoint object to its spawn qpos0 before re-randomizing,
        # so jitter is applied from the canonical spawn, not the last grasp pose.
        for jid in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            qadr = self.mj_model.jnt_qposadr[jid]
            self.mj_data.qpos[qadr:qadr + 7] = self.mj_model.qpos0[qadr:qadr + 7]
            vadr = self.mj_model.jnt_dofadr[jid]
            self.mj_data.qvel[vadr:vadr + 6] = 0.0
        self._randomize_table_height()
        self._randomize_object_pose()
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _seed_default_pose(self):
        """Seed the standing default joint angles after mj_resetData.

        mj_resetData zeros every joint, leaving the legs straight. A floating
        base is fine (the WBC policy poses + balances the robot from there), but
        the constrained (fixed) base has no policy: its PD controller can't pull
        the legs from 0 up to a squat against floor contact, so the robot looks
        collapsed/curved. Seed DEFAULT_DOF_ANGLES so the legs START standing and
        the direct planner only has to HOLD them. Only for the constrained base
        so the SONIC floating-base reset is unchanged.
        """
        if self.use_floating_root_link:
            return
        default = self.config.get("DEFAULT_DOF_ANGLES")
        if not default:
            return
        # body_joint_index are model joint ids in DEFAULT_DOF order (leg→waist→
        # arm). Use jnt_qposadr directly so the mapping is exact regardless of
        # the root joint's dof count.
        for k, jid in enumerate(self.body_joint_index):
            if k >= len(default):
                break
            self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]] = float(default[k])
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _randomize_table_height(self):
        """Jitter the z of any static body that declared a `table_z_range` in
        the sidecar (key `statics`), and remember the dz so resting objects can
        be shifted onto the new surface by _randomize_object_pose.

        body_pos is model-level and persists across resets, so we re-base from
        the ORIGINAL baked z (captured once) each reset rather than accumulate.
        """
        self._table_dz = {}  # object_name -> dz to apply this reset
        statics = (self._random_spec or {}).get("statics") or {}
        if not statics:
            return
        if not hasattr(self, "_table_base_z"):
            self._table_base_z = {}  # body_name -> original baked body_pos[z]
        for name, spec in statics.items():
            try:
                bid = self.mj_model.body(name).id
            except Exception:
                print(f"[base_sim] table_z: body '{name}' not found, skipping")
                continue
            if name not in self._table_base_z:
                self._table_base_z[name] = float(self.mj_model.body_pos[bid][2])
            dz = self._uniform(spec.get("z", [0.0, 0.0]))
            self.mj_model.body_pos[bid][2] = self._table_base_z[name] + dz
            # Freejoint objects resting on this surface: shift their spawn qpos z
            # (applied in _randomize_object_pose via self._table_dz).
            for obj_name in spec.get("resting_objects", []):
                self._table_dz[obj_name] = dz
            # Static props resting on this surface (e.g. a place_marker): they are
            # body_pos-driven like the table, so shift their body_pos z directly,
            # re-basing from the original baked z each reset.
            for st_name in spec.get("resting_statics", []):
                try:
                    sbid = self.mj_model.body(st_name).id
                except Exception:
                    print(f"[base_sim] table_z: resting static '{st_name}' not found, skipping")
                    continue
                if st_name not in self._table_base_z:
                    self._table_base_z[st_name] = float(self.mj_model.body_pos[sbid][2])
                self.mj_model.body_pos[sbid][2] = self._table_base_z[st_name] + dz
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # Default spawn-pose jitter — used when scene_config.py did not write a
    # sidecar JSON next to the generated XML. Mirrors the original hard-coded
    # behavior so existing scenes keep working unchanged.
    _DEFAULT_RANDOMIZATION = {
        "robot": {
            "x":       [-0.05, 0.05],
            "y":       [-0.05, 0.05],
            "yaw_deg": [0.0, 0.0],
        },
        "objects": {
            "x":       [-0.05, 0.05],
            "y":       [-0.05, 0.05],
            "yaw_deg": [-180.0, 180.0],
        },
    }

    def _load_randomization_spec(self, xml_path):
        """Read the per-scene randomization sidecar JSON written by
        scripts/main/scene_config.py:write_scene_xml().

        Sidecar path is the XML path with `.xml` swapped for `_random.json`.
        Falls back to the legacy global defaults when the sidecar is missing,
        so scenes that have not been regenerated still randomize the same as
        before.
        """
        stem, _ = os.path.splitext(xml_path)
        sidecar = stem + "_random.json"
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r") as f:
                    self._random_spec = json.load(f)
                print(f"[base_sim] Loaded randomization spec from {sidecar}")
                return
            except Exception as e:
                print(f"[base_sim] Failed to read {sidecar}: {e}; using defaults")
        self._random_spec = dict(self._DEFAULT_RANDOMIZATION)

    @staticmethod
    def _uniform(range_pair):
        lo, hi = float(range_pair[0]), float(range_pair[1])
        if lo == hi:
            return lo
        return float(np.random.uniform(lo, hi))

    def _randomize_robot_pose(self):
        """Randomize the robot spawn xy after reset. Yaw is always 0.

        Spawn yaw is pinned because the deploy resets planner_facing_angle_
        to 0 on CONTROL_GOAL_TIMEOUT, so a yawed spawn would snap back to
        world +X. Yaw diversity for recording is applied post-spawn via a
        vyaw burst in the recording client (auto_recorder.py), which uses
        the per-scene `randomization.robot.yaw_deg` range as its source.
        """
        if not self.use_floating_root_link:
            return
        spec = self._random_spec.get("robot") or self._DEFAULT_RANDOMIZATION["robot"]
        self.mj_data.qpos[0] += self._uniform(spec.get("x", [0.0, 0.0]))
        self.mj_data.qpos[1] += self._uniform(spec.get("y", [0.0, 0.0]))

        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _randomize_object_pose(self):
        """Randomize xy and yaw of every manipulable (freejoint) body after reset.

        Per-object overrides come from the sidecar JSON (`objects[name]`):
            None  → pin the object (no jitter)
            dict  → use those ranges instead of the scene default.
        The robot's floating root is excluded via the robot-subtree check.
        """
        robot_subtree = set(
            get_subtree_body_names(self.mj_model, self.mj_model.body(self.root_body).id)
        )
        per_obj = self._random_spec.get("objects") or {}
        scene_default_obj = self._DEFAULT_RANDOMIZATION["objects"]

        for jid in range(self.mj_model.njnt):
            if self.mj_model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            body_id = self.mj_model.jnt_bodyid[jid]
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name in robot_subtree:
                continue

            qpos_adr = self.mj_model.jnt_qposadr[jid]

            # Table-height follow: if this object rests on a height-jittered
            # static, shift its spawn z by the same dz so it stays on the
            # surface. Applies even when the object is otherwise pinned.
            table_dz = getattr(self, "_table_dz", {}).get(name, 0.0)
            if table_dz:
                self.mj_data.qpos[qpos_adr + 2] += table_dz

            spec = per_obj.get(name, scene_default_obj)
            if spec is None:
                continue  # explicitly pinned (xy/yaw); z-follow above still applied

            self.mj_data.qpos[qpos_adr + 0] += self._uniform(spec.get("x", [0.0, 0.0]))
            self.mj_data.qpos[qpos_adr + 1] += self._uniform(spec.get("y", [0.0, 0.0]))

            yaw_deg = self._uniform(spec.get("yaw_deg", [0.0, 0.0]))
            if yaw_deg != 0.0:
                yaw = np.deg2rad(yaw_deg)
                yaw_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
                base_quat = self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7].copy()
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, yaw_quat, base_quat)
                self.mj_data.qpos[qpos_adr + 3 : qpos_adr + 7] = new_quat

        mujoco.mj_forward(self.mj_model, self.mj_data)


class BaseSimulator:
    """Base simulator class that handles initialization and running of simulations"""

    def __init__(
        self, config: Dict[str, any], env_name: str = "default", redis_client=None, **kwargs
    ):
        self.config = config
        self.env_name = env_name
        self.redis_client = redis_client
        if self.redis_client is not None:
            self.redis_client.set("push_left_hand", "false")
            self.redis_client.set("push_right_hand", "false")
            self.redis_client.set("push_torso", "false")

        # Create rate objects
        self.sim_dt = self.config["SIMULATE_DT"]
        self.reward_dt = self.config.get("REWARD_DT", 0.02)
        self.image_dt = self.config.get("IMAGE_DT", 0.033333)
        self.viewer_dt = self.config.get("VIEWER_DT", 0.02)
        self._running = True

        self.robot = Robot(self.config)

        # Create the environment
        if env_name == "default":
            self.sim_env = DefaultEnv(config, env_name, **kwargs)
        else:
            raise ValueError(
                f"Invalid environment name: {env_name}. "
                f"Only 'default' is supported in this minimal build."
            )

        try:
            if self.config.get("INTERFACE", None):
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        except Exception as e:
            print(f"Note: Channel factory initialization attempt: {e}")

        self.init_unitree_bridge()
        self.sim_env.set_unitree_bridge(self.unitree_bridge)

        self.init_subscriber()
        self.init_publisher()

        self.sim_thread = None

    def start_as_thread(self):
        self.sim_thread = Thread(target=self.start)
        self.sim_thread.start()

    def start_image_publish_subprocess(self, start_method: str = "spawn", camera_port: int = 5555):
        self.sim_env.start_image_publish_subprocess(start_method, camera_port)

    def init_subscriber(self):
        pass

    def init_publisher(self):
        """Publish one pose topic per non-robot body (statics + manipulables).

        Topic format: /simulation/pose/{body_name}. The first publisher owns
        the /simulation/reset service.
        """
        mj_model = self.sim_env.mj_model
        mj_data = self.sim_env.mj_data
        root_body_id = mj_model.body(self.sim_env.root_body).id
        robot_subtree = set(get_subtree_body_names(mj_model, root_body_id))

        self.entity_publishers = []
        enable_reset = True
        for bid in range(1, mj_model.nbody):  # skip world (id=0)
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or name in robot_subtree:
                continue
            pub = ObjectPosePublisher(
                mj_model, mj_data,
                object_name=name,
                topic_name=f"/simulation/pose/{name}",
                enable_reset_service=enable_reset,
            )
            enable_reset = False
            self.entity_publishers.append(pub)
            print(f"[MuJoCo] Publishing pose of '{name}' on /simulation/pose/{name}")

    def init_unitree_bridge(self):
        self.unitree_bridge = UnitreeSdk2Bridge(self.config)
        if self.config["USE_JOYSTICK"]:
            self.unitree_bridge.SetupJoystick(
                device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
            )

    def start(self):
        """Main simulation loop"""
        sim_cnt = 0
        ts = time.time()

        try:
            while self._running and (
                (self.sim_env.viewer and self.sim_env.viewer.is_running())
                or (self.sim_env.viewer is None)
            ):
                step_start = time.monotonic()

                self.sim_env.sim_step()
                now = time.time()
                if now - ts > 1 / 10.0 and self.redis_client is not None:
                    head_pose = self.sim_env.get_head_pose()
                    self.redis_client.set("head_pos", pickle.dumps(head_pose[:3]))
                    self.redis_client.set("head_quat", pickle.dumps(head_pose[3:]))
                    ts = now

                if sim_cnt % int(self.viewer_dt / self.sim_dt) == 0:
                    self.sim_env.update_viewer()

                    if hasattr(self, "entity_publishers"):
                        for pub in self.entity_publishers:
                            rclpy.spin_once(pub, timeout_sec=0)
                            if pub._reset_requested:
                                pub._reset_requested = False
                                self.sim_env.reset()
                                print("[MuJoCo] Reset triggered via /simulation/reset")
                            if getattr(pub, "_reset_object_requested", False):
                                pub._reset_object_requested = False
                                self.sim_env.reset_object()
                                print("[MuJoCo] Object reset triggered via /simulation/reset_object")
                            pub.publish()

                if sim_cnt % int(self.reward_dt / self.sim_dt) == 0:
                    self.sim_env.update_reward()

                if sim_cnt % int(self.image_dt / self.sim_dt) == 0:
                    self.sim_env.update_render_caches()

                # Simple rate limiter (replaces ROS rate)
                elapsed = time.monotonic() - step_start
                sleep_time = self.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                sim_cnt += 1
        except KeyboardInterrupt:
            print("Simulator interrupted by user.")
        finally:
            self.close()

    def __del__(self):
        self.close()

    def reset(self):
        self.sim_env.reset()

    def close(self):
        self._running = False
        try:
            if self.sim_env.image_publish_process is not None:
                self.sim_env.image_publish_process.stop()
            if self.sim_env.viewer is not None:
                self.sim_env.viewer.close()
        except Exception as e:
            print(f"Warning during close: {e}")

    def get_privileged_obs(self):
        return self.sim_env.get_privileged_obs()

    def handle_keyboard_button(self, key):
        self.sim_env.handle_keyboard_button(key)
