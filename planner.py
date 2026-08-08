"""NaviSTAR (STAR) wrapper for the arena_planners bridge.

Reconstructs, outside the upstream ``crowd_sim`` simulator, the STAR observation
dict (robot_node + spatial_edges_transformer + visible_masks + robot_pos) exactly
as ``crowd_sim/envs/crowd_sim.py::generate_ob`` builds it, runs the spatio-temporal
graph transformer policy, and returns the holonomic ``[vx, vy]`` velocity command.

NaviSTAR's native action space is holonomic (``ActionXY(vx, vy)``,
``config.action_space.kinematics == 'holonomic'``), so this planner is
omnidirectional. The raw policy output is norm-clipped to ``v_pref`` exactly like
upstream ``crowd_nav/policy/star.py::clip_action``; the differential-drive
projection (if any) is applied downstream by the bridge.

No observation normalization is applied: upstream ``test.py`` evaluates with a
plain ``DummyVecEnv`` (no ``VecNormalize`` / ``ob_rms``) and ships no running
mean/var stats, so the trained policy consumes raw observations.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch
from arena_planners.sdk import load_manifest, main_loop

from navistar_config import Config
from star_net import Policy

_HERE = pathlib.Path(__file__).parent
_MODEL_DIR = _HERE / "model"
_CHECKPOINT = _MODEL_DIR / "00500.pt"

# From data/navigation/star/configs/config.py.
_RADIUS: float = 0.3        # config.robot.radius
_V_PREF: float = 1.0        # config.robot.v_pref
_VISIBLE_DIS: float = 10.0  # config.robot.visible_dis
# Absent / out-of-FOV humans: crowd_sim initializes last_human_states at (15, 15).
_ABSENT_XY: float = 15.0


class _Runner:
    def __init__(self) -> None:
        self.config = Config()
        self.human_num = int(self.config.sim.human_num)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = Policy(None, 2, self.config, device=self.device)
        self.policy.base.nenv = 1  # acting model: single env
        state = torch.load(str(_CHECKPOINT), map_location=self.device)
        self.policy.load_state_dict(state, strict=True)
        self.policy.to(self.device)
        self.policy.eval()
        self.reset_state()

    def reset_state(self) -> None:
        # masks=0 on the very first step (episode start), 1.0 thereafter — matches
        # the upstream eval loop (eval_masks starts at 0).
        self._first = True

    def act(self, robot_node, spatial_edges_transformer, visible_masks, robot_pos):
        masks = torch.zeros(1, 1) if self._first else torch.ones(1, 1)
        obs = {
            "robot_node": torch.as_tensor(robot_node, dtype=torch.float32).reshape(1, 1, 7),
            "spatial_edges_transformer": torch.as_tensor(
                spatial_edges_transformer, dtype=torch.float32
            ).reshape(1, self.human_num + 1, 4),
            "visible_masks": torch.as_tensor(visible_masks, dtype=torch.float32).reshape(
                1, self.human_num + 1, 1
            ),
            "robot_pos": torch.as_tensor(robot_pos, dtype=torch.float32).reshape(1, 1, 4),
        }
        obs = {k: v.to(self.device) for k, v in obs.items()}
        masks = masks.to(self.device)
        with torch.no_grad():
            _, action, _ = self.policy.act(obs, {}, masks, deterministic=True)
        self._first = False
        return action.reshape(-1).cpu().numpy()


_runner: _Runner | None = None


def _get_runner() -> _Runner:
    global _runner
    if _runner is None:
        _runner = _Runner()
    return _runner


def step(features: dict) -> list[float]:
    """Map the bridge feature dict to a holonomic [vx, vy] velocity command."""
    runner = _get_runner()

    robot_pose = features.get("robot_pose")
    goal_pose = features.get("goal_pose")
    if robot_pose is None or goal_pose is None:
        return [0.0, 0.0]

    px, py, theta = float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])
    robot_state = features.get("robot_state")
    if robot_state is not None and len(robot_state) >= 4:
        vx, vy = float(robot_state[2]), float(robot_state[3])
    else:
        vx, vy = 0.0, 0.0
    gx, gy = float(goal_pose[0]), float(goal_pose[1])

    n = runner.human_num

    # robot_node: [px, py, radius, gx, gy, v_pref, theta] (get_full_state_list_noV),
    # robot-centered (STAR is origin-trained, not translation-invariant).
    robot_node = np.array([0.0, 0.0, _RADIUS, gx - px, gy - py, _V_PREF, theta], dtype=np.float32)

    # last_human_states: nearest human_num humans, absent ones fixed at (15, 15).
    peds = features.get("pedestrians")
    rel = []
    if peds is not None:
        for ped in peds:
            hx, hy = float(ped[1]), float(ped[2])
            d2 = (hx - px) ** 2 + (hy - py) ** 2
            rel.append((d2, hx, hy))
    rel.sort(key=lambda r: r[0])

    last_human_xy = np.empty((n, 2), dtype=np.float32)
    last_human_xy[:] = (_ABSENT_XY, _ABSENT_XY)
    for i in range(min(n, len(rel))):
        last_human_xy[i] = (rel[i][1], rel[i][2])

    # spatial_edges_transformer [human_num+1, 4]:
    #   per-human row: [hx-px, hy-py, 0, 1]   (one_hot = [0, 1])
    #   final goal row: [gx-px, gy-py, 1, 0]
    spatial = np.zeros((n + 1, 4), dtype=np.float32)
    spatial[:n, 0] = last_human_xy[:, 0] - px
    spatial[:n, 1] = last_human_xy[:, 1] - py
    spatial[:n, 2] = 0.0
    spatial[:n, 3] = 1.0
    spatial[n] = (gx - px, gy - py, 1.0, 0.0)

    # visible_masks [human_num+1]: (||row|| < visible_dis); last entry forced to 1.
    dis = np.linalg.norm(spatial, axis=-1)
    visible_masks = (dis < _VISIBLE_DIS).astype(np.float32)
    visible_masks[-1] = 1.0

    # robot_pos [1, 4]: [px, py, 1, 0], robot-centered for the GCN adjacency.
    robot_pos = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    raw = runner.act(robot_node, spatial, visible_masks, robot_pos)

    # clip_action (holonomic branch): scale to v_pref if the norm exceeds it.
    raw = np.asarray(raw, dtype=np.float64)
    act_norm = float(np.linalg.norm(raw))
    if act_norm > _V_PREF:
        raw = raw / act_norm * _V_PREF
    return [float(raw[0]), float(raw[1])]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    runner = _get_runner()
    runner.reset_state()


if __name__ == "__main__":
    manifest = load_manifest(_HERE / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
