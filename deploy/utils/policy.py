import numpy as np
import torch


class Policy:
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.joint_inds = np.array(self.cfg["common"]["joint_indices"], dtype=np.int32) if self.cfg["common"].get("joint_indices") else np.array([11,12,13,14,15,16,17,18,19,20,21,22], dtype=np.int32)
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, target_x, target_y):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        # if np.linalg.norm(self.smoothed_commands) < 1e-5:
        #     self.gait_frequency = 0.0
        #else:
        self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        n = self.joint_inds.shape[0]

        # print("xyz:", self.smoothed_commands[0:3])
        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        # self.obs[6] = (
        #     self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        #     )
        # self.obs[7] = (
        #     self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        # )
        # self.obs[8] = (
        #     self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        # )
        self.obs[6] = 0.5 # target_x
        self.obs[7] = 0 # target_y
        self.obs[8] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[9] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10:10+n] = (dof_pos - self.default_dof_pos)[self.joint_inds] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[10+n:10+2*n] = dof_vel[self.joint_inds] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[10+2*n:] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[self.joint_inds] += self.cfg["policy"]["control"]["action_scale"] * self.actions
        # print("obs", self.obs)
        # print("actions", self.actions)
        return self.dof_targets
