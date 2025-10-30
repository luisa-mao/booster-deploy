import numpy as np
import time
import yaml
import logging
import threading

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy import Policy

# ros stuff
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped


class Controller(Node):
    def __init__(self, cfg_file) -> None:
        super().__init__('controller') 
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.ball_pos = np.zeros(2, dtype=np.float32)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.pd_control_runner = B1LowStateSubscriber(self._publish_cmd)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()

            # ros stuff
            self._ball_sub = self.create_subscription(
                        Point,
                        "/apriltag/info",
                        self._ball_callback,
                        10
                    )
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _ball_callback(self, ball_msg: Point):
        try:
            self.ball_pos[:2] = np.array([ball_msg.x, ball_msg.y], dtype=np.float32)
            print("ball message", ball_msg)
            self.logger.info(f"Received ball position: x={ball_msg.x:.3f}, y={ball_msg.y:.3f}, z={ball_msg.z:.3f}")
        except Exception as e:
            self.logger.error(f"Failed to process ball message: {e}")

    def _low_state_handler(self, low_state_msg: LowState):
        print("Low state message received")
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

        self.run()

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)

    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        # self.publish_runner = threading.Thread(target=self._publish_cmd)
        # self.publish_runner.daemon = True
        # self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        print("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        print(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
            target_x = self.ball_pos[0],
            target_y = self.ball_pos[1],
        )

        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        print(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def _publish_cmd(self):
        # while self.running:
        if not self.running:
            return
        time_now = self.timer.get_time()
        if time_now < self.next_publish_time:
            time.sleep(0.001)
            return
        self.next_publish_time += self.cfg["common"]["dt"]
        self.logger.debug(f"Next publish time: {self.next_publish_time}")
        print(f"Next publish time: {self.next_publish_time}")

        self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

        for i in range(B1JointCnt):
            self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

        # Use series-parallel conversion for torque to avoid non-linearity
        for i in self.cfg["mech"]["parallel_mech_indexes"]:
            self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
            self.low_cmd.motor_cmd[i].tau = np.clip(
                (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                -self.cfg["common"]["torque_limit"][i],
                self.cfg["common"]["torque_limit"][i],
            )
            self.low_cmd.motor_cmd[i].kp = 0.0

        # zeros for now
        for i in range(B1JointCnt):
            self.low_cmd.motor_cmd[i].q = 0.0


        start_time = time.perf_counter()
        self._send_cmd(self.low_cmd)
        publish_time = time.perf_counter()
        print(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
        self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    import argparse
    import signal
    import sys
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    # 1️⃣ Initialize ROS 2
    rclpy.init()

    # try:
    #     controller = Controller(cfg_file)
    #     time.sleep(2)
    #     print("Initialization complete.")
    #     controller.start_custom_mode_conditionally()
    #     controller.start_rl_gait_conditionally()

    #     # 2️⃣ Spin the node (so subscriptions/timers actually run)
    #     executor = MultiThreadedExecutor()
    #     executor.add_node(controller)

    #     while controller.running and rclpy.ok():
    #         controller.run()
    #         executor.spin_once(timeout_sec=0.1)


    # finally:
    #     # 3️⃣ Proper shutdown
    #     controller.destroy_node()
    #     rclpy.shutdown()



    try:
        controller = Controller(cfg_file)
        
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        # Use rclpy.spin() to handle ROS2 callbacks while checking controller state
        while controller.running: # and not controller.shutdown_requested:
            try:
                rclpy.spin_once(controller, timeout_sec=0.1)
                time.sleep(min(controller.cfg["common"]["dt"], 0.1))
            except KeyboardInterrupt:
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Cleaning up...")
        controller.cleanup()

    # node = Controller(cfg_file)

    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()
