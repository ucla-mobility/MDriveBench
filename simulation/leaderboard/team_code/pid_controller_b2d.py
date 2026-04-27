from collections import deque
import numpy as np

class PID(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D

		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)
			derivative = (self._window[-1] - self._window[-2])
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative



class PIDController(object):
    
    def __init__(self, turn_KP=0.75, turn_KI=0.75, turn_KD=0.3, turn_n=40, speed_KP=5.0, speed_KI=0.5,speed_KD=1.0, speed_n = 40,max_throttle=0.75, brake_speed=0.4,brake_ratio=1.1, clip_delta=0.25, aim_dist=4.0, angle_thresh=0.3, dist_thresh=10):
        
        self.turn_controller = PID(K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n)
        self.speed_controller = PID(K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n)
        self.max_throttle = max_throttle
        self.brake_speed = brake_speed
        self.brake_ratio = brake_ratio
        self.clip_delta = clip_delta
        self.aim_dist = aim_dist
        self.angle_thresh = angle_thresh
        self.dist_thresh = dist_thresh

    def control_pid(self, waypoints, speed, target, tail_mode=False, tail_passed_terminal=False):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            speed (tensor): speedometer input
        '''

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.aim_dist-best_norm) > abs(self.aim_dist-norm):
                aim = waypoints[i]
                best_norm = norm

        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # ── Smooth blend replaces the original binary switch ──────────
        # The old code: use_target_to_aim = |angle_target| < |angle|
        # selected either the route target angle or the model trajectory
        # angle exclusively.  On long straights both angles are near-zero,
        # so the switch flips on noise each frame, preventing the PID
        # integral from building a consistent cross-track correction.
        #
        # Fix: a sigmoid weight that smoothly favours whichever source
        # predicts a smaller correction.  At large |Δ| this approximates
        # the original binary behaviour; at small |Δ| (straight roads) it
        # produces a ~50/50 blend so the PID integrates smoothly.
        #
        # A floor of 0.15 guarantees the route target always contributes
        # at least 15 %, providing continuous cross-track lane-centring.
        mag_diff = float(np.abs(angle) - np.abs(angle_target))
        w_target = 1.0 / (1.0 + np.exp(-8.0 * mag_diff))
        w_target = max(w_target, 0.15)

        # Override: snap fully to route target on sudden close-range turn
        # commands (original condition 2, kept for backward compatibility).
        sudden_turn = bool(
            np.abs(angle_target - angle_last) > self.angle_thresh
            and target[1] < self.dist_thresh
        )
        if sudden_turn:
            w_target = 1.0

        # Endpoint safety guard: once we're in route tail mode, a behind-ego
        # command target can cause a large wrapped target angle and sharp steer
        # reversal. Keep steering on trajectory angle in this narrow case.
        terminal_behind_guard = bool(tail_mode and target[1] <= 0.0)
        if terminal_behind_guard:
            w_target = 0.0

        angle_final = w_target * angle_target + (1.0 - w_target) * angle

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(waypoints[3].astype(np.float64)),
            'wp_3': tuple(waypoints[2].astype(np.float64)),
            'wp_2': tuple(waypoints[1].astype(np.float64)),
            'wp_1': tuple(waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(target.astype(np.float64)),
            'desired_speed': float(desired_speed.astype(np.float64)),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final),
            'delta': float(delta.astype(np.float64)),
            'tail_mode': bool(tail_mode),
            'tail_passed_terminal': bool(tail_passed_terminal),
            'terminal_behind_guard': bool(terminal_behind_guard),
            'w_target': float(w_target),
            'sudden_turn': bool(sudden_turn),
        }

        return steer, throttle, brake, metadata
