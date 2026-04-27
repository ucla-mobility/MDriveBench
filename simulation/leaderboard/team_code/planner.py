import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get("HAS_DISPLAY", 0))


class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x - r, y - r, x + r, y + r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = []
        self.min_distance = min_distance
        self.max_distance = max_distance
        self._last_run_step_debug = {}

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0])  # for carla 9.10
        self.scale = np.array([111324.60662786, 111319.490945])  # for carla 9.10

        self.debug = Plotter(debug_size)

    @staticmethod
    def _cmd_value(cmd):
        if cmd is None:
            return None
        return getattr(cmd, "value", None)

    @classmethod
    def _snapshot_node(cls, route, index):
        if index is None:
            return None
        idx = int(index)
        if idx < 0 or idx >= len(route):
            return None
        pos, cmd = route[idx]
        return {
            "index": idx,
            "x": round(float(pos[0]), 4),
            "y": round(float(pos[1]), 4),
            "cmd_value": cls._cmd_value(cmd),
            "cmd_type": None if cmd is None else type(cmd).__name__,
            "cmd_repr": None if cmd is None else str(cmd),
        }

    @classmethod
    def _snapshot_window(cls, route, center_index, radius=2):
        if center_index is None:
            return []
        center = int(center_index)
        start = max(0, center - int(radius))
        end = min(len(route), center + int(radius) + 1)
        return [cls._snapshot_node(route, idx) for idx in range(start, end)]

    def get_last_debug_snapshot(self, vehicle_num):
        snapshot = self._last_run_step_debug.get(int(vehicle_num))
        if snapshot is None:
            return {}
        return dict(snapshot)

    def set_route(self, global_plan, gps=False):
        self.route.clear()
        route_num = len(global_plan)
        for route_id in range(route_num):
            route_tmp = deque()
            for pos, cmd in global_plan[route_id]:
                if gps:
                    pos = np.array([pos["lat"], pos["lon"]])
                    pos -= self.mean
                    pos *= self.scale
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean
                route_tmp.append((pos, cmd))
            self.route.append(route_tmp)

    def run_step(self, gps, vehicle_num):
        self.debug.clear()
        route = self.route[vehicle_num]
        route_len_before_pop = len(route)

        if len(route) == 1:
            self._last_run_step_debug[vehicle_num] = {
                "vehicle_num": int(vehicle_num),
                "route_len_before_pop": 1,
                "route_len_after_pop": 1,
                "selected_source": "singleton",
                "selected_index": 0,
                "gps_x": round(float(gps[0]), 4),
                "gps_y": round(float(gps[1]), 4),
                "selected_node": self._snapshot_node(route, 0),
                "route_window": self._snapshot_window(route, 0),
            }
            return route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(
                route[i][0] - route[i - 1][0]
            )
            distance = np.linalg.norm(route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self._cmd_value(route[i][1]) == 4)
            b = 255
            self.debug.dot(gps, route[i][0], (r, g, b))

        actual_popped = 0
        for _ in range(to_pop):
            if len(route) > 2:
                route.popleft()
                actual_popped += 1

        self.debug.dot(gps, route[0][0], (0, 255, 0))
        self.debug.dot(gps, route[1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.show()

        # Guard: whenever route[1] is within safe_ahead of ego (including past
        # it), return a projected forward point.  Fires in tail mode AND
        # mid-route on tight geometry (roundabout curves, dense on-ramp
        # waypoints at high speed).
        seg_vec = route[1][0] - route[0][0]
        seg_norm = np.linalg.norm(seg_vec)
        selected_index = 1
        selected_source = "route_next"
        if seg_norm > 1e-6:
            seg_dir = seg_vec / seg_norm
            safe_ahead = max(1.0, 0.5 * self.min_distance)
            remaining = float(np.dot(route[1][0] - gps, seg_dir))
            if remaining <= safe_ahead:
                safe_pos = gps + seg_dir * safe_ahead
                selected_source = "safe_ahead"
                self._last_run_step_debug[vehicle_num] = {
                    "vehicle_num": int(vehicle_num),
                    "route_len_before_pop": int(route_len_before_pop),
                    "route_len_after_pop": int(len(route)),
                    "actual_popped": int(actual_popped),
                    "selected_source": selected_source,
                    "selected_index": selected_index,
                    "to_pop": int(to_pop),
                    "farthest_in_range": None if farthest_in_range == -np.inf else round(float(farthest_in_range), 4),
                    "cumulative_distance": round(float(cumulative_distance), 4),
                    "gps_x": round(float(gps[0]), 4),
                    "gps_y": round(float(gps[1]), 4),
                    "safe_ahead": round(float(safe_ahead), 4),
                    "remaining": round(float(remaining), 4),
                    "selected_node": self._snapshot_node(route, selected_index),
                    "previous_node": self._snapshot_node(route, 0),
                    "route_window": self._snapshot_window(route, selected_index),
                }
                return (safe_pos, route[1][1])

        self._last_run_step_debug[vehicle_num] = {
            "vehicle_num": int(vehicle_num),
            "route_len_before_pop": int(route_len_before_pop),
            "route_len_after_pop": int(len(route)),
            "actual_popped": int(actual_popped),
            "selected_source": selected_source,
            "selected_index": selected_index,
            "to_pop": int(to_pop),
            "farthest_in_range": None if farthest_in_range == -np.inf else round(float(farthest_in_range), 4),
            "cumulative_distance": round(float(cumulative_distance), 4),
            "gps_x": round(float(gps[0]), 4),
            "gps_y": round(float(gps[1]), 4),
            "selected_node": self._snapshot_node(route, selected_index),
            "previous_node": self._snapshot_node(route, 0),
            "route_window": self._snapshot_window(route, selected_index),
        }
        return route[1]

    def get_future_waypoints(self, vehicle_num, num=10):
        res = []
        for i in range(min(num, len(self.route[vehicle_num]))):
            res.append(
                [
                    self.route[vehicle_num][i][0][0],
                    self.route[vehicle_num][i][0][1],
                    self._cmd_value(self.route[vehicle_num][i][1]),
                ]
            )
        return res
