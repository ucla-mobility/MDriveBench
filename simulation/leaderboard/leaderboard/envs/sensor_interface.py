import copy
import logging
import numpy as np
import os
import time
from pathlib import Path
from threading import Condition, Lock, Thread

from queue import Queue
from queue import Full

import carla
import cv2
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.setDaemon(True)
        thread.start()

        return thread
    return wrapper


class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class GenericMeasurement(object):
    def __init__(self, data, frame):
        self.data = data
        self.frame = frame


class BaseReader(object):
    def __init__(self, vehicle, reading_frequency=1.0):
        self._vehicle = vehicle
        self._reading_frequency = reading_frequency
        self._callback = None
        self._run_ps = True
        self.run()

    def __call__(self):
        pass

    @threaded
    def run(self):
        first_time = True
        latest_time = GameTime.get_time()
        while self._run_ps:
            if self._callback is not None:
                current_time = GameTime.get_time()

                # Second part forces the sensors to send data at the first tick, regardless of frequency
                if current_time - latest_time > (1 / self._reading_frequency) \
                        or (first_time and GameTime.get_frame() != 0):
                    self._callback(GenericMeasurement(self.__call__(), GameTime.get_frame()))
                    latest_time = GameTime.get_time()
                    first_time = False

                else:
                    time.sleep(0.001)

    def listen(self, callback):
        # Tell that this function receives what the producer does.
        self._callback = callback

    def stop(self):
        self._run_ps = False

    def destroy(self):
        self._run_ps = False


class SpeedometerReader(BaseReader):
    """
    Sensor to measure the speed of the vehicle.
    """
    MAX_CONNECTION_ATTEMPTS = 10

    def _get_forward_speed(self, transform=None, velocity=None, acc=None, vel_angle=None):
        """ Convert the vehicle transform directly to forward speed """
        if not velocity:
            velocity = self._vehicle.get_velocity()
        if not transform:
            transform = self._vehicle.get_transform()
        if not acc:
            acc=self._vehicle.get_acceleration()
        if not vel_angle:
            vel_angle = self._vehicle.get_angular_velocity()

        vel_list = [velocity.x, velocity.y, velocity.z]
        vel_np = np.array(vel_list)
        acc_np = [acc.x, acc.y, acc.z]
        vel_angle_np = [vel_angle.x, vel_angle.y, vel_angle.z]
        transform_dict = {'x':transform.location.x,
                          'y':transform.location.y,
                          'z':transform.location.z,
                          'roll':transform.rotation.roll,
                          'yaw':transform.rotation.yaw,
                          'pitch':transform.rotation.pitch}
        pitch = np.deg2rad(transform.rotation.pitch)
        yaw = np.deg2rad(transform.rotation.yaw)
        orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
        speed = np.dot(vel_np, orientation)
        return speed, vel_list, acc_np, vel_angle_np, transform_dict

    def __call__(self):
        """ We convert the vehicle physics information into a convenient dictionary """

        # protect this access against timeout
        attempts = 0
        while attempts < self.MAX_CONNECTION_ATTEMPTS:
            try:
                velocity = self._vehicle.get_velocity()
                transform = self._vehicle.get_transform()
                acc=self._vehicle.get_acceleration()
                vel_angle = self._vehicle.get_angular_velocity()
                break
            except Exception:
                attempts += 1
                time.sleep(0.2)
                continue

        speed, vel_np, acc_np, vel_angle_np, transform_dict = self._get_forward_speed(transform=transform, velocity=velocity, acc=acc, vel_angle=vel_angle)

        return {'speed': speed,
                'move_state':{'speed': speed,
                    'speed_xyz': vel_np,
                    'acc_xyz': acc_np,
                    'speed_angle_xyz': vel_angle_np,
                    'transform_xyz_rollyawpitch':transform_dict}}


class OpenDriveMapReader(BaseReader):
    def __call__(self):
        return {'opendrive': CarlaDataProvider.get_map().to_opendrive()}


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag
        self._data_provider = data_provider

        self._data_provider.register_sensor(tag, sensor_type, sensor)

    def __call__(self, data):
        if isinstance(data, carla.libcarla.Image):
            self._parse_image_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.LidarMeasurement):
            self._parse_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
            self._parse_semantic_lidar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.RadarMeasurement):
            self._parse_radar_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.GnssMeasurement):
            self._parse_gnss_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.IMUMeasurement):
            self._parse_imu_cb(data, self._tag)
        elif isinstance(data, carla.libcarla.CollisionEvent):
            self._parse_collision_cb(data, self._tag)
        elif isinstance(data, GenericMeasurement):
            self._parse_pseudosensor(data, self._tag)
        else:
            logging.error('No callback method for this sensor.')

    # Parsing CARLA physical Sensors
    def _parse_image_cb(self, image, tag):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = copy.deepcopy(array)
        array = np.reshape(array, (image.height, image.width, 4))
        self._data_provider.maybe_save_image_frame(tag, array, image.frame)
        self._data_provider.update_sensor(tag, array, image.frame)

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_semantic_lidar_cb(self, semantic_lidar_data, tag):
        points = np.frombuffer(semantic_lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        self._data_provider.update_sensor(tag, points, semantic_lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        # [depth, azimuth, altitute, velocity]
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        points = np.flip(points, 1)
        self._data_provider.update_sensor(tag, points, radar_data.frame)

    def _parse_gnss_cb(self, gnss_data, tag):
        array = np.array([gnss_data.latitude,
                          gnss_data.longitude,
                          gnss_data.altitude], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, gnss_data.frame)

    def _parse_imu_cb(self, imu_data, tag):
        array = np.array([imu_data.accelerometer.x,
                          imu_data.accelerometer.y,
                          imu_data.accelerometer.z,
                          imu_data.gyroscope.x,
                          imu_data.gyroscope.y,
                          imu_data.gyroscope.z,
                          imu_data.compass,
                         ], dtype=np.float64)
        self._data_provider.update_sensor(tag, array, imu_data.frame)

    def _parse_collision_cb(self, collision_event, tag):
        impulse = collision_event.normal_impulse
        other_actor = getattr(collision_event, "other_actor", None)
        role_name = ""
        type_id = ""
        actor_id = None
        if other_actor is not None:
            actor_id = int(getattr(other_actor, "id", -1))
            type_id = str(getattr(other_actor, "type_id", ""))
            if hasattr(other_actor, "attributes"):
                role_name = other_actor.attributes.get("role_name", "")
        payload = {
            "has_collision": True,
            "event_frame": int(getattr(collision_event, "frame", GameTime.get_frame())),
            "other_actor_id": actor_id,
            "other_actor_type": type_id,
            "other_actor_role_name": role_name,
            "normal_impulse": [
                float(getattr(impulse, "x", 0.0)),
                float(getattr(impulse, "y", 0.0)),
                float(getattr(impulse, "z", 0.0)),
            ],
            "normal_impulse_magnitude": float(
                np.linalg.norm([
                    float(getattr(impulse, "x", 0.0)),
                    float(getattr(impulse, "y", 0.0)),
                    float(getattr(impulse, "z", 0.0)),
                ])
            ),
        }
        self._data_provider.update_sensor(tag, payload, payload["event_frame"])

    def _parse_pseudosensor(self, package, tag):
        self._data_provider.update_sensor(tag, package.data, package.frame)


class SensorInterface(object):
    def __init__(self):
        self._sensors_objects = {}
        self._sensor_types = {}
        self._mailbox_lock = Lock()
        self._mailbox_cv = Condition(self._mailbox_lock)
        self._latest_data = {}
        self._queue_timeout = 100 # default: 10
        self._last_returned_timestamps = {}
        self._event_sensor_types = {"sensor.other.collision"}

        # Only sensor that doesn't get the data on tick, needs special treatment
        self._opendrive_tag = None
        # Sensor forensics: per-tag lifecycle tracking + detailed timeout diagnostics.
        # Enable with CARLA_SENSOR_FORENSICS=1. Cheap (one dict update per packet).
        self._forensics_enabled = os.environ.get("CARLA_SENSOR_FORENSICS", "").lower() in ("1", "true", "yes")
        # Verbose: log EVERY sensor callback fire. Heavy. Implies _forensics_enabled.
        self._forensics_trace_callbacks = os.environ.get("CARLA_SENSOR_TRACE_CALLBACKS", "").lower() in ("1", "true", "yes")
        if self._forensics_trace_callbacks:
            self._forensics_enabled = True
        # Per-tag dict: tag -> {registered_at, first_packet_at, last_packet_at, packet_count,
        #                       frame_first, frame_last, sensor_type, ego_id, dropped_stale,
        #                       dropped_unregistered, last_dropped_at}
        self._sensor_lifecycle = {}
        # Module-level instance counter so multiple SensorInterface objects (per-ego)
        # can be told apart in the log stream.
        SensorInterface._instance_counter = getattr(SensorInterface, "_instance_counter", 0) + 1
        self._forensics_instance_id = SensorInterface._instance_counter

        # Grace period after the primary _queue_timeout expires.  If a missing
        # sensor's first packet arrives during the grace window, get_data()
        # succeeds and we log a DELAYED_FIRST_PACKET event.  Empirically the
        # CARLA streaming layer can take 100-103s to deliver the first packet
        # for late-spawned IMU/GPS sensors (race in session-open), so a 15s
        # grace catches the long tail without significantly extending failures.
        try:
            # 60s default grace (was 15s).  As long as the streaming layer
            # eventually delivers, accept the late packet rather than failing
            # the whole scenario.  Higher values are essentially free since
            # the path only triggers when a sensor is genuinely missing data
            # (most scenarios never enter the grace window at all).
            self._queue_grace_s = max(0.0, float(os.environ.get("CARLA_SENSOR_GRACE_S", "60")))
        except (TypeError, ValueError):
            self._queue_grace_s = 60.0

        self._dense_capture_enabled = os.environ.get("TCP_CAPTURE_SENSOR_FRAMES", "").lower() in ("1", "true", "yes")
        self._dense_capture_targets = {}
        self._dense_capture_lock = Lock()
        self._dense_capture_queue = None
        self._dense_capture_writer_thread = None
        self._dense_capture_queue_max = 0
        self._dense_capture_png_compression = 1
        self._dense_capture_dropped = 0
        if self._dense_capture_enabled:
            try:
                self._dense_capture_queue_max = max(16, int(os.environ.get("TCP_CAPTURE_QUEUE_MAX", "256")))
            except (TypeError, ValueError):
                self._dense_capture_queue_max = 256
            try:
                self._dense_capture_png_compression = min(
                    9, max(0, int(os.environ.get("TCP_CAPTURE_PNG_COMPRESSION", "1")))
                )
            except (TypeError, ValueError):
                self._dense_capture_png_compression = 1
            self._dense_capture_queue = Queue(maxsize=self._dense_capture_queue_max)
            self._dense_capture_writer_thread = Thread(
                target=self._dense_capture_writer_loop,
                name="dense_image_writer",
                daemon=True,
            )
            self._dense_capture_writer_thread.start()

    def configure_image_capture(self, tag, output_dir):
        """
        Register a per-sensor image sink. When dense capture is enabled, every incoming
        frame for this sensor will be written to disk from the sensor callback thread.
        """
        if not self._dense_capture_enabled:
            return
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with self._dense_capture_lock:
            self._dense_capture_targets[tag] = {
                "dir": output_path,
                "counter": 0,
                "last_frame": None,
            }

    def unregister_image_capture(self, tag):
        if not self._dense_capture_enabled:
            return
        with self._dense_capture_lock:
            self._dense_capture_targets.pop(tag, None)

    def _dense_capture_writer_loop(self):
        while True:
            item = self._dense_capture_queue.get()
            if item is None:
                self._dense_capture_queue.task_done()
                break
            out_path, bgr = item
            try:
                cv2.imwrite(
                    str(out_path),
                    bgr,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), self._dense_capture_png_compression],
                )
            except Exception:
                pass
            finally:
                self._dense_capture_queue.task_done()

    def finalize_image_capture(self):
        if not self._dense_capture_enabled or self._dense_capture_queue is None:
            return
        # Flush pending writes before cleanup.
        try:
            self._dense_capture_queue.join()
        except Exception:
            pass
        try:
            self._dense_capture_queue.put_nowait(None)
        except Exception:
            pass
        if self._dense_capture_writer_thread is not None:
            self._dense_capture_writer_thread.join(timeout=5.0)
        if self._dense_capture_dropped > 0:
            print(
                "[SensorInterface] Dense capture dropped {} frames (queue max={}).".format(
                    self._dense_capture_dropped, self._dense_capture_queue_max
                )
            )

    def maybe_save_image_frame(self, tag, image_bgra, frame):
        if not self._dense_capture_enabled:
            return
        if self._dense_capture_queue is None:
            return
        target = None
        with self._dense_capture_lock:
            target = self._dense_capture_targets.get(tag)
            if target is None:
                return
            if target["last_frame"] == frame:
                return
            out_path = target["dir"] / f"{target['counter']:06d}.png"
            target["counter"] += 1
            target["last_frame"] = frame
        try:
            # Callback payload is BGRA. Keep BGR and offload PNG encoding to writer thread.
            bgr = np.ascontiguousarray(image_bgra[:, :, :3])
            self._dense_capture_queue.put_nowait((out_path, bgr))
        except Full:
            self._dense_capture_dropped += 1
        except Exception:
            pass


    def register_sensor(self, tag, sensor_type, sensor):
        with self._mailbox_cv:
            if tag in self._sensors_objects:
                raise SensorConfigurationInvalid("Duplicated sensor tag [{}]".format(tag))

            self._sensors_objects[tag] = sensor
            self._sensor_types[tag] = sensor_type
            self._last_returned_timestamps[tag] = None
            self._latest_data.pop(tag, None)
            if sensor_type in self._event_sensor_types:
                self._latest_data[tag] = (
                    -1,
                    {
                        "has_collision": False,
                        "event_frame": None,
                        "other_actor_id": None,
                        "other_actor_type": "",
                        "other_actor_role_name": "",
                        "normal_impulse": [0.0, 0.0, 0.0],
                        "normal_impulse_magnitude": 0.0,
                    },
                )

            if sensor_type == 'sensor.opendrive_map':
                self._opendrive_tag = tag

            if self._forensics_enabled:
                # Sensor tag convention: "<base>_<ego_id>" (e.g. "rgb_front_2", "lidar_0").
                ego_id = None
                try:
                    ego_id = int(tag.rsplit("_", 1)[1])
                except (ValueError, IndexError):
                    pass
                sensor_actor_id = None
                try:
                    sensor_actor_id = int(getattr(sensor, "id", None))
                except (TypeError, ValueError):
                    pass
                self._sensor_lifecycle[tag] = {
                    "registered_at": time.time(),
                    "first_packet_at": None,
                    "last_packet_at": None,
                    "packet_count": 0,
                    "frame_first": None,
                    "frame_last": None,
                    "sensor_type": sensor_type,
                    "ego_id": ego_id,
                    "sensor_actor_id": sensor_actor_id,
                    "dropped_stale": 0,
                    "dropped_unregistered": 0,
                    "last_dropped_at": None,
                }
                print(
                    f"[SENSOR_FORENSICS] REGISTER instance={self._forensics_instance_id} "
                    f"tag={tag} type={sensor_type} ego_id={ego_id} actor_id={sensor_actor_id}",
                    flush=True,
                )

    def unregister_sensor(self, tag):
        with self._mailbox_cv:
            if self._forensics_enabled and tag in self._sensor_lifecycle:
                lc = self._sensor_lifecycle[tag]
                now = time.time()
                lifetime_s = now - lc["registered_at"]
                first_age = (now - lc["first_packet_at"]) if lc["first_packet_at"] else None
                last_age = (now - lc["last_packet_at"]) if lc["last_packet_at"] else None
                print(
                    f"[SENSOR_FORENSICS] UNREGISTER instance={self._forensics_instance_id} "
                    f"tag={tag} lifetime_s={lifetime_s:.2f} packets={lc['packet_count']} "
                    f"first_packet_age_s={first_age if first_age is None else f'{first_age:.2f}'} "
                    f"last_packet_age_s={last_age if last_age is None else f'{last_age:.2f}'} "
                    f"dropped_stale={lc['dropped_stale']} dropped_unregistered={lc['dropped_unregistered']}",
                    flush=True,
                )
            self._sensors_objects.pop(tag, None)
            self._sensor_types.pop(tag, None)
            self._last_returned_timestamps.pop(tag, None)
            self._latest_data.pop(tag, None)
            if self._opendrive_tag == tag:
                self._opendrive_tag = None
            self._mailbox_cv.notify_all()

    def update_sensor(self, tag, data, timestamp):
        # Keep only the latest packet per sensor tag (bounded mailbox).
        with self._mailbox_cv:
            if tag not in self._sensors_objects:
                # sensor may have been removed (e.g., ego destroyed); ignore late callbacks
                if self._forensics_enabled and tag in self._sensor_lifecycle:
                    lc = self._sensor_lifecycle[tag]
                    lc["dropped_unregistered"] += 1
                    lc["last_dropped_at"] = time.time()
                return

            # Forensic lifecycle tracking: record EVERY callback that arrives at the
            # Python side for a registered tag, including ones that will be dropped
            # by the staleness checks below. This distinguishes "sensor never fired"
            # from "sensor fired but data was rejected".
            if self._forensics_enabled and tag in self._sensor_lifecycle:
                lc = self._sensor_lifecycle[tag]
                now = time.time()
                if lc["first_packet_at"] is None:
                    lc["first_packet_at"] = now
                    lc["frame_first"] = timestamp
                lc["last_packet_at"] = now
                lc["frame_last"] = timestamp
                lc["packet_count"] += 1
                if self._forensics_trace_callbacks:
                    print(
                        f"[SENSOR_FORENSICS] CALLBACK instance={self._forensics_instance_id} "
                        f"tag={tag} frame={timestamp} count={lc['packet_count']} "
                        f"age_since_register_s={now - lc['registered_at']:.3f}",
                        flush=True,
                    )

            last_returned = self._last_returned_timestamps.get(tag)
            if last_returned is not None and timestamp < last_returned:
                if self._forensics_enabled and tag in self._sensor_lifecycle:
                    self._sensor_lifecycle[tag]["dropped_stale"] += 1
                    self._sensor_lifecycle[tag]["last_dropped_at"] = time.time()
                return

            current = self._latest_data.get(tag)
            if current is not None and timestamp < current[0]:
                if self._forensics_enabled and tag in self._sensor_lifecycle:
                    self._sensor_lifecycle[tag]["dropped_stale"] += 1
                    self._sensor_lifecycle[tag]["last_dropped_at"] = time.time()
                return

            self._latest_data[tag] = (timestamp, data)
            self._mailbox_cv.notify_all()

    def reset_after_recovery(self):
        """
        Reset mailbox progression so fresh post-recovery sensor packets are accepted
        even though CARLA frame counters may restart from zero.
        """
        with self._mailbox_cv:
            for tag in list(self._last_returned_timestamps.keys()):
                self._last_returned_timestamps[tag] = None
            # Keep event-sensor defaults but clear stale frame-bound payloads.
            preserved = {}
            for tag, sensor_type in self._sensor_types.items():
                if sensor_type in self._event_sensor_types:
                    preserved[tag] = (
                        -1,
                        {
                            "has_collision": False,
                            "event_frame": None,
                            "other_actor_id": None,
                            "other_actor_type": "",
                            "other_actor_role_name": "",
                            "normal_impulse": [0.0, 0.0, 0.0],
                            "normal_impulse_magnitude": 0.0,
                        },
                    )
            self._latest_data = preserved
            self._mailbox_cv.notify_all()

    def get_data(self):
        # Two-stage deadline:
        #   stage 1: primary timeout (_queue_timeout, default 100s)
        #   stage 2: grace period (_queue_grace_s, default 15s) -- this catches
        #     the empirically-observed ~101s late-arrival of CARLA IMU/GPS first
        #     packets when sessions race during multi-ego setup.  See
        #     INVESTIGATION.md for full root-cause analysis.
        primary_deadline = time.time() + self._queue_timeout
        grace_deadline = primary_deadline + self._queue_grace_s
        in_grace = False
        grace_entered_at = None
        grace_initial_missing = None
        with self._mailbox_cv:
            while True:
                sensor_tags = list(self._sensors_objects.keys())
                required_tags = [tag for tag in sensor_tags if tag != self._opendrive_tag]

                data_dict = {}
                missing_tags = []
                for tag in required_tags:
                    sample = self._latest_data.get(tag)
                    sensor_type = self._sensor_types.get(tag)
                    is_event_sensor = sensor_type in self._event_sensor_types
                    if sample is None:
                        if is_event_sensor:
                            data_dict[tag] = (
                                -1,
                                {
                                    "has_collision": False,
                                    "event_frame": None,
                                    "other_actor_id": None,
                                    "other_actor_type": "",
                                    "other_actor_role_name": "",
                                    "normal_impulse": [0.0, 0.0, 0.0],
                                    "normal_impulse_magnitude": 0.0,
                                },
                            )
                        else:
                            missing_tags.append(tag)
                        continue
                    timestamp, payload = sample
                    last_returned = self._last_returned_timestamps.get(tag)
                    # Require a strictly newer sample per get_data() call.
                    if (not is_event_sensor) and last_returned is not None and timestamp <= last_returned:
                        missing_tags.append(tag)
                        continue
                    data_dict[tag] = (timestamp, payload)

                if not missing_tags:
                    # opendrive_map is optional and may stay stale by design
                    if self._opendrive_tag and self._opendrive_tag in sensor_tags:
                        opendrive_sample = self._latest_data.get(self._opendrive_tag)
                        if opendrive_sample is not None:
                            data_dict[self._opendrive_tag] = opendrive_sample

                    # If we entered the grace period and recovered, log which
                    # sensors arrived late so we can confirm the warmup-tick fix
                    # is working (or not needed) on this scenario.
                    if in_grace and self._forensics_enabled:
                        self._emit_grace_recovery(
                            grace_entered_at=grace_entered_at,
                            grace_initial_missing=grace_initial_missing,
                        )

                    for tag, (timestamp, _) in data_dict.items():
                        self._last_returned_timestamps[tag] = timestamp
                    return data_dict

                now = time.time()
                # First crossing of the primary deadline -- enter grace.
                if (not in_grace) and now >= primary_deadline and self._queue_grace_s > 0:
                    in_grace = True
                    grace_entered_at = now
                    grace_initial_missing = list(missing_tags)
                    if self._forensics_enabled:
                        print(
                            f"[SENSOR_FORENSICS] GRACE_ENTERED instance={self._forensics_instance_id} "
                            f"primary_timeout_s={self._queue_timeout} grace_s={self._queue_grace_s} "
                            f"missing_at_grace_entry={missing_tags}",
                            flush=True,
                        )

                # Final deadline (primary + grace) -- give up.
                if now >= grace_deadline:
                    if self._forensics_enabled:
                        self._emit_sensor_timeout_diagnostics(
                            missing_tags=missing_tags,
                            required_tags=required_tags,
                            grace_used_s=(self._queue_grace_s if in_grace else 0.0),
                        )
                    raise SensorReceivedNoData("A sensor took too long to send their data")

                # Important: cap the wait at the *primary* deadline (not the
                # grace deadline) so we always re-check + emit GRACE_ENTERED
                # exactly at the primary expiration. Without this cap, a
                # single long wait could span both windows and we'd never
                # log entering grace.
                next_check = grace_deadline if in_grace else primary_deadline
                self._mailbox_cv.wait(timeout=max(0.001, next_check - now))

    def _emit_grace_recovery(self, grace_entered_at, grace_initial_missing):
        """
        Called when get_data() recovers during the grace period.  Reports
        which sensors arrived late and how late, so the operator can verify
        the warmup-tick fix is reducing or eliminating these recoveries.
        """
        now = time.time()
        grace_used = now - grace_entered_at
        # For each tag that was missing at grace entry, check whether it now
        # has a packet, and how long after the primary deadline it arrived.
        details = []
        for tag in grace_initial_missing:
            lc = self._sensor_lifecycle.get(tag)
            if lc is None:
                details.append(f"{tag}=NO_LIFECYCLE")
                continue
            if lc["first_packet_at"] is None:
                # Tag is no longer "missing" but has no first_packet_at either;
                # likely an event sensor that filled with default.
                details.append(f"{tag}=event_default")
                continue
            arrived_age_s = now - lc["first_packet_at"]
            registered_age_s = now - lc["registered_at"]
            details.append(
                f"{tag}(first_packet_arrived_age_s={registered_age_s - arrived_age_s:.2f},"
                f"packets_now={lc['packet_count']})"
            )
        print(
            f"[SENSOR_FORENSICS] GRACE_RECOVERY instance={self._forensics_instance_id} "
            f"grace_used_s={grace_used:.2f} recovered_tags={grace_initial_missing} "
            f"details=[{', '.join(details)}]",
            flush=True,
        )

    def _emit_sensor_timeout_diagnostics(self, missing_tags, required_tags, grace_used_s=0.0):
        """
        Called immediately before raising SensorReceivedNoData. Dumps a structured
        forensic summary that distinguishes:
          - sensors that never delivered ANY packet (channel never opened),
          - sensors that delivered some packets then went silent (channel hung),
          - sensors delivering normally (so we can compute coverage % per ego),
          - which egos are most affected.
        Output is grep-friendly (prefix "[SENSOR_FORENSICS]") and contains all the
        information needed to classify the failure into one of the decision-tree
        branches (all-egos / one-ego / one-type / random / mid-scenario).
        """
        now = time.time()
        per_ego = {}
        delivered_normally = 0

        # First: per-tag detail for every required sensor (delivered or not)
        for tag in required_tags:
            lc = self._sensor_lifecycle.get(tag)
            sensor_type = self._sensor_types.get(tag, "?")
            if lc is None:
                # Forensics enabled but no lifecycle entry — sensor was registered
                # before forensics turned on, OR registration path skipped.
                print(
                    f"[SENSOR_FORENSICS] TIMEOUT_TAG instance={self._forensics_instance_id} "
                    f"tag={tag} type={sensor_type} state=NO_LIFECYCLE_ENTRY",
                    flush=True,
                )
                continue

            ego_id = lc["ego_id"]
            per_ego.setdefault(ego_id, {"delivered": 0, "missing": 0, "total": 0})
            per_ego[ego_id]["total"] += 1

            is_missing = tag in missing_tags
            if is_missing:
                per_ego[ego_id]["missing"] += 1
            else:
                per_ego[ego_id]["delivered"] += 1
                delivered_normally += 1

            registered_age = now - lc["registered_at"]
            first_age = (now - lc["first_packet_at"]) if lc["first_packet_at"] else None
            last_age = (now - lc["last_packet_at"]) if lc["last_packet_at"] else None

            if is_missing:
                # Classify the missing-tag failure mode for this single tag.
                if lc["packet_count"] == 0:
                    failure_mode = "NEVER_DELIVERED"
                elif last_age is not None and last_age > 1.0:
                    failure_mode = "WENT_SILENT_AFTER_N_PACKETS"
                else:
                    # Packets are arriving but every recent one was rejected as
                    # stale by the timestamp check (frame counter not advancing).
                    failure_mode = "DELIVERING_BUT_STALE_FRAMES"

                print(
                    f"[SENSOR_FORENSICS] TIMEOUT_TAG instance={self._forensics_instance_id} "
                    f"tag={tag} type={sensor_type} ego_id={ego_id} actor_id={lc['sensor_actor_id']} "
                    f"state=MISSING mode={failure_mode} "
                    f"registered_age_s={registered_age:.2f} packets={lc['packet_count']} "
                    f"first_packet_age_s={first_age if first_age is None else f'{first_age:.2f}'} "
                    f"last_packet_age_s={last_age if last_age is None else f'{last_age:.2f}'} "
                    f"frame_first={lc['frame_first']} frame_last={lc['frame_last']} "
                    f"dropped_stale={lc['dropped_stale']} dropped_unregistered={lc['dropped_unregistered']}",
                    flush=True,
                )

        # Aggregate: which sensor TYPES are missing? (e.g. all rgb_front_*)
        type_missing = {}
        type_total = {}
        base_missing = {}
        base_total = {}
        for tag in required_tags:
            stype = self._sensor_types.get(tag, "?")
            type_total[stype] = type_total.get(stype, 0) + 1
            # Strip trailing "_<ego_id>" to get the sensor "base" name (rgb_front, lidar, etc.)
            base = tag.rsplit("_", 1)[0] if "_" in tag else tag
            base_total[base] = base_total.get(base, 0) + 1
            if tag in missing_tags:
                type_missing[stype] = type_missing.get(stype, 0) + 1
                base_missing[base] = base_missing.get(base, 0) + 1

        type_breakdown = {
            t: f"{type_missing.get(t, 0)}/{type_total[t]}" for t in sorted(type_total.keys())
        }
        base_breakdown = {
            b: f"{base_missing.get(b, 0)}/{base_total[b]}" for b in sorted(base_total.keys())
        }
        ego_breakdown = {
            (str(eid) if eid is not None else "noid"): f"{v['delivered']}/{v['total']}"
            for eid, v in sorted(per_ego.items(), key=lambda x: (x[0] is None, x[0]))
        }

        # Classify the OVERALL pattern so we know which branch of the decision tree to follow.
        n_missing = len(missing_tags)
        n_total = len(required_tags)
        affected_egos = [eid for eid, v in per_ego.items() if v["missing"] > 0]
        affected_types = [t for t, v in type_missing.items() if v > 0]
        affected_bases = [b for b, v in base_missing.items() if v > 0]

        # Sub-classify: how many of the missing tags NEVER delivered any packet
        # vs delivered some packets and then went silent.
        never = sum(1 for t in missing_tags if (self._sensor_lifecycle.get(t, {}).get("packet_count", 0) == 0))
        silent = n_missing - never

        if n_missing == n_total:
            if never == n_total:
                overall_pattern = "ALL_SENSORS_NEVER_DELIVERED_carla_did_not_render_or_streaming_failed_to_open"
            elif silent == n_total:
                overall_pattern = "ALL_SENSORS_WENT_SILENT_TOGETHER_carla_server_tick_hung"
            else:
                overall_pattern = f"ALL_SENSORS_MISSING_mixed_never={never}_silent={silent}"
        elif len(affected_egos) == 1 and per_ego[affected_egos[0]]["missing"] == per_ego[affected_egos[0]]["total"]:
            ego_pattern = "NEVER_DELIVERED" if never == n_missing else ("WENT_SILENT" if silent == n_missing else "MIXED")
            overall_pattern = f"ONE_EGO_ALL_SENSORS_MISSING_{ego_pattern}_ego{affected_egos[0]}_streaming_broken_or_ego_destroyed"
        elif len(affected_bases) == 1:
            overall_pattern = f"ONE_SENSOR_TYPE_MISSING_ACROSS_EGOS_listen_race_in_{affected_bases[0]}_never={never}_silent={silent}"
        elif never > 0 and silent == 0:
            overall_pattern = f"SCATTERED_NEVER_DELIVERED_{never}_tags_streaming_open_race"
        elif silent > 0 and never == 0:
            overall_pattern = f"SCATTERED_WENT_SILENT_{silent}_tags_streaming_send_hang"
        else:
            overall_pattern = f"SCATTERED_MIXED_never={never}_silent={silent}"

        print(
            f"[SENSOR_FORENSICS] TIMEOUT_SUMMARY instance={self._forensics_instance_id} "
            f"queue_timeout_s={self._queue_timeout} grace_used_s={grace_used_s:.2f} "
            f"missing={n_missing}/{n_total} delivered_normally={delivered_normally} "
            f"affected_egos={sorted([e for e in affected_egos if e is not None])} "
            f"affected_sensor_types={sorted(affected_types)} "
            f"affected_sensor_bases={sorted(affected_bases)} "
            f"ego_breakdown={ego_breakdown} "
            f"type_breakdown={type_breakdown} "
            f"base_breakdown={base_breakdown} "
            f"OVERALL_PATTERN={overall_pattern}",
            flush=True,
        )
