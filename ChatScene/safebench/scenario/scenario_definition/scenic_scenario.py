import numpy as np
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.scenario.tools.scenario_utils import convert_transform_to_location
from safebench.scenario.scenario_definition.scenic.dynamic_scenic import DynamicScenic as scenario_scenic
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from typing import Optional

SECONDS_GIVEN_PER_METERS = 1
from safebench.scenario.scenario_definition.atomic_criteria import (
    Status,
    CollisionTest,
    DrivenDistanceTest,
    AverageVelocityTest,
    OffRoadTest,
    KeepLaneTest,
    InRouteTest,
    RouteCompletionTest,
    RunningRedLightTest,
    RunningStopTest,
    ActorSpeedAboveThresholdTest
)


class ScenicScenario():
    """
        Implementation of a ScenicScenario, i.e., a scenario that is controlled by scenic
    """

    @staticmethod
    def _infer_actor_kind(type_id: Optional[str]) -> str:
        """Derive a coarse actor kind from the CARLA blueprint type_id."""
        if not type_id:
            return "npc"
        type_id_lower = type_id.lower()
        if type_id_lower.startswith("static.") or type_id_lower.startswith("prop."):
            return "static"
        if type_id_lower.startswith("walker."):
            return "pedestrian"
        if "bike" in type_id_lower or "bicycle" in type_id_lower:
            return "bike"
        if type_id_lower.startswith("vehicle."):
            return "nonego"
        return "npc"

    def __init__(self, world, config, ego_id, logger, max_running_step):
        self.world = world
        self.logger = logger
        self.config = config
        self.ego_id = ego_id
        self.max_running_step = max_running_step
        self.timeout = 60

        self.route, self.ego_vehicle = self._update_route_and_ego()
        self.other_actors = []
        self.static_objects = []
        self.list_scenarios = [scenario_scenic(world, self.ego_vehicle, self.config, timeout=self.timeout)]
        self.criteria = self._create_criteria()
                
    def _update_route_and_ego(self, timeout=None):
        ego_vehicle = self.world.scenic.simulation.ego.carlaActor
        actor = ego_vehicle
        CarlaDataProvider._carla_actor_pool[actor.id] = actor
        CarlaDataProvider.register_actor(actor)       
        
        self.adv_actors = []
        self.static_objects = []
        self.logger.log(f">> Processing {len(self.world.scenic.simulation.objects)} scenic objects", color='cyan')
        for other_actor in self.world.scenic.simulation.objects:
            carla_actor = getattr(other_actor, 'carlaActor', None)
            if carla_actor is None or carla_actor.id == ego_vehicle.id:
                self.logger.log(f"   Skipping actor: carla_actor={carla_actor is not None}, is_ego={carla_actor.id == ego_vehicle.id if carla_actor else 'N/A'}", color='cyan')
                continue
            behavior = getattr(other_actor, 'behavior', None)
            self.logger.log(f"   Actor type_id={getattr(carla_actor, 'type_id', 'unknown')}, behavior={behavior}", color='cyan')
            if 'Adv' in str(behavior):
                adv_actor = carla_actor
                self.adv_actors.append(adv_actor)
                CarlaDataProvider._carla_actor_pool[adv_actor.id] = adv_actor
                CarlaDataProvider.register_actor(adv_actor)      
            else:
                self.static_objects.append(carla_actor)
                CarlaDataProvider._carla_actor_pool[carla_actor.id] = carla_actor
                CarlaDataProvider.register_actor(carla_actor)
        self.logger.log(f">> Found {len(self.adv_actors)} adv actors, {len(self.static_objects)} static objects", color='cyan')
        
        if len(self.config.trajectory) == 0:
            # coarse traj ##
            routeplanner = RoutePlanner(ego_vehicle, 200, [])

            _waypoint_buffer = []
            while len(_waypoint_buffer) < 50:
                pop = routeplanner._waypoints_queue.popleft()
                _waypoint_buffer.append(pop[0].transform.location)

            ### dense route planning ###
            route = interpolate_trajectory(self.world, _waypoint_buffer)
            index = 1
            prev_wp = route[0][0].location
            _accum_meters = 0
            while _accum_meters < 100:
                pop = route[index]
                wp = pop[0].location
                d = wp.distance(prev_wp)
                _accum_meters += d
                prev_wp = wp
                index += 1
            route = route[:index]
        else:
            route = interpolate_trajectory(self.world, self.config.trajectory)
            
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(route))
        CarlaDataProvider.set_scenario_config(self.config)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout(route) if timeout is None else timeout
        return route, ego_vehicle

    def _estimate_route_timeout(self, route):
        route_length = 0.0  # in meters
        min_length = 100.0

        if len(route) == 1:
            return int(SECONDS_GIVEN_PER_METERS * min_length)

        prev_point = route[0][0]
        for current_point, _ in route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point
        return int(SECONDS_GIVEN_PER_METERS * route_length)

    def initialize_actors(self):
        """
            Set other_actors to the superset of all scenario actors
        """
        pass

    def get_running_status(self, running_record):
        ego_velocity_vec = self.ego_vehicle.get_velocity()
        ego_acc_vec = self.ego_vehicle.get_acceleration()
        ego_transform = CarlaDataProvider.get_transform(self.ego_vehicle)
        running_status = {
            'ego_velocity': CarlaDataProvider.get_velocity(self.ego_vehicle),
            'ego_velocity_x': ego_velocity_vec.x,
            'ego_velocity_y': ego_velocity_vec.y,
            'ego_velocity_z': ego_velocity_vec.z,
            'ego_acceleration_x': ego_acc_vec.x,
            'ego_acceleration_y': ego_acc_vec.y,
            'ego_acceleration_z': ego_acc_vec.z,
            'ego_x': ego_transform.location.x,
            'ego_y': ego_transform.location.y,
            'ego_z': ego_transform.location.z,
            'ego_roll': ego_transform.rotation.roll,
            'ego_pitch': ego_transform.rotation.pitch,
            'ego_yaw': ego_transform.rotation.yaw,
            'current_game_time': GameTime.get_time()
        }
        adv_actor_status = {}
        for i, adv_actor in enumerate(self.adv_actors):
            adv_velocity_vec = adv_actor.get_velocity()
            adv_acc_vec = adv_actor.get_acceleration()
            adv_transform = CarlaDataProvider.get_transform(adv_actor)
            adv_actor_status[f'adv_agent_{i}'] = {
                'actor_type': self._infer_actor_kind(getattr(adv_actor, "type_id", None)),
                'vehicle_model': getattr(adv_actor, "type_id", None),  # exact CARLA blueprint e.g. vehicle.kawasaki.ninja
                'velocity': CarlaDataProvider.get_velocity(adv_actor),
                'velocity_x': adv_velocity_vec.x,
                'velocity_y': adv_velocity_vec.y,
                'velocity_z': adv_velocity_vec.z,
                'acceleration_x': adv_acc_vec.x,
                'acceleration_y': adv_acc_vec.y,
                'acceleration_z': adv_acc_vec.z,
                'x': adv_transform.location.x,
                'y': adv_transform.location.y,
                'z': adv_transform.location.z,
                'roll': adv_transform.rotation.roll,
                'pitch': adv_transform.rotation.pitch,
                'yaw': adv_transform.rotation.yaw,
            }
        running_status.update(adv_actor_status)
        
        static_actor_status = {}
        for i, static_actor in enumerate(self.static_objects):
            try:
                if hasattr(static_actor, "is_alive") and not static_actor.is_alive:
                    continue
                static_velocity = static_actor.get_velocity()
                static_acc = static_actor.get_acceleration()
                static_transform = CarlaDataProvider.get_transform(static_actor)
            except (RuntimeError, ReferenceError):
                # Actor may already be destroyed; skip logging to keep run alive
                continue
            base_kind = self._infer_actor_kind(getattr(static_actor, "type_id", None))
            if base_kind == "static":
                actor_type = "static_prop"
            elif base_kind:
                actor_type = f"static_{base_kind}"
            else:
                actor_type = "static"
            static_actor_status[f'static_prop_{i}'] = {
                'actor_type': actor_type,
                'vehicle_model': getattr(static_actor, "type_id", None),  # exact CARLA blueprint
                'velocity': CarlaDataProvider.get_velocity(static_actor),
                'velocity_x': static_velocity.x,
                'velocity_y': static_velocity.y,
                'velocity_z': static_velocity.z,
                'acceleration_x': static_acc.x,
                'acceleration_y': static_acc.y,
                'acceleration_z': static_acc.z,
                'x': static_transform.location.x,
                'y': static_transform.location.y,
                'z': static_transform.location.z,
                'roll': static_transform.rotation.roll,
                'pitch': static_transform.rotation.pitch,
                'yaw': static_transform.rotation.yaw,
            }
        running_status.update(static_actor_status)
            
        for criterion_name, criterion in self.criteria.items():
            running_status[criterion_name] = criterion.update()

        stop = False
        # collision with other objects
        if running_status['collision'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to collision', color='yellow')

        # out of the road detection
        if running_status['off_road'] == Status.FAILURE:
            stop = True
            self.logger.log('>> Scenario stops due to off road', color='yellow')

        # only check when evaluating
        if self.config.scenario_id != 0:  
            # route completed
            if running_status['route_complete'] == 100:
                stop = True
                self.logger.log('>> Scenario stops due to route completion', color='yellow')

        # stop at max step
        if len(running_record) >= self.max_running_step: 
            stop = True
            self.logger.log('>> Scenario stops due to max steps', color='yellow')

        for scenario in self.list_scenarios:
            # only check when evaluating
            if self.config.scenario_id != 0:  
                if running_status['driven_distance'] >= scenario.ego_max_driven_distance:
                    stop = True
                    self.logger.log('>> Scenario stops due to max driven distance', color='yellow')
                    break
            if running_status['current_game_time'] >= scenario.timeout:
                stop = True
                self.logger.log('>> Scenario stops due to timeout', color='yellow') 
                break
            if scenario.check_scenic_terminate():
                reason = getattr(self.world.scenic, 'last_termination_reason', None)
                termination_type = getattr(self.world.scenic, 'last_termination_type', None)
                extra = f' (reason={reason}, type={termination_type})' if reason or termination_type else ''
                self.logger.log(f'>> Scenario stops due to scenic termination{extra}', color='yellow') 
                stop = True
                break
        return running_status, stop

    def _create_criteria(self):
        criteria = {}
        route = convert_transform_to_location(self.route)

        criteria['driven_distance'] = DrivenDistanceTest(actor=self.ego_vehicle, distance_success=1e4, distance_acceptable=1e4, optional=True)
        criteria['average_velocity'] = AverageVelocityTest(actor=self.ego_vehicle, avg_velocity_success=1e4, avg_velocity_acceptable=1e4, optional=True)
        criteria['lane_invasion'] = KeepLaneTest(actor=self.ego_vehicle, optional=True)
        criteria['off_road'] = OffRoadTest(actor=self.ego_vehicle, optional=True)
        criteria['collision'] = CollisionTest(actor=self.ego_vehicle, terminate_on_failure=True)
        criteria['run_red_light'] = RunningRedLightTest(actor=self.ego_vehicle)
        criteria['run_stop'] = RunningStopTest(actor=self.ego_vehicle)
        if self.config.scenario_id != 0:  # only check when evaluating
            criteria['distance_to_route'] = InRouteTest(self.ego_vehicle, route=route, offroad_max=30)
            criteria['route_complete'] = RouteCompletionTest(self.ego_vehicle, route=route)
        return criteria

    @staticmethod
    def _get_actor_state(actor):
        actor_trans = actor.get_transform()
        actor_x = actor_trans.location.x
        actor_y = actor_trans.location.y
        actor_yaw = actor_trans.rotation.yaw / 180 * np.pi
        yaw = np.array([np.cos(actor_yaw), np.sin(actor_yaw)])
        velocity = actor.get_velocity()
        acc = actor.get_acceleration()
        return [actor_x, actor_y, actor_yaw, yaw[0], yaw[1], velocity.x, velocity.y, acc.x, acc.y]

    def update_info(self):
        ego_state = self._get_actor_state(self.ego_vehicle)
        actor_info = [ego_state]
        for s_i in self.list_scenarios:
            for a_i in s_i.other_actors:
                actor_state = self._get_actor_state(a_i)
                actor_info.append(actor_state)

        actor_info = np.array(actor_info)
        # get the info of the ego vehicle and the other actors
        return {
            'actor_info': actor_info
        }

    def clean_up(self):
        # stop criterion and destroy sensors
        for _, criterion in self.criteria.items():
            criterion.terminate()
        # clean all actors
        self.world.scenic.endSimulation()
