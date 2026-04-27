#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Labs.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Module to manipulate the routes, by making then more or less dense (Up to a certain parameter).
It also contains functions to convert the CARLA world location do GPS coordinates.
"""

import math
import xml.etree.ElementTree as ET

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.local_planner import RoadOption


def _location_to_gps(lat_ref, lon_ref, location):
    """
    Convert from world coordinates to GPS coordinates
    :param lat_ref: latitude reference for the current map
    :param lon_ref: longitude reference for the current map
    :param location: location to translate
    :return: dictionary with lat, lon and height
    """

    EARTH_RADIUS_EQUA = 6378137.0   # pylint: disable=invalid-name
    scale = math.cos(lat_ref * math.pi / 180.0)
    mx = scale * lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
    my = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))
    mx += location.x
    my -= location.y

    lon = mx * 180.0 / (math.pi * EARTH_RADIUS_EQUA * scale)
    lat = 360.0 * math.atan(math.exp(my / (EARTH_RADIUS_EQUA * scale))) / math.pi - 90.0
    z = location.z

    return {'lat': lat, 'lon': lon, 'z': z}


def location_route_to_gps(route, lat_ref, lon_ref):
    """
        Locate each waypoint of the route into gps, (lat long ) representations.
    :param route:
    :param lat_ref:
    :param lon_ref:
    :return:
    """
    gps_route = []

    for transform, connection in route:
        gps_point = _location_to_gps(lat_ref, lon_ref, transform.location)
        gps_route.append((gps_point, connection))

    return gps_route


def _get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def downsample_route(route, sample_factor):
    """
    Downsample the route by some factor.
    :param route: the trajectory , has to contain the waypoints and the road options
    :param sample_factor: Maximum distance between samples
    :return: returns the ids of the final route that can
    """

    ids_to_sample = []
    prev_option = None
    dist = 0

    for i, point in enumerate(route):
        curr_option = point[1]

        # Lane changing
        if curr_option in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # When road option changes
        elif prev_option != curr_option and prev_option not in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
            ids_to_sample.append(i)
            dist = 0

        # After a certain max distance
        elif dist > sample_factor:
            ids_to_sample.append(i)
            dist = 0

        # At the end
        elif i == len(route) - 1:
            ids_to_sample.append(i)
            dist = 0

        # Compute the distance traveled
        else:
            curr_location = point[0].location
            prev_location = route[i-1][0].location
            dist += curr_location.distance(prev_location)

        prev_option = curr_option

    return ids_to_sample


def interpolate_trajectory(world, waypoints_trajectory, hop_resolution=1.0):
    """
    Given some raw keypoints interpolate a full dense trajectory to be used by the user.
    returns the full interpolated route both in GPS coordinates and also in its original form.
    
    Args:
        - world: an reference to the CARLA world so we can use the planner
        - waypoints_trajectory: the current coarse trajectory
        - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
    """

    # Try CARLA 9.12+ API first, fall back to older DAO-based API
    try:
        grp = GlobalRoutePlanner(world.get_map(), hop_resolution)
    except TypeError:
        dao = GlobalRoutePlannerDAO(world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

    # ── Per-segment loop guard (mirrors route_manipulation.py) ───────────────
    _LOOP_RATIO   = 2.0
    _LOOP_SLACK_M = 5.0
    _DENSE_STEP_M = 3.0
    _wmap = world.get_map()

    def _seg_len(trace):
        total = 0.0
        prev = None
        for wp, _ in trace:
            loc = wp.transform.location if hasattr(wp, 'transform') else wp.location
            if prev is not None:
                total += math.hypot(loc.x - prev[0], loc.y - prev[1])
            prev = (loc.x, loc.y)
        return total

    def _densified_trace(a, b):
        d = math.hypot(b.x - a.x, b.y - a.y)
        n = max(2, int(math.ceil(d / _DENSE_STEP_M)) + 1)
        out = []
        for k in range(n - 1):
            t0 = k / (n - 1)
            t1 = (k + 1) / (n - 1)
            loc_a = type(a)(x=a.x + (b.x - a.x) * t0, y=a.y + (b.y - a.y) * t0, z=a.z + (b.z - a.z) * t0)
            loc_b = type(a)(x=a.x + (b.x - a.x) * t1, y=a.y + (b.y - a.y) * t1, z=a.z + (b.z - a.z) * t1)
            try:
                seg = grp.trace_route(loc_a, loc_b)
            except Exception:
                seg = []
            if not seg:
                try:
                    mid = type(a)(x=(loc_a.x + loc_b.x) / 2, y=(loc_a.y + loc_b.y) / 2, z=(loc_a.z + loc_b.z) / 2)
                    wp = _wmap.get_waypoint(mid, project_to_road=True)
                    seg = [(wp, RoadOption.LANEFOLLOW)]
                except Exception:
                    pass
            out.extend(seg)
        return out

    # Obtain route plan
    route_trace = []
    loop_fallback_count = 0
    for i in range(len(waypoints_trajectory) - 1):   # Goes until the one before the last.

        waypoint = waypoints_trajectory[i]
        waypoint_next = waypoints_trajectory[i + 1]
        straight = math.hypot(waypoint_next.x - waypoint.x, waypoint_next.y - waypoint.y)
        interpolated_trace = grp.trace_route(waypoint, waypoint_next)
        seg_len = _seg_len(interpolated_trace)
        if interpolated_trace and seg_len > (_LOOP_RATIO * straight) + _LOOP_SLACK_M:
            loop_fallback_count += 1
            interpolated_trace = _densified_trace(waypoint, waypoint_next)
        route_trace.extend(interpolated_trace)

    if loop_fallback_count:
        print(
            f"[interpolate_trajectory] loop_fallbacks={loop_fallback_count}/{len(waypoints_trajectory)-1} "
            f"segment(s) replaced with dense-hop re-routing"
        )

    route_before = [(wp_tuple[0].transform, wp_tuple[1]) for wp_tuple in route_trace]
    postprocess_meta = {}
    if hasattr(grp, "postprocess_route_trace"):
        try:
            route = grp.postprocess_route_trace(
                route_trace,
                enable_ucla_v2_smoothing=True,
                return_transforms=True,
            )
            postprocess_meta = dict(getattr(grp, "_last_postprocess_meta", {}) or {})
        except Exception:
            route = route_before
            postprocess_meta = {
                "status": "postprocess_exception",
                "applied": False,
            }
    else:
        route = route_before
        postprocess_meta = {
            "status": "postprocess_unavailable",
            "applied": False,
        }

    interpolate_trajectory.last_debug = {
        "before_route": route_before,
        "after_route": route,
        "postprocess_meta": postprocess_meta,
    }

    lat_ref, lon_ref = _get_latlon_ref(world)

    return location_route_to_gps(route, lat_ref, lon_ref), route
