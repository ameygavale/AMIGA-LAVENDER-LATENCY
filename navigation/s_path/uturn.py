#!/usr/bin/env python3
"""Custom path for Amiga robot: straight -> 180° anti-clockwise U-turn -> straight."""

import argparse
import asyncio
from math import radians, sin, cos, ceil
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfigList, SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import Track, TrackFollowRequest, TrackFollowerState
from farm_ng_core_pybind import Isometry3F64, Pose3F64, Rotation3F64
from google.protobuf.empty_pb2 import Empty


async def get_pose(clients: dict[str, EventClient]) -> Pose3F64:
    state: FilterState = await clients["filter"].request_reply("/get_state", Empty(), decode=True)
    print(f"Current filter state:\n{state}")
    return Pose3F64.from_proto(state.pose)


async def set_track(clients: dict[str, EventClient], track: Track) -> None:
    print(f"Setting track with {len(track.waypoints)} waypoints")
    await clients["track_follower"].request_reply("/set_track", TrackFollowRequest(track=track))


async def start(clients: dict[str, EventClient]) -> None:
    print("Starting track following...")
    await clients["track_follower"].request_reply("/start", Empty())


def create_straight_segment(
    previous_pose: Pose3F64, next_frame_b: str, distance: float, spacing: float = 0.1
) -> list[Pose3F64]:
    """Discretize a straight line in the robot's current x-direction."""
    segment_poses: list[Pose3F64] = []
    steps = max(1, ceil(abs(distance) / spacing))
    step = distance / steps
    for i in range(1, steps + 1):
        dx = i * step
        waypoint = Pose3F64(
            a_from_b=Isometry3F64([dx, 0.0, 0.0], Rotation3F64.Rz(0.0)),
            frame_a=previous_pose.frame_b,
            frame_b=f"{next_frame_b}_{i}" if i < steps else next_frame_b,
        )
        segment_poses.append(previous_pose * waypoint)
    return segment_poses


def create_arc_segment(
    previous_pose: Pose3F64,
    next_frame_b: str,
    radius: float,
    angle_rad: float,
    spacing: float = 0.1,
) -> list[Pose3F64]:
    """Discretize a constant-radius circular arc in the local frame.

    Positive angle => left turn (anti-clockwise). Negative angle => right turn (clockwise).
    The robot moves forward while its heading rotates by angle_rad.
    """
    assert radius > 0.0, "radius must be > 0"
    arc_len = abs(angle_rad) * radius
    steps = max(1, ceil(arc_len / spacing))
    poses: list[Pose3F64] = []
    for i in range(1, steps + 1):
        theta = angle_rad * (i / steps)  # incremental heading change
        # Circular arc in local frame starting aligned with +x
        x = radius * sin(theta)
        y = radius * (1.0 - cos(theta)) * (1.0 if angle_rad >= 0.0 else -1.0)
        waypoint = Pose3F64(
            a_from_b=Isometry3F64([x, y, 0.0], Rotation3F64.Rz(theta)),
            frame_a=previous_pose.frame_b,
            frame_b=f"{next_frame_b}_{i}" if i < steps else next_frame_b,
        )
        poses.append(previous_pose * waypoint)
    return poses


async def build_path(
    clients: dict[str, EventClient],
    straight_spacing: float,
    arc_spacing: float,
) -> Track:
    """a1->b1->c1->d1 path with 180° anti-clockwise U-turn.
    
    a1 -> 15m straight to b1
    180° anti-clockwise U-turn from b1 to c1 (3m lateral displacement)
    -> 15m straight from c1 to d1 (parallel to a1->b1)
    """
    world_pose_robot: Pose3F64 = await get_pose(clients)
    waypoints: list[Pose3F64] = [world_pose_robot]
    
    print("a1->b1: 15m straight")
    waypoints += create_straight_segment(waypoints[-1], "point_b1", 15.0, spacing=straight_spacing)
    
    print("Turn at b1: 180° anti-clockwise U-turn to reach c1 (3m lateral displacement)")
    # For 180° arc with 3m separation, radius = 1.5m
    turn_radius = 1.5
    waypoints += create_arc_segment(
        waypoints[-1], 
        "point_c1", 
        turn_radius, 
        radians(180),  # Positive = left/anti-clockwise U-turn
        spacing=arc_spacing
    )
    
    print("c1->d1: 15m straight (parallel to a1->b1)")
    waypoints += create_straight_segment(waypoints[-1], "point_d1", 15.0, spacing=straight_spacing)
    
    print(f"Total waypoints: {len(waypoints)}")
    return Track(waypoints=[p.to_proto() for p in waypoints])


async def stream_track_state(clients: dict[str, EventClient]) -> None:
    await asyncio.sleep(1.0)
    async for _, msg in clients["track_follower"].subscribe(SubscribeRequest(uri=Uri(path="/state"))):
        print(f"Track follower state: {msg.state}")
        if msg.state == TrackFollowerState.State.GOAL_REACHED:
            print("Goal reached.")
            break


async def run(args) -> None:
    clients: dict[str, EventClient] = {}
    expected = ["track_follower", "filter"]
    cfg_list = proto_from_json_file(args.service_config, EventServiceConfigList())
    for cfg in cfg_list.configs:
        if cfg.name in expected:
            clients[cfg.name] = EventClient(cfg)
    for name in expected:
        if name not in clients:
            raise RuntimeError(f"No {name} service config in {args.service_config}")

    track = await build_path(
        clients,
        straight_spacing=args.straight_spacing,
        arc_spacing=args.arc_spacing,
    )
    await set_track(clients, track)
    await start(clients)
    await stream_track_state(clients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amiga path: 15m straight -> 180° anti-clockwise U-turn -> 15m straight")
    parser.add_argument("--service-config", type=Path, required=True, help="EventServiceConfigList JSON")
    parser.add_argument("--straight-spacing", type=float, default=0.05, help="Waypoint spacing on straight segments (m)")
    parser.add_argument("--arc-spacing", type=float, default=0.1, help="Arc chord spacing along the arc length (m)")
    args = parser.parse_args()
    asyncio.run(run(args))
