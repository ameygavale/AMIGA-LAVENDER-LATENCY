"""Example using the track_follower service to drive a two-by-two alternating track with X/Y leg lengths."""
# Copyright (c) farm-ng, inc.
# License: Amiga Development Kit License
from __future__ import annotations

import argparse
import asyncio
from math import copysign, radians
from pathlib import Path

from farm_ng.core.event_client import EventClient
from farm_ng.core.event_service_pb2 import EventServiceConfigList
from farm_ng.core.event_service_pb2 import SubscribeRequest
from farm_ng.core.events_file_reader import proto_from_json_file
from farm_ng.core.uri_pb2 import Uri
from farm_ng.filter.filter_pb2 import FilterState
from farm_ng.track.track_pb2 import Track
from farm_ng.track.track_pb2 import TrackFollowerState
from farm_ng.track.track_pb2 import TrackFollowRequest
from farm_ng_core_pybind import Isometry3F64
from farm_ng_core_pybind import Pose3F64
from farm_ng_core_pybind import Rotation3F64
from google.protobuf.empty_pb2 import Empty


async def get_pose(clients: dict[str, EventClient]) -> Pose3F64:
    """Get the current pose of the robot in the world frame, from the filter service."""
    state: FilterState = await clients["filter"].request_reply("/get_state", Empty(), decode=True)
    print(f"Current filter state:\n{state}")
    return Pose3F64.from_proto(state.pose)


async def set_track(clients: dict[str, EventClient], track: Track) -> None:
    """Set the track of the track_follower."""
    print(f"Setting track:\n{track}")
    await clients["track_follower"].request_reply("/set_track", TrackFollowRequest(track=track))


async def start(clients: dict[str, EventClient]) -> None:
    """Request to start following the track."""
    print("Sending request to start following the track...")
    await clients["track_follower"].request_reply("/start", Empty())


async def build_two_by_two_track_xy(
    clients: dict[str, EventClient],
    x_length: float,
    y_length: float,
    clockwise: bool,
    num_legs: int,
) -> Track:
    """Build a 'two 90° turns one way, two 90° turns the other way' track with alternating X/Y leg lengths.

    Turn pattern (if clockwise=False i.e., left-first):  +90, +90, -90, -90, +90, +90, ...
                 (if clockwise=True  i.e., right-first): -90, -90, +90, +90, -90, -90, ...

    Leg lengths alternate: X, Y, X, Y, ...

    Args:
        clients: Event clients dict (uses filter + track_follower).
        x_length: Straight distance (meters) for 1st, 3rd, 5th, ... legs.
        y_length: Straight distance (meters) for 2nd, 4th, 6th, ... legs.
        clockwise: If True, first two turns are right; else first two turns are left.
        num_legs: Number of straight legs to execute (>=2). Example:
                  6 legs ⇒ waypoints A→B→C→D→E→F (5 turns).
    Returns:
        Track protobuf containing waypoints for the follower.
    """
    if num_legs < 2:
        num_legs = 2

    # Current world pose
    world_pose_robot: Pose3F64 = await get_pose(clients)

    track_waypoints: list[Pose3F64] = []

    base_angle = radians(90.0)
    first_sign = -1.0 if clockwise else 1.0
    turn_pattern = [first_sign, first_sign, -first_sign, -first_sign]  # repeat every 4 turns

    # Start waypoint
    world_pose_goal0: Pose3F64 = world_pose_robot * Pose3F64(
        a_from_b=Isometry3F64(), frame_a="robot", frame_b="goal0"
    )
    track_waypoints.append(world_pose_goal0)

    # Build legs
    for i in range(1, num_legs):
        # Alternate X/Y distances: i=1 -> X, i=2 -> Y, i=3 -> X, ...
        seg_len = x_length if (i % 2 == 1) else y_length

        # Straight to next goal i
        track_waypoints.extend(create_straight_segment(track_waypoints[-1], f"goal{i}", seg_len))

        # Turn between legs i and i+1 (skip after final straight)
        if i < num_legs - 1:
            sign = turn_pattern[(i - 1) % 4]
            angle = sign * base_angle
            track_waypoints.extend(create_turn_segment(track_waypoints[-1], f"turn{i}", angle))

    return format_track(track_waypoints)


def create_straight_segment(
    previous_pose: Pose3F64, next_frame_b: str, distance: float, spacing: float = 0.1
) -> list[Pose3F64]:
    """Compute a straight segment."""
    segment_poses: list[Pose3F64] = [previous_pose]

    counter: int = 0
    remaining_distance: float = distance

    while abs(remaining_distance) > 0.01:
        segment_distance: float = copysign(min(abs(remaining_distance), spacing), distance)

        straight_segment: Pose3F64 = Pose3F64(
            a_from_b=Isometry3F64([segment_distance, 0, 0], Rotation3F64.Rz(0)),
            frame_a=segment_poses[-1].frame_b,
            frame_b=f"{next_frame_b}_{counter}",
        )
        segment_poses.append(segment_poses[-1] * straight_segment)

        counter += 1
        remaining_distance -= segment_distance

    segment_poses[-1].frame_b = next_frame_b
    return segment_poses


def create_turn_segment(
    previous_pose: Pose3F64, next_frame_b: str, angle: float, spacing: float = 0.1
) -> list[Pose3F64]:
    """Compute an in-place yaw turn (±90°) at the junction."""
    segment_poses: list[Pose3F64] = [previous_pose]

    counter: int = 0
    remaining_angle: float = angle

    while abs(remaining_angle) > 0.01:
        segment_angle: float = copysign(min(abs(remaining_angle), spacing), angle)

        turn_segment: Pose3F64 = Pose3F64(
            a_from_b=Isometry3F64.Rz(segment_angle),
            frame_a=segment_poses[-1].frame_b,
            frame_b=f"{next_frame_b}_{counter}",
        )
        segment_poses.append(segment_poses[-1] * turn_segment)

        counter += 1
        remaining_angle -= segment_angle

    segment_poses[-1].frame_b = next_frame_b
    return segment_poses


def format_track(track_waypoints: list[Pose3F64]) -> Track:
    """Pack the track waypoints into a Track proto message."""
    return Track(waypoints=[pose.to_proto() for pose in track_waypoints])


async def start_track(
    clients: dict[str, EventClient],
    x_length: float,
    y_length: float,
    clockwise: bool,
    num_legs: int,
) -> None:
    """Build and start the two-by-two alternating track with X/Y leg lengths."""
    track: Track = await build_two_by_two_track_xy(clients, x_length, y_length, clockwise, num_legs)
    await set_track(clients, track)
    await start(clients)


async def stream_track_state(clients: dict[str, EventClient]) -> None:
    """Stream the track_follower state."""
    await asyncio.sleep(1.0)  # brief delay to let the follower start
    message: TrackFollowerState
    async for _, message in clients["track_follower"].subscribe(SubscribeRequest(uri=Uri(path="/state"))):
        print("###################")
        print(message)


async def run(args) -> None:
    # Create EventClients to the required services
    clients: dict[str, EventClient] = {}
    expected_configs = ["track_follower", "filter"]
    config_list = proto_from_json_file(args.service_config, EventServiceConfigList())
    for config in config_list.configs:
        if config.name in expected_configs:
            clients[config.name] = EventClient(config)

    for config in expected_configs:
        if config not in clients:
            raise RuntimeError(f"No {config} service config in {args.service_config}")

    # Resolve x/y lengths (keep --side-length as a fallback for convenience)
    x_len = args.x_length if args.x_length is not None else args.side_length
    y_len = args.y_length if args.y_length is not None else args.side_length

    tasks: list[asyncio.Task] = [
        asyncio.create_task(start_track(clients, x_len, y_len, args.clockwise, args.num_legs)),
        asyncio.create_task(stream_track_state(clients)),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Amiga track_follower two-by-two alternating track with X/Y leg lengths.",
    )
    parser.add_argument("--service-config", type=Path, required=True, help="The service config.")

    # New: X/Y leg lengths
    parser.add_argument("--x-length", type=float, default=None, help="Meters for legs 1,3,5,...")
    parser.add_argument("--y-length", type=float, default=None, help="Meters for legs 2,4,6,...")

    # Back-compat: if provided alone, both X and Y use this
    parser.add_argument("--side-length", type=float, default=5.0, help="Fallback length for both X and Y (default 5.0).")

    parser.add_argument(
        "--clockwise",
        action="store_true",
        help="If set, the first two 90° turns are right (CW). Default is left-first (CCW).",
    )
    parser.add_argument(
        "--num-legs",
        type=int,
        default=6,
        help="Number of straight legs to execute (>=2). Example: 6 ⇒ A→B→C→D→E→F.",
    )
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(run(args))
