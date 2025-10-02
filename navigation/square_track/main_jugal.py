from __future__ import annotations

import argparse
import asyncio
from math import copysign, radians, sin, cos
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
    print(f"Setting track:\n{track}")
    await clients["track_follower"].request_reply("/set_track", TrackFollowRequest(track=track))


async def start(clients: dict[str, EventClient]) -> None:
    print("Sending request to start following the track...")
    await clients["track_follower"].request_reply("/start", Empty())


async def build_square(
    clients: dict[str, EventClient], side_length: float, clockwise: bool, turn_radius: float = 0.5
) -> Track:
    world_pose_robot: Pose3F64 = await get_pose(clients)
    track_waypoints: list[Pose3F64] = []

    angle: float = radians(-90) if clockwise else radians(90)

    world_pose_goal0: Pose3F64 = world_pose_robot * Pose3F64(
        a_from_b=Isometry3F64(), frame_a="robot", frame_b="goal0"
    )
    track_waypoints.append(world_pose_goal0)

    track_waypoints.extend(create_straight_segment(track_waypoints[-1], "goal1", side_length))
    track_waypoints.extend(create_turn_segment_arc(track_waypoints[-1], "goal2", angle, radius=turn_radius))

    track_waypoints.extend(create_straight_segment(track_waypoints[-1], "goal3", side_length))
    track_waypoints.extend(create_turn_segment_arc(track_waypoints[-1], "goal4", angle, radius=turn_radius))

    track_waypoints.extend(create_straight_segment(track_waypoints[-1], "goal5", side_length))
    track_waypoints.extend(create_turn_segment_arc(track_waypoints[-1], "goal6", angle, radius=turn_radius))

    track_waypoints.extend(create_straight_segment(track_waypoints[-1], "goal7", side_length))
    track_waypoints.extend(create_turn_segment_arc(track_waypoints[-1], "goal8", angle, radius=turn_radius))

    return format_track(track_waypoints)


def create_straight_segment(
    previous_pose: Pose3F64, next_frame_b: str, distance: float, spacing: float = 0.1
) -> list[Pose3F64]:
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


def create_turn_segment_arc(
    previous_pose: Pose3F64,
    next_frame_b: str,
    angle: float,
    radius: float = 0.5,
    angle_spacing: float = 0.1,
) -> list[Pose3F64]:
    segment_poses: list[Pose3F64] = [previous_pose]
    counter = 0
    remaining_angle = angle

    while abs(remaining_angle) > 0.01:
        delta_angle = copysign(min(abs(remaining_angle), angle_spacing), angle)

        dx = radius * sin(delta_angle)
        dy = radius * (1 - cos(delta_angle)) * (-1 if angle < 0 else 1)
        dtheta = delta_angle

        arc_pose = Pose3F64(
            a_from_b=Isometry3F64([dx, dy, 0], Rotation3F64.Rz(dtheta)),
            frame_a=segment_poses[-1].frame_b,
            frame_b=f"{next_frame_b}_{counter}",
        )
        segment_poses.append(segment_poses[-1] * arc_pose)

        counter += 1
        remaining_angle -= delta_angle

    segment_poses[-1].frame_b = next_frame_b
    return segment_poses


def format_track(track_waypoints: list[Pose3F64]) -> Track:
    return Track(waypoints=[pose.to_proto() for pose in track_waypoints])


async def start_track(clients: dict[str, EventClient], side_length: float, clockwise: bool) -> None:
    track: Track = await build_square(clients, side_length, clockwise, turn_radius=0.5)
    await set_track(clients, track)
    await start(clients)


async def stream_track_state(clients: dict[str, EventClient]) -> None:
    await asyncio.sleep(1.0)

    async for _, message in clients["track_follower"].subscribe(SubscribeRequest(uri=Uri(path="/state"))):
        print("###################")
        print(message)


async def run(args) -> None:
    clients: dict[str, EventClient] = {}
    expected_configs = ["track_follower", "filter"]
    config_list = proto_from_json_file(args.service_config, EventServiceConfigList())
    for config in config_list.configs:
        if config.name in expected_configs:
            clients[config.name] = EventClient(config)

    for config in expected_configs:
        if config not in clients:
            raise RuntimeError(f"No {config} service config in {args.service_config}")

    tasks: list[asyncio.Task] = [
        asyncio.create_task(start_track(clients, args.side_length, args.clockwise)),
        asyncio.create_task(stream_track_state(clients)),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="python main.py", description="Amiga track_follower square example.")
    parser.add_argument("--service-config", type=Path, required=True, help="The service config.")
    parser.add_argument("--side-length", type=float, default=2.0, help="The side length of the square.")
    parser.add_argument(
        "--clockwise",
        action="store_true",
        help="Set to drive the square clockwise (right hand turns). Default is counter-clockwise (left hand turns).",
    )
    args = parser.parse_args()
    asyncio.run(run(args))
