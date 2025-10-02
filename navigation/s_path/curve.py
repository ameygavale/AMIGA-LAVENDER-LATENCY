#!/usr/bin/env python3
"""Custom S-shaped path for Amiga robot navigation."""

import argparse
import asyncio
from math import radians
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
    """Get the current pose of the robot in the world frame."""
    state: FilterState = await clients["filter"].request_reply("/get_state", Empty(), decode=True)
    print(f"Current filter state:\n{state}")
    return Pose3F64.from_proto(state.pose)


async def set_track(clients: dict[str, EventClient], track: Track) -> None:
    """Set the track for the track_follower."""
    print(f"Setting track with {len(track.waypoints)} waypoints")
    await clients["track_follower"].request_reply("/set_track", TrackFollowRequest(track=track))


async def start(clients: dict[str, EventClient]) -> None:
    """Start following the track."""
    print("Starting track following...")
    await clients["track_follower"].request_reply("/start", Empty())


def create_straight_segment(
    previous_pose: Pose3F64, next_frame_b: str, distance: float, spacing: float = 0.1
) -> list[Pose3F64]:
    """Create a straight segment with waypoints."""
    segment_poses: list[Pose3F64] = []
    num_waypoints = int(abs(distance) / spacing)
    
    for i in range(num_waypoints + 1):
        segment_distance = i * spacing if distance > 0 else -i * spacing
        waypoint = Pose3F64(
            a_from_b=Isometry3F64([segment_distance, 0, 0], Rotation3F64.Rz(0)),
            frame_a=previous_pose.frame_b,
            frame_b=f"{next_frame_b}_{i}" if i < num_waypoints else next_frame_b
        )
        segment_poses.append(previous_pose * waypoint)
    
    return segment_poses[1:]  # Skip first pose (it's the previous pose)


def create_turn_segment(
    previous_pose: Pose3F64, next_frame_b: str, angle: float, radius: float = 1.0, spacing: float = 0.05
) -> list[Pose3F64]:
    """Create a smoother turn by following a circular arc instead of rotating in place.
    
    Args:
        previous_pose: The last pose before starting the turn.
        next_frame_b: The name of the next frame.
        angle: Total angle to turn (in radians). Positive is left turn, negative is right.
        radius: The turning radius (in meters).
        spacing: Angular spacing between waypoints (in radians).
    
    Returns:
        A list of poses along the arc (excluding the starting pose).
    """
    segment_poses: list[Pose3F64] = []
    num_waypoints = int(abs(angle) / spacing)

    for i in range(1, num_waypoints + 1):
        theta = i * spacing if angle > 0 else -i * spacing

        # Arc position (x = R*sin(theta), y = R*(1 - cos(theta))) for left turn
        x = radius * (1 - math.cos(theta))
        y = radius * math.sin(theta) if angle > 0 else -radius * math.sin(-theta)

        # Rotation at that point
        rot = Rotation3F64.Rz(theta)

        # Pose relative to starting pose
        arc_pose = Pose3F64(
            a_from_b=Isometry3F64([x, y, 0], rot),
            frame_a=previous_pose.frame_b,
            frame_b=f"{next_frame_b}_{i}"
        )

        # Transform into world frame
        segment_poses.append(previous_pose * arc_pose)

    return segment_poses



async def build_s_path(clients: dict[str, EventClient]) -> Track:
    """Build the S-shaped path from current position.
    
    Path structure:
    A (start) -> B (10m straight)
    B -> turn left 90° -> C (5m straight)  
    C -> turn left 90° -> D (10m straight, parallel to AB)
    D -> turn right 90° -> E (5m straight)
    E -> turn right 90° -> F (10m straight, parallel to CD)
    """
    
    # Get current pose (Point A)
    world_pose_robot: Pose3F64 = await get_pose(clients)
    
    # Initialize track with starting pose
    track_waypoints: list[Pose3F64] = [world_pose_robot]
    
    # A to B: 10m straight
    print("Building segment A->B (10m straight)")
    track_waypoints.extend(
        create_straight_segment(track_waypoints[-1], "point_B", 10.0)
    )
    
    # B: Turn left (anti-clockwise) 90°
    print("Building turn at B (left 90°)")
    track_waypoints.extend(
        create_turn_segment(track_waypoints[-1], "point_B_turned", radians(90))
    )
    
    # B to C: 5m straight
    print("Building segment B->C (5m straight)")
    track_waypoints.extend(
        create_straight_segment(track_waypoints[-1], "point_C", 5.0)
    )
    
    # C: Turn left (anti-clockwise) 90°
    print("Building turn at C (left 90°)")
    track_waypoints.extend(
        create_turn_segment(track_waypoints[-1], "point_C_turned", radians(90))
    )
    
    # C to D: 10m straight (parallel to AB but opposite direction)
    print("Building segment C->D (10m straight)")
    track_waypoints.extend(
        create_straight_segment(track_waypoints[-1], "point_D", 10.0)
    )
    
    # D: Turn right (clockwise) 90°
    print("Building turn at D (right 90°)")
    track_waypoints.extend(
        create_turn_segment(track_waypoints[-1], "point_D_turned", radians(-90))
    )
    
    # D to E: 5m straight
    print("Building segment D->E (5m straight)")
    track_waypoints.extend(
        create_straight_segment(track_waypoints[-1], "point_E", 5.0)
    )
    
    # E: Turn right (clockwise) 90°
    print("Building turn at E (right 90°)")
    track_waypoints.extend(
        create_turn_segment(track_waypoints[-1], "point_E_turned", radians(-90))
    )
    
    # E to F: 10m straight (parallel to AB and CD)
    print("Building segment E->F (10m straight)")
    track_waypoints.extend(
        create_straight_segment(track_waypoints[-1], "point_F", 10.0)
    )
    
    print(f"Total waypoints: {len(track_waypoints)}")
    
    # Convert to Track proto
    return Track(waypoints=[pose.to_proto() for pose in track_waypoints])


async def stream_track_state(clients: dict[str, EventClient]) -> None:
    """Stream the track_follower state."""
    await asyncio.sleep(1.0)
    
    message: TrackFollowerState
    async for _, message in clients["track_follower"].subscribe(SubscribeRequest(uri=Uri(path="/state"))):
        print(f"Track follower state: {message.state}")
        if message.state == TrackFollowerState.State.GOAL_REACHED:
            print("Goal reached! Path complete.")
            break


async def run(args) -> None:
    """Main execution function."""
    # Create EventClients
    clients: dict[str, EventClient] = {}
    expected_configs = ["track_follower", "filter"]
    config_list = proto_from_json_file(args.service_config, EventServiceConfigList())
    
    for config in config_list.configs:
        if config.name in expected_configs:
            clients[config.name] = EventClient(config)
    
    # Verify all required services
    for config in expected_configs:
        if config not in clients:
            raise RuntimeError(f"No {config} service config in {args.service_config}")
    
    # Build and execute the S-path
    track = await build_s_path(clients)
    await set_track(clients, track)
    await start(clients)
    
    # Monitor progress
    await stream_track_state(clients)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amiga S-shaped path navigation")
    parser.add_argument("--service-config", type=Path, required=True, help="The service config file")
    args = parser.parse_args()
    
    asyncio.run(run(args))
