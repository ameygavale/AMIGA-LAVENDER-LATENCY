#!/usr/bin/env python3
"""Custom path for Amiga robot: three parallel lanes with two 180° U-turns."""

import argparse
import asyncio
from math import radians, sin, cos, ceil, pi
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


def correct_heading_at_point(pose: Pose3F64, target_yaw: float, frame_name: str) -> Pose3F64:
    """Create a new pose with corrected heading while maintaining position."""
    # Extract position from the pose
    iso = pose.a_from_b
    position = [iso.translation[0], iso.translation[1], iso.translation[2]]
    
    # Create new pose with exact target heading
    corrected_pose = Pose3F64(
        a_from_b=Isometry3F64(position, Rotation3F64.Rz(target_yaw)),
        frame_a=pose.frame_a,
        frame_b=frame_name,
    )
    return corrected_pose


def get_yaw_from_pose(pose: Pose3F64) -> float:
    """Extract yaw angle from a pose."""
    rotation = pose.a_from_b.rotation
    yaw = rotation.log()[2]  # Extract yaw from rotation
    return yaw


async def build_path(
    clients: dict[str, EventClient],
    straight_spacing: float,
    arc_spacing: float,
) -> Track:
    """a1->b1->c1->d1->e1->f1 path with two 180° anti-clockwise U-turns.
    
    Three parallel lanes connected by U-turns:
    - a1 -> b1: 15m straight
    - b1 -> c1: 180° U-turn (5.5m lateral)
    - c1 -> d1: 15m straight (parallel to a1->b1)
    - d1 -> e1: 180° U-turn (2.75m lateral)
    - e1 -> f1: 15m straight (parallel to previous lanes)
    """
    world_pose_robot: Pose3F64 = await get_pose(clients)
    waypoints: list[Pose3F64] = [world_pose_robot]
    
    # Store the initial heading (yaw) for later verification
    initial_yaw = get_yaw_from_pose(world_pose_robot)
    print(f"Initial heading at a1 (yaw): {initial_yaw:.4f} rad = {initial_yaw * 180 / pi:.2f} degrees")
    print("=" * 70)
    
    # Segment 1: a1 -> b1
    print("a1->b1: 15m straight")
    waypoints += create_straight_segment(waypoints[-1], "point_b1", 15.0, spacing=straight_spacing)
    
    # U-turn 1: b1 -> c1 (5.5m lateral spacing)
    print("\nTurn at b1: 180° anti-clockwise U-turn to reach c1 (5.5m lateral displacement)")
    turn_radius_1 = 5.5 / 2.0  # radius = 2.75m
    print(f"U-turn radius: {turn_radius_1}m")
    arc_waypoints_1 = create_arc_segment(
        waypoints[-1], 
        "point_c1_before_correction", 
        turn_radius_1, 
        radians(180),  # Positive = left/anti-clockwise U-turn
        spacing=arc_spacing
    )
    waypoints += arc_waypoints_1
    
    # Correct heading at c1 to be exactly parallel (opposite direction)
    target_yaw_at_c1 = initial_yaw + pi
    print(f"Correcting heading at c1 to: {target_yaw_at_c1:.4f} rad = {target_yaw_at_c1 * 180 / pi:.2f} degrees")
    c1_corrected = correct_heading_at_point(waypoints[-1], target_yaw_at_c1, "point_c1")
    waypoints[-1] = c1_corrected
    
    # Segment 2: c1 -> d1
    print("\nc1->d1: 15m straight (parallel to a1->b1)")
    waypoints += create_straight_segment(waypoints[-1], "point_d1", 15.0, spacing=straight_spacing)
    
    # U-turn 2: d1 -> e1 (2.75m lateral spacing)
    print("\nTurn at d1: 180° anti-clockwise U-turn to reach e1 (2.75m lateral displacement)")
    turn_radius_2 = 2.75 / 2.0  # radius = 1.375m
    print(f"U-turn radius: {turn_radius_2}m")
    arc_waypoints_2 = create_arc_segment(
        waypoints[-1], 
        "point_e1_before_correction", 
        turn_radius_2, 
        radians(180),  # Positive = left/anti-clockwise U-turn
        spacing=arc_spacing
    )
    waypoints += arc_waypoints_2
    
    # Correct heading at e1 to match original heading at a1
    target_yaw_at_e1 = initial_yaw  # Back to original direction after two 180° turns
    print(f"Correcting heading at e1 to: {target_yaw_at_e1:.4f} rad = {target_yaw_at_e1 * 180 / pi:.2f} degrees")
    e1_corrected = correct_heading_at_point(waypoints[-1], target_yaw_at_e1, "point_e1")
    waypoints[-1] = e1_corrected
    
    # Verify e1 heading matches a1 heading
    e1_yaw = get_yaw_from_pose(waypoints[-1])
    print(f"\n✓ VERIFICATION: Heading at e1: {e1_yaw:.4f} rad = {e1_yaw * 180 / pi:.2f} degrees")
    print(f"✓ VERIFICATION: Heading at a1: {initial_yaw:.4f} rad = {initial_yaw * 180 / pi:.2f} degrees")
    heading_diff = abs(e1_yaw - initial_yaw)
    print(f"✓ VERIFICATION: Heading difference: {heading_diff:.6f} rad = {heading_diff * 180 / pi:.4f} degrees")
    if heading_diff < 0.001:  # Within 0.001 radians (~0.057 degrees)
        print("✓ SUCCESS: e1 heading matches a1 heading! Paths are perfectly parallel.")
    else:
        print(f"⚠ WARNING: Heading mismatch of {heading_diff * 180 / pi:.4f} degrees")
    
    # Segment 3: e1 -> f1
    print("\ne1->f1: 15m straight (parallel to a1->b1 and c1->d1)")
    waypoints += create_straight_segment(waypoints[-1], "point_f1", 15.0, spacing=straight_spacing)
    
    print("=" * 70)
    print(f"Total waypoints: {len(waypoints)}")
    print(f"Path summary:")
    print(f"  - Lane 1 (a1->b1): 15m")
    print(f"  - U-turn 1 (b1->c1): 5.5m lateral, radius {turn_radius_1}m")
    print(f"  - Lane 2 (c1->d1): 15m")
    print(f"  - U-turn 2 (d1->e1): 2.75m lateral, radius {turn_radius_2}m")
    print(f"  - Lane 3 (e1->f1): 15m")
    
    return Track(waypoints=[p.to_proto() for p in waypoints])


async def stream_track_state(clients: dict[str, EventClient]) -> None:
    await asyncio.sleep(1.0)
    async for _, msg in clients["track_follower"].subscribe(SubscribeRequest(uri=Uri(path="/state"))):
        print(f"Track follower state: {msg.state}")
        if msg.state == TrackFollowerState.State.GOAL_REACHED:
            print("Goal reached!")
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
    parser = argparse.ArgumentParser(description="Amiga three-lane parallel path with two U-turns")
    parser.add_argument("--service-config", type=Path, required=True, help="EventServiceConfigList JSON")
    parser.add_argument("--straight-spacing", type=float, default=0.05, help="Waypoint spacing on straight segments (m)")
    parser.add_argument("--arc-spacing", type=float, default=0.1, help="Arc chord spacing along the arc length (m)")
    args = parser.parse_args()
    asyncio.run(run(args))
