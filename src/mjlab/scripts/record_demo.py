"""Script to record a demo and export it as a .viser file for GitHub Pages.

This script runs a demo with a pretrained policy and exports the visualization
as a .viser file that can be embedded in static webpages for GitHub Pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
import viser

from mjlab.scripts.gcs import ensure_default_checkpoint, ensure_default_motion
from mjlab.scripts.play import PlayConfig, create_env_and_policy
from mjlab.sim.sim import Simulation
from mjlab.viewer.viser_scene import ViserMujocoScene


@dataclass
class RecordConfig:
  """Configuration for recording demos."""

  output_dir: Path = Path("docs/recordings")
  """Directory to save .viser recordings."""

  output_name: str = "demo"
  """Base name for the output file (will add .viser extension)."""

  num_steps: int = 500
  """Number of simulation steps to record."""

  num_envs: int = 8
  """Number of parallel environments to visualize."""

  frame_skip: int = 2
  """Only record every Nth frame (1 = record all frames)."""

  sleep_duration: float = 0.016
  """Sleep duration between frames in seconds (default: ~60fps)."""


def main() -> None:
  """Record demo with pretrained tracking policy and export to .viser file."""
  print("🎬 Setting up MJLab demo recording...")

  # Parse config
  record_cfg = tyro.cli(RecordConfig)

  # Ensure output directory exists
  record_cfg.output_dir.mkdir(parents=True, exist_ok=True)

  try:
    checkpoint_path = ensure_default_checkpoint()
    motion_path = ensure_default_motion()
  except RuntimeError as e:
    print(f"❌ Failed to download demo assets: {e}")
    print("Please check your internet connection and try again.")
    return

  # Create play config
  play_cfg = PlayConfig(
    checkpoint_file=checkpoint_path,
    motion_file=motion_path,
    num_envs=record_cfg.num_envs,
    viewer="viser",
    motion_command_sampling_mode="uniform",
  )

  print(f"🚀 Creating environment and loading policy...")
  env, policy = create_env_and_policy("Mjlab-Tracking-Flat-Unitree-G1", play_cfg)
  # Get simulation
  sim = env.unwrapped.sim
  assert isinstance(sim, Simulation)

  print(f"📹 Starting recording server...")
  # Create Viser server
  server = viser.ViserServer(label="mjlab-recording", verbose=False)

  # Create scene
  scene = ViserMujocoScene.create(
    server=server,
    mj_model=sim.mj_model,
    num_envs=record_cfg.num_envs,
  )

  # Get serializer for recording
  serializer = server.get_scene_serializer()

  # Reset environment
  env.reset()

  print(f"🎥 Recording {record_cfg.num_steps} steps...")
  print(f"   Output: {record_cfg.output_dir / (record_cfg.output_name + '.viser')}")

  frame_count = 0
  for step in range(record_cfg.num_steps):
    # Run policy
    obs = env.unwrapped.get_observations()
    action = policy(obs)
    env.unwrapped.step(action)

    # Update visualization (only every Nth frame)
    if step % record_cfg.frame_skip == 0:
      with server.atomic():
        scene.update(sim.wp_data)
        server.flush()

      # Insert sleep for animation timing
      serializer.insert_sleep(record_cfg.sleep_duration)
      frame_count += 1

      if (step + 1) % 100 == 0:
        print(f"   Recorded {step + 1}/{record_cfg.num_steps} steps ({frame_count} frames)")

  print(f"💾 Saving recording...")

  # Save the recording
  output_path = record_cfg.output_dir / f"{record_cfg.output_name}.viser"
  with output_path.open("wb") as f:
    f.write(serializer.serialize())

  # Stop server
  server.stop()

  print(f"✅ Recording saved to: {output_path}")
  print(f"   Total frames: {frame_count}")
  print(f"   Duration: ~{frame_count * record_cfg.sleep_duration:.1f} seconds")
  print()
  print("📝 Next steps:")
  print("   1. Run 'viser-build-client --out-dir docs/viser-client' to build the viewer")
  print("   2. Deploy to GitHub Pages")
  print(f"   3. Access at: https://[username].github.io/mjlab/viser-client/?recording=../recordings/{record_cfg.output_name}.viser")


if __name__ == "__main__":
  main()
