"""Script to record velocity task demos and export them as .viser files for GitHub Pages.

This script runs a demo with a pretrained velocity control policy and exports
the visualization as a .viser file that can be embedded in static webpages for
GitHub Pages.

Note: This script is designed to run headless for CI/CD environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import tyro


@dataclass
class RecordConfig:
  """Configuration for recording velocity demos."""

  task: str = "Mjlab-Velocity-Flat-Unitree-G1"
  """Task ID to record (e.g., Mjlab-Velocity-Flat-Unitree-G1)."""

  checkpoint: str | None = None
  """Path to checkpoint file. If None, will try to find trained checkpoint."""

  output_dir: Path = Path("docs/recordings")
  """Directory to save .viser recordings."""

  output_name: str = "velocity-demo"
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
  """Record demo with pretrained velocity policy and export to .viser file."""
  print("🎬 Setting up MJLab velocity demo recording...")

  # Parse config
  record_cfg = tyro.cli(RecordConfig)

  # Ensure output directory exists
  record_cfg.output_dir.mkdir(parents=True, exist_ok=True)

  # Get checkpoint path
  if record_cfg.checkpoint is not None:
    checkpoint_path = record_cfg.checkpoint
    print(f"📦 Using checkpoint: {checkpoint_path}")
  else:
    print("❌ No checkpoint specified.")
    print("Please provide a checkpoint path using --checkpoint")
    return

  print("🚀 Creating environment and loading policy...")

  # Import here to avoid issues with MuJoCo GL context initialization
  import torch  # type: ignore[import-not-found]
  import viser

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.sim.sim import Simulation
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer.viser.scene import ViserMujocoScene

  configure_torch_backends()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  env_cfg = load_env_cfg(record_cfg.task, play=True)
  agent_cfg = load_rl_cfg(record_cfg.task)

  # Override num_envs
  env_cfg.scene.num_envs = record_cfg.num_envs

  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  # Load policy
  runner = VelocityOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(checkpoint_path, map_location=torch.device(device))
  policy = runner.get_inference_policy(device)

  # Get simulation
  sim = env.unwrapped.sim
  assert isinstance(sim, Simulation)

  print("📹 Starting recording server...")
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

  # Set forward velocity command for all environments
  # Access the command manager and set forward velocity
  cmd_manager = env.unwrapped.command_manager
  if "twist" in cmd_manager.active_terms:
    import torch

    # Get the twist command term and set forward velocity
    twist_term = cmd_manager.get_term("twist")

    # Set commands to move forward at 1.0 m/s
    twist_term.command[:, 0] = 1.0  # linear velocity x (forward)
    twist_term.command[:, 1] = 0.0  # linear velocity y (lateral)
    twist_term.command[:, 2] = 0.0  # angular velocity z (yaw)

    print("   Command: Forward velocity = 1.0 m/s")

  print(f"🎥 Recording {record_cfg.num_steps} steps...")
  print(f"   Task: {record_cfg.task}")
  print(f"   Output: {record_cfg.output_dir / (record_cfg.output_name + '.viser')}")

  frame_count = 0
  for step in range(record_cfg.num_steps):
    # Run policy
    obs = env.get_observations()
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
        print(
          f"   Recorded {step + 1}/{record_cfg.num_steps} steps ({frame_count} frames)"
        )

  print("💾 Saving recording...")

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
  print(
    "   1. Run 'viser-build-client --out-dir docs/viser-client' to build the viewer"
  )
  print("   2. Deploy to GitHub Pages")
  print(
    f"   3. Access at: https://[username].github.io/mjlab/viser-client/?playbackPath=../recordings/{record_cfg.output_name}.viser"
  )


if __name__ == "__main__":
  main()
