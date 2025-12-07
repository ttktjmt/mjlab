"""Generalized script to record any MJLab environment and export as .viser file.

This script works with any registered MJLab task and supports loading checkpoints
from local files or WandB runs. The visualization is exported as a .viser file
that can be embedded in static webpages.

Usage examples:
  # Using WandB checkpoint
  uv run record Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path org/project/run-id --name rec1

  # Using local checkpoint
  uv run record Mjlab-Cartpole --checkpoint-file logs/rsl_rl/cartpole/model_60.pt --num-envs 4 --name rec2

  # Motion tracking task with WandB
  uv run record Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path org/project/run-id

Note: This script is designed to run headless for CI/CD environments.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro


@dataclass
class RecordConfig:
  """Configuration for recording MJLab environments."""

  checkpoint_file: str | None = None
  """Path to local checkpoint file (.pt)."""

  wandb_run_path: str | None = None
  """WandB run path (e.g., 'entity/project/run-id') to download checkpoint."""

  motion_file: str | None = None
  """Path to motion file for tracking tasks. If not provided, will try to resolve from WandB."""

  output_dir: Path = Path("recordings")
  """Directory to save .viser recordings."""

  name: str | None = None
  """Name for the output file (without extension). If not provided, auto-generated from task."""

  num_steps: int = 500
  """Number of simulation steps to record."""

  num_envs: int = 8
  """Number of parallel environments to visualize."""

  frame_skip: int = 2
  """Only record every Nth frame (1 = record all frames)."""

  sleep_duration: float = 0.016
  """Sleep duration between frames in seconds (default: ~60fps)."""

  device: str | None = None
  """Device to run on (e.g., 'cuda:0', 'cpu'). Auto-detected if not provided."""


def _generate_output_name(task_id: str) -> str:
  """Generate a reasonable output name from task ID.

  Examples:
    Mjlab-Velocity-Flat-Unitree-G1 -> velocity-g1
    Mjlab-Tracking-Flat-Unitree-G1 -> tracking-g1
    Mjlab-Cartpole -> cartpole
  """
  # Remove "Mjlab-" prefix
  name = task_id.replace("Mjlab-", "").lower()

  # Extract task type and robot name
  parts = name.split("-")
  if len(parts) >= 2:
    task_type = parts[0]  # velocity, tracking, etc.
    robot = parts[-1]  # g1, go1, etc.
    return f"{task_type}-{robot}"
  else:
    # Simple task like "Cartpole"
    return name


def main() -> None:
  """Record MJLab environment with trained policy and export to .viser file."""
  # Parse first argument to choose the task
  # Import tasks to populate the registry
  import mjlab.tasks  # noqa: F401

  from mjlab.tasks.registry import list_tasks

  all_tasks = list_tasks()
  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments
  record_cfg = tyro.cli(
    RecordConfig,
    args=remaining_args,
    default=RecordConfig(),
    prog=sys.argv[0] + f" {task_id}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  print(f"🎬 Setting up MJLab recording for task: {task_id}")

  # Validate checkpoint source
  if record_cfg.checkpoint_file is None and record_cfg.wandb_run_path is None:
    print("❌ Error: Must provide either --checkpoint-file or --wandb-run-path")
    return

  if record_cfg.checkpoint_file is not None and record_cfg.wandb_run_path is not None:
    print("❌ Error: Cannot provide both --checkpoint-file and --wandb-run-path")
    return

  # Ensure output directory exists
  record_cfg.output_dir.mkdir(parents=True, exist_ok=True)

  # Generate output name if not provided
  output_name = record_cfg.name or _generate_output_name(task_id)

  print("🚀 Creating environment and loading policy...")

  # Import here to avoid issues with MuJoCo GL context initialization
  import torch  # type: ignore[import-not-found]
  import viser
  from rsl_rl.runners import OnPolicyRunner

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.sim.sim import Simulation
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.tasks.tracking.mdp import MotionCommandCfg
  from mjlab.utils.os import get_wandb_checkpoint_path
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer.viser.scene import ViserMujocoScene

  configure_torch_backends()

  device = record_cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Load configurations
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Check if this is a tracking task by checking for motion command
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  # Handle motion file for tracking tasks
  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    if record_cfg.motion_file is not None:
      print(f"   Using motion file from CLI: {record_cfg.motion_file}")
      motion_cmd.motion_file = record_cfg.motion_file
    else:
      import wandb

      api = wandb.Api()
      if record_cfg.wandb_run_path is None and record_cfg.checkpoint_file is not None:
        print(
          "❌ Error: Tracking tasks require --motion-file when using --checkpoint-file, "
          "or provide --wandb-run-path so the motion artifact can be resolved."
        )
        return
      if record_cfg.wandb_run_path is not None:
        wandb_run = api.run(str(record_cfg.wandb_run_path))
        art = next(
          (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
        )
        if art is None:
          print("❌ Error: No motion artifact found in the WandB run.")
          return
        motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    # Use uniform sampling for more diversity with multiple environments
    motion_cmd.sampling_mode = "uniform"

  # Get checkpoint path
  if record_cfg.checkpoint_file is not None:
    checkpoint_path = Path(record_cfg.checkpoint_file)
    if not checkpoint_path.exists():
      print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
      return
    print(f"📦 Using checkpoint: {checkpoint_path.name}")
  else:
    assert record_cfg.wandb_run_path is not None
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(record_cfg.wandb_run_path)
    )
    # Extract run_id and checkpoint name from path for display
    run_id = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      f"📦 Using checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
    )

  # Override num_envs
  env_cfg.scene.num_envs = record_cfg.num_envs

  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env)

  # Load policy with the appropriate runner
  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint_path), map_location=torch.device(device))
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

  print(f"🎥 Recording {record_cfg.num_steps} steps...")
  print(f"   Task: {task_id}")
  print(f"   Output: {record_cfg.output_dir / (output_name + '.viser')}")

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
  output_path = record_cfg.output_dir / f"{output_name}.viser"
  with output_path.open("wb") as f:
    f.write(serializer.serialize())

  # Stop server
  server.stop()

  print(f"✅ Recording saved to: {output_path}")
  print(f"   Total frames: {frame_count}")
  print(f"   Duration: ~{frame_count * record_cfg.sleep_duration:.1f} seconds")


if __name__ == "__main__":
  main()
