"""Script to record RL agent demonstrations and export as .viser files."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class RecordConfig:
  """Configuration for recording demonstrations."""

  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  output_dir: Path = Path("recordings")
  output_name: str | None = None
  num_steps: int = 500
  frame_skip: int = 2
  sleep_duration: float = 0.016
  share: bool = False


def _generate_output_name(task_id: str) -> str:
  """Generate output name from task ID.

  Examples:
    Mjlab-Velocity-Flat-Unitree-G1 -> velocity-g1
    Mjlab-Tracking-Flat-Unitree-G1 -> tracking-g1
    Mjlab-Cartpole -> cartpole
  """
  name = task_id.replace("Mjlab-", "").lower()
  parts = name.split("-")
  if len(parts) >= 2:
    task_type = parts[0]
    robot = parts[-1]
    return f"{task_type}-{robot}"
  else:
    return name


def run_record(task_id: str, cfg: RecordConfig):
  """Record demonstration with trained policy and export to .viser file."""
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
  )

  # Handle motion file for tracking tasks (same logic as play.py)
  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Use uniform sampling for more diversity with num_envs > 1
    motion_cmd.sampling_mode = "uniform"

    if cfg.motion_file is not None:
      print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    else:
      import wandb

      api = wandb.Api()
      if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
        raise ValueError(
          "Tracking tasks require `motion_file` when using `checkpoint_file`, "
          "or provide `wandb_run_path` so the motion artifact can be resolved."
        )
      if cfg.wandb_run_path is not None:
        wandb_run = api.run(str(cfg.wandb_run_path))
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
          raise RuntimeError("No motion artifact found in the run.")
        motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  # Get checkpoint path (same logic as play.py)
  resume_path: Path | None = None

  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  if cfg.checkpoint_file is not None:
    resume_path = Path(cfg.checkpoint_file)
    if not resume_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
    print(f"[INFO]: Loading checkpoint: {resume_path.name}")
  else:
    if cfg.wandb_run_path is None:
      raise ValueError(
        "`wandb_run_path` is required when `checkpoint_file` is not provided."
      )
    resume_path, was_cached = get_wandb_checkpoint_path(
      log_root_path, Path(cfg.wandb_run_path)
    )
    # Extract run_id and checkpoint name from path for display.
    run_id = resume_path.parent.name
    checkpoint_name = resume_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
    )

  # Override num_envs if specified
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Load policy with the appropriate runner (same as play.py)
  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  policy = runner.get_inference_policy(device=device)

  # Get simulation
  from mjlab.sim.sim import Simulation

  sim = env.unwrapped.sim
  assert isinstance(sim, Simulation)

  print("[INFO]: Starting recording server...")

  # Import viser here to avoid issues with MuJoCo GL context initialization
  import viser

  from mjlab.viewer.viser.scene import ViserMujocoScene

  # Create Viser server
  server = viser.ViserServer(label="mjlab-recording", verbose=False)

  if cfg.share:
    server.request_share_url()

  # Set initial camera pose for better default view
  @server.on_client_connect
  def _(client: viser.ClientHandle) -> None:
    # Good overview angle for humanoid robots
    client.camera.position = (3.0, 3.0, 2.0)
    client.camera.look_at = (0.0, 0.0, 0.8)

  # Create scene
  scene = ViserMujocoScene.create(
    server=server,
    mj_model=sim.mj_model,
    num_envs=env_cfg.scene.num_envs,
  )

  # Get serializer for recording
  serializer = server.get_scene_serializer()

  # Reset environment
  env.reset()

  # Generate output name if not provided
  output_name = cfg.output_name or _generate_output_name(task_id)

  # Ensure output directory exists
  cfg.output_dir.mkdir(parents=True, exist_ok=True)

  print(f"[INFO]: Recording {cfg.num_steps} steps...")
  print(f"[INFO]: Task: {task_id}")
  print(f"[INFO]: Output: {cfg.output_dir / (output_name + '.viser')}")

  # Recording loop
  frame_count = 0
  for step in range(cfg.num_steps):
    # Run policy
    obs = env.get_observations()
    action = policy(obs)
    env.unwrapped.step(action)

    # Update visualization (only every Nth frame)
    if step % cfg.frame_skip == 0:
      with server.atomic():
        scene.update(sim.wp_data)
        server.flush()

      # Insert sleep for animation timing
      serializer.insert_sleep(cfg.sleep_duration)
      frame_count += 1

      if (step + 1) % 100 == 0:
        print(
          f"[INFO]: Recorded {step + 1}/{cfg.num_steps} steps ({frame_count} frames)"
        )

  print("[INFO]: Saving recording...")

  # Save the recording
  output_path = cfg.output_dir / f"{output_name}.viser"
  with output_path.open("wb") as f:
    f.write(serializer.serialize())

  # Stop server
  server.stop()

  print(f"[INFO]: Recording saved to: {output_path}")
  print(f"[INFO]: Total frames: {frame_count}")
  print(f"[INFO]: Duration: ~{frame_count * cfg.sleep_duration:.1f} seconds")

  env.close()


def main():
  """Main entry point for record script."""
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments.
  args = tyro.cli(
    RecordConfig,
    args=remaining_args,
    default=RecordConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  run_record(chosen_task, args)


if __name__ == "__main__":
  main()
