"""Train with durable checkpoints, CSV metrics, and deterministic evaluation."""

import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import numpy as np
import torch
import warp as wp
from rsl_rl.runners import OnPolicyRunner

from env import FoosballEnv
from policy_cfg import train_cfg


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def evaluate(runner, count, seed, use_cuda_graph):
    """Evaluate the first complete episode in each world without changing training RNG."""
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.device(runner.device)]):
            env = FoosballEnv(
                num_envs=count,
                device=runner.device,
                dt=1.0 / 60.0,
                always_blue=True,
                bias_to_blue=True,
                seed=seed,
                use_cuda_graph=use_cuda_graph,
                op_policy=getattr(runner, "frozen_opponent", None),
            )
            policy = runner.get_inference_policy()
            finished = torch.zeros(count, dtype=torch.bool, device=runner.device)
            returns = torch.zeros(count, device=runner.device)
            lengths = torch.zeros(count, device=runner.device)
            goals = torch.zeros(count, device=runner.device)
            conceded = torch.zeros(count, device=runner.device)
            out_of_bounds = torch.zeros(count, device=runner.device)
            obs = env.get_observations()
            with torch.inference_mode():
                for _ in range(env.max_episode_length + 1):
                    obs, reward, done, extras = env.step(policy(obs))
                    active = ~finished
                    returns += reward * active
                    lengths += active
                    goals += ((extras["log"]["Metrics/goal"] > 0) & active)
                    conceded += ((extras["log"]["Metrics/goal"] < 0) & active)
                    out_of_bounds += ((extras["log"]["Metrics/oob_rate"] < 0) & active)
                    finished |= done.bool()
                    if finished.all().item():
                        break
            if not finished.all().item():
                raise RuntimeError("Evaluation did not finish every episode")
            return {
                "seed": seed,
                "episodes": count,
                "goal_rate": goals.mean().item(),
                "conceded_rate": conceded.mean().item(),
                "out_of_bounds_rate": out_of_bounds.mean().item(),
                "mean_return": returns.mean().item(),
                "mean_episode_length": lengths.mean().item(),
            }
    finally:
        np.random.set_state(numpy_state)
        runner.alg.train_mode()


def plot_results(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, [
        ("mean_episode_return", "Training episode return"),
        ("transitions_per_second", "Training transitions per second"),
        ("value_loss", "Value loss"),
        ("surrogate_loss", "Surrogate loss"),
    ]):
        axis.plot([int(row["completed_iterations"]) for row in rows],
                  [float(row[key]) for row in rows])
        axis.set(title=title, xlabel="Completed iterations")
        axis.grid(alpha=0.2)
    figure.savefig(output / "training-curves.png", dpi=160)
    plt.close(figure)
    with (output / "evaluations.csv").open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for key, label in [("goal_rate", "Scored"), ("conceded_rate", "Conceded"),
                       ("out_of_bounds_rate", "Out of bounds")]:
        axis.plot([int(row["completed_iterations"]) for row in evaluations],
                  [float(row[key]) for row in evaluations], marker="o", label=label)
    axis.set(xlabel="Completed iterations", ylabel="Fraction of evaluation episodes",
             title="Deterministic policy evaluation against configured opponent", ylim=(-0.02, 1.02))
    axis.legend()
    axis.grid(alpha=0.2)
    figure.savefig(output / "evaluation-curves.png", dpi=160)
    plt.close(figure)


class TrainingTimeLimit(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=7.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--training-seconds", type=float)
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.num_envs, args.save_interval,
           args.eval_interval, args.eval_envs, args.max_hours) <= 0:
        parser.error("Iteration counts, environment counts, intervals, and time limit must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run-config.json").exists():
        parser.error("Choose a new output directory for each run, including resumed runs")
    started = time.monotonic()
    device = "cuda:0"
    cfg = copy.deepcopy(train_cfg)
    cfg["save_interval"] = args.save_interval
    cfg["obs_groups"] = {"actor": ["policy"], "critic": ["policy"]}
    config = {
        "arguments": {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "training": cfg,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib"]},
    }
    write_json(output / "run-config.json", config)
    env = FoosballEnv(num_envs=args.num_envs, dt=1.0 / 60.0, device=device,
                      always_blue=True, bias_to_blue=True, seed=args.seed,
                      use_cuda_graph=not args.no_cuda_graph)
    runner = OnPolicyRunner(env, cfg, log_dir=str(output / "checkpoints"), device=device)
    if args.resume:
        resume_info = runner.load(str(args.resume), map_location=device) or {}
        if "completed_iterations" in resume_info:
            runner.current_learning_iteration = resume_info["completed_iterations"]
        else:
            # Legacy checkpoints store the last completed zero-based index.
            runner.current_learning_iteration += 1
    if args.init_from:
        if args.resume:
            parser.error("Use either --resume or --init-from")
        runner.load(str(args.init_from), load_cfg={"actor": True, "critic": True}, map_location=device)
    if args.opponent_checkpoint:
        opponent = copy.deepcopy(runner.alg.actor)
        checkpoint = torch.load(args.opponent_checkpoint, map_location=device, weights_only=False)
        opponent.load_state_dict(checkpoint["actor_state_dict"])
        opponent.eval()
        opponent.requires_grad_(False)
        runner.frozen_opponent = opponent
        env.op_policy = opponent
        print("FROZEN_OPPONENT", str(args.opponent_checkpoint), flush=True)
    completed = runner.current_learning_iteration
    initial_completed = completed
    best_score = (-float("inf"), -float("inf"))
    best_evaluation = None
    last_evaluation_iteration = None
    training_seconds = 0.0
    original_save = runner.save

    def save_checkpoint(path, infos=None):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.pt")
        checkpoint_info = {"completed_iterations": completed, **(infos or {})}
        original_save(str(temporary), checkpoint_info)
        os.replace(temporary, destination)

    runner.save = save_checkpoint

    def record_evaluation():
        nonlocal best_score, best_evaluation, last_evaluation_iteration
        result = {"completed_iterations": completed,
                  **evaluate(runner, args.eval_envs, 12345, not args.no_cuda_graph)}
        append_csv(output / "evaluations.csv", result)
        last_evaluation_iteration = completed
        print("EVALUATION", json.dumps(result), flush=True)
        score = (result["goal_rate"], result["mean_return"])
        if score > best_score:
            best_score = score
            best_evaluation = result
            runner.save(output / "best.pt", infos={"evaluation": result})

    original_log = runner.logger.log

    def log_iteration(**fields):
        nonlocal completed, training_seconds
        completed = fields["it"] + 1
        duration = fields["collect_time"] + fields["learn_time"]
        training_seconds += duration
        row = {
            "completed_iterations": completed,
            "elapsed_seconds": time.monotonic() - started,
            "total_transitions_this_run": (completed - initial_completed) * args.num_envs * cfg["num_steps_per_env"],
            "rollout_seconds": fields["collect_time"],
            "learning_seconds": fields["learn_time"],
            "transitions_per_second": args.num_envs * cfg["num_steps_per_env"] / duration,
            "mean_episode_return": statistics.fmean(runner.logger.rewbuffer) if runner.logger.rewbuffer else float("nan"),
            "mean_episode_length": statistics.fmean(runner.logger.lenbuffer) if runner.logger.lenbuffer else float("nan"),
            "value_loss": fields["loss_dict"].get("value", float("nan")),
            "surrogate_loss": fields["loss_dict"].get("surrogate", float("nan")),
        }
        append_csv(output / "metrics.csv", row)
        write_json(output / "progress.json", row)
        original_log(**fields)
        if completed % args.eval_interval == 0:
            runner.save(output / "latest.pt")
            record_evaluation()
        if (args.training_seconds is not None and training_seconds >= args.training_seconds) or time.monotonic() - started >= args.max_hours * 3600:
            raise TrainingTimeLimit()

    runner.logger.log = log_iteration
    status = "completed"
    try:
        record_evaluation()
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    except TrainingTimeLimit:
        status = "time_limit"
    except BaseException:
        status = "failed"
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # On failure, the periodic checkpoints remain the last completed updates.
        if status != "failed":
            runner.save(output / "final.pt")
            runner.save(output / "latest.pt")
        summary = {
            "status": status,
            "completed_iterations": completed,
            "elapsed_seconds": time.monotonic() - started,
            "training_seconds": training_seconds,
            "best_evaluation": best_evaluation,
            "convergence": "Assess the evaluation curve and final test rates; a completed budget does not establish convergence.",
        }
        write_json(output / "summary.json", summary)
        if runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
    if last_evaluation_iteration != completed:
        record_evaluation()
    summary["best_evaluation"] = best_evaluation
    summary["final_test"] = evaluate(runner, args.eval_envs, 67890, not args.no_cuda_graph)
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(output / "summary.json", summary)
    plot_results(output)
    print("RUN_SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
