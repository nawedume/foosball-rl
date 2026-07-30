import copy
import os
import torch
from torch.profiler import profile, ProfilerActivity
from rsl_rl.runners import OnPolicyRunner
from env import FoosballEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chpt", type=str, help="Checkpoint path", default=None)
parser.add_argument("--device", type=str, help="Device str", default='cuda:0')
parser.add_argument("--iter", type=int, help="Number of iterations", default=3)
parser.add_argument("--op", type=str, help="Filepath to op policy", default=None)
parser.add_argument("--profile", action="store_true", help="Run in profiling mode")
parser.add_argument("--trace_file", type=str, help="Output trace file name", default="foosball_trace.json")
args_cli = parser.parse_args()

train_cfg = {
    "obs_groups": {},
    "num_steps_per_env": 128,
    "save_interval": 50,
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.001,
        "num_learning_epochs": 5,
        "num_mini_batches": 8,
        "learning_rate": 3e-4,
        "max_grad_norm": 1.0,
    },
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "scalar",
        },
        "obs_normalization": True,
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
        "obs_normalization": True,
    },
}

if __name__ == "__main__":
    print("--- EXECUTING MAIN BLOCK ---")
    
    device = args_cli.device
    
    # Safely reduce environments if profiling to avoid OOM on your RTX 2060
    env_count = 256 if args_cli.profile else 4096
    
    # Initialize the environment
    env = FoosballEnv(num_envs=env_count, dt=1.0/60.0, device=device, model="model.xml", always_blue=True, bias_to_blue=True)

    # Initialize the runner
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir="./logs/", device=device)
    if args_cli.chpt:
        runner.load(args_cli.chpt, map_location=device)

    if args_cli.op:
        policy = runner.get_inference_policy(device=device)
        env.op_policy = policy

    print("Starting training block...")

    if args_cli.profile:
        print(f"PROFILING MODE ENABLED. Environments restricted to {env_count}.")
        trace_path = os.path.abspath(args_cli.trace_file)
        
        iterations = 1
        
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
        except Exception as e:
            print(f"Could not Profile: {e}")
                
        finally:
            print("Exporting trace... DO NOT press Ctrl+C.")
            prof.export_chrome_trace(trace_path)
            print(f"SUCCESS: Trace hard-saved to {trace_path}")
            
    else:
        # Standard execution loop
        runner.learn(num_learning_iterations=args_cli.iter, init_at_random_ep_len=True)