import copy
from rsl_rl.runners import OnPolicyRunner
from env import FoosballEnv


train_cfg = {
    "obs_groups": {},
    "num_steps_per_env": 128,
    "save_interval": 50,
    "algorithm": {
        "class_name": "PPO",
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "entropy_coef": 0.002,
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
    },
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 64],
        "activation": "elu",
    },
}

if __name__ == "__main__":
    # Initialize the environment
    env = FoosballEnv(num_envs=4096, dt=1.0/60.0, device="cuda:1", model="model.xml", always_blue=True, bias_to_blue=True)

    print("Loading enemy...")

    # temp_runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir="foosball", device="cuda:0")
    # temp_runner.load("logs/foosball2/opp_2.pt")
    # env.opponent_policy = temp_runner.get_inference_policy(device="cuda:0")

    # Initialize the runner
    runner = OnPolicyRunner(env, copy.deepcopy(train_cfg), log_dir="./logs/", device="cuda:1")
    # runner.load("../drive/MyDrive/logs/foosball4/model_499.pt", map_location="cuda:0")
    #runner.load("logs/foosball/model_1450.pt")
    print("Starting training block...")

    # Execute the learning loop
    runner.learn(num_learning_iterations=500, init_at_random_ep_len=True)
