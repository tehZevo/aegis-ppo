import os
import json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from env import AegisEnv
import json

import sys
import signal
signal.signal(signal.SIGTERM, lambda: sys.exit(0))

obs_shape = os.getenv("OBS_SHAPE", "[]")
obs_shape = json.loads(obs_shape)
action_shape = os.getenv("ACTION_SHAPE", "[]")
action_shape = json.loads(action_shape)
port = int(os.getenv("PORT", 80))
policy = os.getenv("POLICY", "MlpPolicy")
#default to training every 32 steps
N_STEPS = int(os.getenv("N_STEPS", 32))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
N_EPOCHS = int(os.getenv("N_EPOCHS", 1))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", 1000))
RESET = os.getenv("RESET", "").lower() in (True, 'true') #default to false
MODEL_PATH = os.getenv("MODEL_PATH", "models/model")
VERBOSE = int(os.getenv("VERBOSE", 0))

#ppo params
#according to defaults here https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
#TODO: make these default to none so they arent filled in maybe idk
LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0003))
GAMMA = float(os.getenv("GAMMA", 0.99))
LAMBDA = float(os.getenv("LAMBDA", 0.95))

# Create environment
env = AegisEnv(obs_shape, action_shape, port=port)
env = DummyVecEnv([lambda: env])
#TODO: support LSTM/CNN policies

model = None
#load model if not RESET
if RESET:
    model = PPO(policy, env)
else:
    try:
        print("Loading", MODEL_PATH)
        model = PPO.load(
            MODEL_PATH,
            env=env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=LAMBDA,
            verbose=VERBOSE
        )
        print(MODEL_PATH, "loaded")
    except FileNotFoundError as e:
        print(e)
        print('"{}" not found. Creating new model.'.format(MODEL_PATH))
        model = PPO(policy, env, learning_rate=LEARNING_RATE, gamma=GAMMA, gae_lambda=LAMBDA, verbose=VERBOSE)
        model.save(MODEL_PATH)

class SaveCallback(BaseCallback):
    def __init__(self, save_steps=1000, verbose: int = 0):
        super().__init__(verbose)
        self.steps_since_last_save = 0
        self.save_steps = save_steps

    def _on_step(self):
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.save_steps:
          print(f"Saving model to '{MODEL_PATH}'...")
          self.model.save(MODEL_PATH)
          self.steps_since_last_save = 0

        return True

save_callback = SaveCallback(SAVE_STEPS)

#train
while True:
  model.learn(total_timesteps=999999999, callback=save_callback)
