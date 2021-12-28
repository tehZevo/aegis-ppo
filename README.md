# Aegis PPO node

Powered by [stable-baselines](https://stable-baselines.readthedocs.io/en/master/)

## Environment variables
* `OBS_SHAPE` - observation shape as a json array
* `ACTION_SHAPE` - action shape as int/json array
* `PORT` - port to listen for action requests on, defaults to 80
* `POLICY` - Stable Baselines PPO policy to use, defaults to `MlpPolicy`
* `STEPS` - Steps to run for before and saving, defaults to 10000
* `MODEL_PATH` - load/save path for the Stable Baselines PPO model, defaults to `"models/model"`
* `RESET` - if true, will create a new model instead of loading an existing one
* `VERBOSE` - Stable Baselines PPO2 verbosity level (int)

## TODO
* more documentation (env vars, request/response)
* support LSTM/CNN policies
* test :)
* figure out where first obs should come from..
* allow done to be set through a route eg `/done`
* more configuration of aegis env
  * especially allow configuration of obs/action low/high
* move AegisEnv to separate repo?
* update to [stable baselines / zoo 3](https://github.com/DLR-RM/rl-baselines3-zoo)
