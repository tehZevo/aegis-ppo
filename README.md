# Aegis PPO node

## Environment variables
* `OBS_URL` - observation url
* `OBS_SHAPE` - observation shape as a json array
* `ACTION_SHAPE` - action shape as int/json array
* `PORT` - port to listen for action requests on, defaults to 80
* `POLICY` - Stable Baselines PPO policy to use, defaults to `MlpPolicy`
* `STEPS` - Steps to run for before looping and saving, defaults to 10000
* `MODEL_PATH` - load/save path for the Stable Baselines PPO model, defaults to `"models/model"`
* `RESET` - if true, will create a new model instead of loading an existing one

## TODO
* more documentation
* support LSTM/CNN policies
