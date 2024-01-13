# Aegis PPO node

Powered by [stable-baselines](https://stable-baselines.readthedocs.io/en/master/)

## Environment variables
* `OBS_SHAPE` - observation shape as a json array
* `ACTION_SHAPE` - action shape as int/json array
* `PORT` - port to listen for action requests on, defaults to 80
* `POLICY` - Stable Baselines PPO policy to use, defaults to `MlpPolicy`
* `SAVE_STEPS` - Save every <this many> steps; defaults to 1000
* `MODEL_PATH` - load/save path for the Stable Baselines PPO model, defaults to `"models/model"`
* `RESET` - if true, will create a new model instead of loading an existing one
* `VERBOSE` - Stable Baselines PPO2 verbosity level (int)
* `N_STEPS` - number of steps to run between training; defaults to 2048
* `BATCH_SIZE` - batch size when training; defaults to 64
* `N_EPOCHS` - number of epochs to train; defaults to 10

## TODO
* more documentation (env vars, request/response)
* support LSTM/CNN policies
* test :)
* figure out where first obs should come from..
* allow done to be set through a route eg `/done`
