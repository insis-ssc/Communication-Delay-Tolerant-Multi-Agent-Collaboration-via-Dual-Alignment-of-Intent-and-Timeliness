## Installation instructions

Set up StarCraft II and SMAC:

```shell
bash install_sc2.sh
```

This will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You may also need to set the environment variable for SC2:

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Install Python environment with conda:

```bash
conda create -n pymarl python=3.7 -y
conda activate pymarl
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install sacred numpy scipy matplotlib seaborn pyyaml pygame pytest probscale imageio snakeviz tensorboard-logger tensorboard tensorboardx

or install with `requirements.txt` of pip:

## Run an experiment 

```shell
python3 src/main.py --config=[Algorithm name] --env-config=[Env name] with env_args.map_name=[Map name if choosing SC2 env]

python3 src/main.py --config=[Algorithm name] --env-config=[Env name] with env_args.map_name=[Map name if choosing SC2 env]
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs` includeing MAIC and other baselines.
`--env-config` refers to the config files in `src/config/envs` including `sc2`, `foraging` as the LB-Foraging environment (https://github.com/semitable/lb-foraging), `join1` as the hallway environment (https://github.com/TonghanWang/NDQ).

All results will be stored in the `results` folder.
