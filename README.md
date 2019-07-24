# Deep Reinforcement Learning with Sonic the Hedgehog

### Requirements
* ROMS for Sonic the Hedgehog (to be purchased, e.g. from Steam) 
* Docker 
* VcXsrv Windows X Server (for Windows)


### Getting started
* Run Xlaunch 
  * follow initial configuration steps, additionally check box "Disable access control" in "Extra settings"
* In PowerShell: Build and launch docker (see Dockerfile for instruction)  
* import ROMS: `python -m retro.import.sega_classics` (authentification required)
* once installed, ROMS can be imported: `python -m retro.import <path to folder>`

#### Brief overview (see __main__.py for all options) 
* `--train`: start training
* `--eval`: evalue model
* `--render`: render image, optional 
* `--retrain`: continue training existing model, pass model using `--model`
* `--state`: level, has to be specified
<br>

#### Examples 
##### Training 
trained model will be saved as .pkl file in folder <br>
`python . –-train –-render –-state GreenHillZone.Act1`

##### Evaluation
`python . –-eval –-render –-state GreenHillZone.Act1 --model green_hill_1.pkl`

##### Retraining
`python . --retrain --render --state GreenHillZone.Act2 --model green_hill_1.pkl`



### Further information
* green_hill_1.pkl and green_hill2.pkl have been pretrained on GreenHillZone.Act1 and GreenHillZone.Act2, respectively. 
* gym_rl.py: Train to play Atari Games 



### Ressources
https://openai.com/blog/first-retro-contest-retrospective/
https://github.com/openai/retro
https://github.com/openai/baselines/blob/master/baselines/ppo2/ppo2.py

