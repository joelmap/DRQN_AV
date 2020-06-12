# DRQN_AV
Decision making and planning for driverless vehicles using Deep Recurrent Q-Network


### Usage guide

1. Install Python 3.7 or higher
1. [Install Gym openAI](https://github.com/openai/gym)
1. [Install macad-gym](https://github.com/praveen-palanisamy/macad-gym#getting-started)
1. [Change the scenarios.py and stop_sign_3c_town03.py files from macad-gym to the files that are inside the macad-gym_mod folder](https://github.com/joelmap/DRQN_AV/tree/master/macad-gym_mod)
	- This will modify the objects on the scene, the town and the location of the vehicles
1. Run python Carlamacad_DRQN.py to start DRQN agent training on Carla simulator
