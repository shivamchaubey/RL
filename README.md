# RC_car_reinforcement_learning

Gazebo Simulation: 

Folder structure:
Describing only the folders which are necessary for this document and purpose. Other things are kept there because it will be used for real implementation on vehicle or adding different sensors for simulation. 

Folders which are being used for the following development: 
my_pkg >> vehicle >>  src >> simulation >> src >> seat_car_simulator >> * :: This has both the gazebo simulation and environment_interaction folder. For simulation all the folders are used except  environment_interaction folder.
my_pkg >> vehicle >>  src >> simulation >> src >> seat_car_simulator >> environment_interaction :: This location have all the code related to interaction of gazebo environment.

Build the whole workspace using catkin_make and open two terminal and source devel/setup.bash in both. 

Running simulation:
There is two part to run and interact with gazebo simulation.
To run the gazebo simulator launch the sim.launch using command: roslaunch seat_car_gazebo sim.launch
To run the interaction module launch file using roslaunch simulation_control interact_sim.launch. 

Basic:
There is an vehilceSimulator,py file which takes input of steering and acceleration command using topic “ecu”. This file implements the kinematic model of the vehicle and generates the states of vehicle in the topic “vehicle_state”. The gazebo uses topic “/gazebo/set_model_state” to move the vechile using x,y and yaw which is obtained from the vehicle state and also there is an offset added to it as the vehicle can start away from the origin. Now using this information we can build our action, states and rewards for RL.

Action:
	The action is sent via file “controller.py” using node “sim_control” and topic “ecu”.	Actions input are:
Acceleration : [-3.0 , 3.0]
Steering angle: [-0.249 , 0.249]
	The minimum and maximum deviation allowed otherwise there will be slippage:
Dot(Acceleration) = 0.1*acceleration.
Dot(Steering angle) = 0.1*Steering angle.
	These parameters could be changed (lowered down)
	Limit the vehicle speed (vehicle states not action) to 0.01 to 5.0 (Easy for testing on real vehicle) 
State (observation):
	For now just camera information is retrieved in later version Lidar will also be integrated.
	The file “sensor_observer.py”  and node “sim_observer”is written to transform the 	compressed image to numpy 3D 	array which can be directly fed to neural network.  The 	topic from gazebo 	“/camera/rgb/image_raw/compressed” is decoded to be used as	numpy array. To turn off the separate display using numpy data set the “disp” param value to 	off from “interact_sim.launch” file. The display also detect features using GridFast 	detector which can be used to interpolate the shape for lane detection.
	Output:
		For now this doesn’t return anything as I don’t know how the integration will happen 		in OpenAI but it’s easy to get data as we already have transformed. 

Rewards:
	For now I am using the vehicle state to accelerate the development and focus more 	towards learning part but later on we can use other sensors to estimate the position 	separately. 



