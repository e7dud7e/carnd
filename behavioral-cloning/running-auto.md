run the simulator in your laptop

ssh into aws
ssh carnd@xxx.xxx.xxx.xxx -L 4567:localhost:4567
ssh carnd@ -L 4567:localhost:4567


#start your model
python drive.py model.h5

drive autonomously on laptop 


Also, I need to run python drive.py model.h5
and then click on the "autonomous mode."  
If I open autonomous mode and then run python drive.py model.5,
it won't drive the car.
