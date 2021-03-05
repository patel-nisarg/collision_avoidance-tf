# Vehicle Collision Avoidance
Jupyter Notebook for collision avoidance using TensorFlow on Jetson Nano

This module covers the use of TensorFlow and MobileNetV2 to train and perform collision avoidance on your robot. 

### What you need:

- Jetson board
- Adafruit DC motors (and their drivers)
- Raspberry Pi camera

The way you configure these is up to you. For example, here is mine below:
![image](https://github.com/patel-nisarg/collision_avoidance-tf/blob/main/Images/jetbot.jpeg)

The model consists of pre-trained MobileNetV2 layers with one output layer. The output layer detects whether there is a possible collision ahead or the path is free. In order to do this, you will need to take images of both blocked and free paths like so:

![jetbot_collision_system](https://user-images.githubusercontent.com/74885742/110142745-7d95a600-7da4-11eb-938e-03c75efb2b76.png)

For my model, I collected roughly 100 images of both blocked and free paths and got 99% validation accuracy. You might need more or less depending on the inherent variance of the environment where you deploy your robot. It cannot hurt to have more data and simply filter it.

Future Goals:
- Use cheap [vibration sensor](https://www.adafruit.com/product/1766) to autonomously collect training data while robot drives around at random
    - 3D Print a front, side, and rear bumper for collision to place sensors
    - Can use a reinforcement learning type model with this and a state space update approach
    - Can also better train the same model 
