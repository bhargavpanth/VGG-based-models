> Project consists of VGG16 VGG19 and a baseline model that was used in my visual recognition project.

## Project objective
I will soon be starting to build a quadcopter. This is an extension to the quadcopter project. I built this repo to abstract out the DL models that I used to bench mark my dataset. The ultimate objective is to use the weights from these models and transfer them onto an ARM based device such as RaspberryPi. Using the camera module of a Raspberry Pi, the idea is to conduct aerial census. I am planning on using this approach to build visual census model that can classify and detect a given object (the object has to be part of the training dataset).

## What next
I have a repo where I am documenting my (x-frame quadcopter)[https://github.com/bhargavpanth/drone-build-along] build. Once the copter is ready and we have all the components soldered and ready to fly, I will get back to this repo where I will explain the steps to run our DL model on a Raspberry Pi or just use this DL model on your computer and connect the camera from the drone via an RTMP server.
