# Current goals

Get a minimum viable vehicle running in a month! 

- Vehicle drives autonomously between taped lines
- Duct tape, chewing gum, and superglue allowed! 

[(Github Repo with source)](https://github.com/MarcusJones/ai.drive)

___

# Mar. 6, 2018 - Cats and Dogs

What makes a cat a cat? How would you define dogness in contrast to catness? 

![catsdogssample](/Post4_2018MAR06/catsdogs.jpg)

What relevance does this have to ai.drive()?
 
Let's rephrase the question: 

![leftright](/Post4_2018MAR06/left.jpg)

How does a machine know left from right? With the progress on hardware, it's
time to shift focus on the software and Machine Learning problem. To warm up, I'm
spending some time on my favourite Machine Learning problem; Cats vs. Dogs!  

Using ```keras```, I am buidling a basic neural network for classification; 

![model](/Post4_2018MAR06/modelTB.jpg)

And executing some basic training runs on a subset of the data; 
 
![trainingexample](/Post4_2018MAR06/Training example.jpg)

The results of a longer training run over 30 epochs are presented below. 

![30epoch](/Post4_2018MAR06/CatsDogs 30 epochs.jpg)










___

# Mar. 5, 2018 - Sony DS3 controller operational

## Connection debugging
After a few hours going through all the commands several times, I somehow got 
the controller to communicate via BlueTooth with the car.

To test the controller, I tried connecting it to my laptop. Running the 
```bluetoothctl``` utility gave immediate results;
```
user@laptop2 ~ $ bluetoothctl
[NEW] Controller 58:91:CF:1C:30:17 laptop2 [default]
[NEW] Device FC:62:B9:3B:B3:B5 PLAYSTATION(R)3 Controller
```
This provided the necessary verifcation to show that the controller was functional and the 
correct model. 

After reading several articles online, there was mention of power connections, 
so instead of running from the battery pack, I connected to a wall USB adapter 
and started debugging with monitor and keyboard connected. Either the power 
supply, or simply the act of reconnecting the power, resulted in recognition
of the device. 

After so much time debugging this problem, I learned of several tools for 
checking bluetooth and USB in linux;

```bluetoothctl``` - Utility for scanning and managing connections at CLI

```lsusb``` - List USB devices

```service bluetooth status``` - Status of bluetooth

```dmesg | grep -i bluetooth``` - Linux <> hardware messages related to bluetooth
 
```ls -a /dev/input/``` - List mounted input devices, check for js0

```vi /var/log/syslog``` - View the Linux system log file

I also used the joystick (```apt install joystick```) utility for verifying 
joystick signals;  

```jstest /dev/input/js0``` - Raw RAM IO map for this device

 
## Calibration

The PWM signal is controlled in the ```config.py``` file on the Pi. Two 
parameters influence the throttle signal - the calibrated maxium, which I set at
390, and the joystick throttle multiplier, which I incrementally increased from 
0.50 to 0.95, searching for the **minimum** speed for my current hardware configuration. 
In the future, I will set the PWM maximum calibration to the real maximum PWM 
signal, and use the multipliers to control the safe speed for my application. 

Here is the summary of the configuration parameters; 
```
THROTTLE_CHANNEL = 0
THROTTLE_FORWARD_PWM = 390
THROTTLE_STOPPED_PWM = 370
THROTTLE_REVERSE_PWM = 350
JOYSTICK_MAX_THROTTLE = 0.95
```


___

# Feb. 28, 2018 - It's alive! 

There were two main areas of progress so far; hardware and software. 

In both areas, the official project documentation is *outstanding*, and I will just summarize and highlight the differences and challenges. 

 [Project instructions are here](http://docs.donkeycar.com/). 


## Hardware

### Chassis

Here is the vehicle, in the nude;
![chassis](/Post2_2018FEB28/OriginalChassis.jpg)

And my current modifications. I chose to use cardboard for prototyping a structure. Lightweight, easy to work with, it gets the job done. Layers were glued together, and recesses were cut to hold the Raspberry Pi USB battery, and to route cables. Duct tape is my friend. 
{::nomarkdown}
<img src="https://marcusjones.github.io/ai.drive/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" width = 80%>
{:/} 

In a later stage, I would like to apply this additive layering technique using laser-cut plywood, below a snapshot of my progress in learning Fusion 360. 

{::nomarkdown}
<img src="https://marcusjones.github.io/ai.drive/Post2_2018FEB28/3DprintTest.png" width = 100%>
{:/} 

### Electrical systems 

As mentioned in the first post, the main difference in chassis, compared to the 'stock' car, is the battery pack. In this case, the 
double 1700mah 2s 20c batteries store significantly more energy, but critically, also have a higher discharge rate, which is represented by the **C-Rating**; 

```Max Current Draw = Capacity x C-Rating```

Secondly, the car has twice the voltage rating compared to the original 'donkey car' stock project. These two factors lead the car to have **way too much power** to make any real autonomous driving tests possible in this early phase. 

The table below lists the key parameters, and the third entry in the table is the current solution for de-powering the chassis: removing one battery from the power bank. 

| Variant                              | Technology | Voltage <br> [V] | Capacity <br> [mAH] | Discharge <br> [C-Rating] | Configuration <br>[S] |
|---------------------------------------------|------------|------------------|---------------------|---------------------------|-----------------------|
| **'Stock' project car  <br> Exceed Magnet** | NiMh       | 7.2              | 1100                | Much less than 20C        | ?                     |
| **My car 'HobbyKing Bad Bug'**           <br>        | LiPo       | 14.4             | 3400                | More than 20C             | 2S * 2 = **4S1P**     |
| **50% Bad-Bug  **    <br>   | LiPo       | 7.2              | 1700                | 20C                       | 2S * 2 = **2S1P**     |

This is therefore my current solution. Of course, I can put the second battery in parallel instead of removing it completely from the circuit. 

The figure below presents the electrical connections. 

{::nomarkdown}
<img src="https://marcusjones.github.io/ai.drive/Post2_2018FEB28/Electronics.jpg" width = 100%>
{:/} 

## Software installation

Thanks to the documentation and the USB disk image, installation was mostly painless. Here's a quick summary of the process, see the project page for details. 

1. Install software to the Raspberry Pi
	a. Write the project disk image to the SD card
	a. Wrote WiFi access to ```/etc/wpa_supplicant/wpa_supplicant.conf```
	a. Hostname remains as ```d2``` for now
	a. Power on RPi, have a keyboard and monitor handy for troubleshooting
	a. Test: Ping d2.local 
1. Installing software on the host (linux on my laptop)
	a. Tensorflow and Keras
	a. Clone the source for the 'donkeycar' project, the python code for running the vehicle
1. Testing
	a. ```ping d2.local ```
	a. SSH into the RPi ```ssh pi@d2.local```
	a. Git-pull the donkeycar repo
	a. Start a new car software by template ```donkey createcar --template donkey2 --path ~/d2```

## Running

With the hardware and software set, the normal procedure becomes;
1. ```source activate drive``` environment in linux
1. ```ssh pi@d2.local```
1. Run the ```python manage.py drive``` command to start the car and web service
1. Access the car at ```d2:8887```

## ESC Programming

In my ESC, there are different settings for controlling aspects of the motor performance. These settings are programmed by simple beep-code feedback. The user manual has some mistakes in the numbering, here are the codes for reference; 

| #  | Beep   | Name                       | Description                                       |
|----|--------|----------------------------|---------------------------------------------------|
| 1  | *      | Running mode               | Enable/Disable braking, reverse                   |
| 2  | **     | Reverse force              | 20% increments                                    |
| 3  | ***    | Brake force                | 25% increments                                    |
| 4  | ****   | Drag brake force           | Extra drag if coasting, in various % levels       |
| 5  | *****  | Neutral range              | Sensitivity of stick throttle in 3/6/9/12% levels |
| 6  | _*     | Start mode                 | Accel. at start, Very fast, Fast, Normal, Slow    |
| 7  | _**    | Timing                     | Brushless DC step timing in 0/5/10/15/20%levels   |
| 8  | _***   | Battery voltage protection | Keep default Li-xx 3.2V/cell                      |
| 9  | _****  | Over-heat protection       | Keep default 110C                                 |
| 10 | _***** | Protection mode            | Keep default 'lower power'                        |

The interesting ones are;
-	[4 - drag brake] to test slowing the car down with no throttle (glide)
-	[5 - neutral range] since my throttle is controlled by direct input of PWM values, I should be able to tighten this range
-	[6 - start mode] to keep the driving smooth

I experimented with several settings, but I'm not sure which ones I successfully changed, so that's on the list for later. 

## Calibration

I followed the steps for calibration, and this is still ongoing. To keep things slow, I used the minimum range on the throttle PWM signal which caused movement. This ended up being 355 / 370 / 385 for reverse, neutral, forward. 

The turning PWM signal was set to 290 / 420 for Right / Left. 

## Controller problems

I purchased the recommended sony dualshock sixaxis controller, but still can't get it running. There are several threads on connecting this bluetooth controller, using ```bluetoothctl``` to pair. On my setup, I couldn't get the device recognized by the ```devices``` command. Work-In-Progress. 

## Summary, next steps

So that's the next milestone. Coming up;
1. Getting the DS3 controller working
1. Tuning the calibration
1. Sketching the deep learning network model

___

# Feb. 2, 2018 - Procurement Update

Here's my experience matching the parts list with the goal being getting delivery of everything down to a week or so. 

| Supplier     | Price [EUR] | Tag            | Qty | Part description                                                                                                                                                       |
|--------------|-------------|----------------|-----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| (Amazon)     | 35          | R Pi           | 1   | Raspberry Pi 3 Model B ARM-Cortex-A53 4x 1,2GHz, 1GB RAM, WLAN, Bluetooth, LAN, 4x USB                                                                                 |
| (Amazon)     | 23          | SD             | 1   | SanDisk Ultra 64GB microSDXC Speicherkarte + Adapter bis zu 100 MB/Sek., Class 10, U1, A1, FFP                                                                         |
| (Amazon)     | 8           | Servo Driver   | 1   | Xcsource PCA9685 16 Kanal 12 Bit PWM Servo Motor Treiber IIC Schnittstelle I2C Modul für Arduino Roboter TE477 von XCSOURCE                                            |
| (Amazon)     | 9           | Pi power       | 1   | Externer Akku Power Bank Power Pack Ladegerät 5200mAh Tragbar Handy Ladegerät für iPhone X 8 8Plus 7 6s 6Plus, iPad, Samsung Galaxy und weitere Smartphones von TESSAN |
| (Amazon)     | 32          | Camera         | 1   | SainSmart Wide Angle Fish-Eye Camera Lenses for Raspberry Pi Arduino von SainSmart                                                                                     |
| (Amazon)     | 13          | Parts          | 1   | Misc electro parts                                                                                                                                                     |
| HobbyKing    | 110         | Car            | 1   | Basher 1/16 4wd mini monster truck v2 - bad bug (arr)                                                                                                                  |
| HobbyKing    | 17          | Batteries      | 2   | Turnigy 1700mah 2s 20c lipo-pack (anzüge 1/16 monster beatle, sct & buggy)                                                                                             |
| HobbyKing    | 27          | Charger        | 1   | Accucell s60 ac-ladegerät (eu-stecker)                                                                                                                                 |
| Konsolenkost | 33          | DS3 Controller | 1   | 1 x PS3 - Original DualShock 3 Wireless Controller #schwarz [Sony] (sehr guter Zustand) (gebraucht)                                                                    |
|              | **306**         |                |     |                                                                                                                                                                      |

The major difference being the base car. The Exceed Magnet is basically sold out in Europe, I contacted two suppliers in Berlin who informed me that they will not be restocking either. So I spent an inordinate amount of time trying to decide on an alternative 1/16 truck chassis, with the requirement of shipping from Europe. If you allow for more shipping time/expense, you can of course expand your search. 

Here's what I came up with;

|  ![car](/Post1_2018FEB02/Car1.jpg) |  ![chassis](/Post1_2018FEB02/Chassis.jpg) | 
|:---:|:---:|
| Car | Chassis |

There are two models selling with the same chassis. The AAR version ships from Europe, other variants may not. 

Here's the marketing video;

[![Video](https://img.youtube.com/vi/GdtnAzs16lQ/0.jpg)](https://www.youtube.com/watch?v=GdtnAzs16lQ)

Next steps: 
- **Mounting:** Obviously the stock 3D printed parts are not going to fit, so will need to make a new design. I am considering laser-cut acrylic. I want to make a design with flexibility for mounting more sensors. I will wait until the car arrives to make measurements. 
- **Software Installation**

