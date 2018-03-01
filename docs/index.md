# Current goals

Get a minimum viable vehicle running in a month! 

- Vehicle drives autonomously between taped lines
- Duct tape, chewing gum, and superglue allowed! 

[(Github Repo with source)](https://github.com/MarcusJones/ai.drive)

___

# Feb. 28, 2018 - It's alive! 

There were two main areas of progress so far; hardware and software. 

In both areas, the official project documentation is *outstanding*, and I will just summarize and highlight the differences and challenges. 

 [Project instructions are here](http://docs.donkeycar.com/). 


### Hardware
1. Get 

Here is the vehicle, in the nude;
![chassis](/Post2_2018FEB28/OriginalChassis.jpg)

As mentioned in the first post, the main difference in chassis, compared to the 'stock' car, is the battery pack. In this case, the 
double 1700mah 2s 20c batteries store significantly more energy, but critically, also have a higher discharge rate, which is represented by the **C-Rating```. The table below lists the key parameters;

```Max Current Draw = Capacity x C-Rating```


| Battery pack                                | Technology | Voltage <br> [V] | Capacity <br> [mAH] | Discharge <br> [C-Rating] | Configuration <br>[S] |
|---------------------------------------------|------------|------------------|---------------------|---------------------------|-----------------------|
| **'Stock' project car  <br> Exceed Magnet** | NiMh       | 7.2              | 1100                | Much less than 20C        | ?                     |
| **HobbyKing Bad Bug**           <br>        | LiPo       | 14.4             | 3400                | More than 20C             | 2S * 2 = **4S1P**     |
| **50% Bad-Bug  **    <br>   | LiPo       | 7.2              | 1700                | 20C                       | 2S * 2 = **2S1P**     |

7.2V 1100mAh Ni-MH 

![hardware](/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg)

![3dprint](/Post2_2018FEB28/3DprintTest.png, =250x250)

![electronics](/Post2_2018FEB28/Electronics.png)

<img src="/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" width="48">

<img src="/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" height = 100px width="48">



<img src="https://marcusjones.github.io/ai.drive/Post2_2018FEB28/BuiltUpChassis_SMALL.jpg" height = 100px width="48">



### Software

1. Install software to the Raspberry Pi
	a. Write the project disk image to the SD card

1. Installing software on the host (linux on my laptop)
	a. Tensorflow and Keras
	a. Clone the source for the 'donkeycar' project, the python code for running the vehicle
	a. 





http://localhost:8887

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

