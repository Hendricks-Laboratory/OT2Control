# OT2Control
software to provide detailed control of Opentrons OT2 robot
## Overview  
The code is designed to be run from two computers. The controlling computer uses a 
Protocol Executor object to run a protocol from googlesheets. The raspberry pi on the robot runs 
a recieving code to take commands from the executor and run them on the robot.
## Contents  
1. controller.py: code for the controller  
2. ot2\_robot.py: code for the robot  
3. Armchair/: code for interapplication communication. Start with armchair\_spec.txt  
4. df\_utils: assorted helper functions  
5. LabwareDefs/: Opentrons Definitions for the platereader  
6. excel\_spreadsheet\_preconditoins.txt: specs for google sheets  
