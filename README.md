# OT2Control
software to provide detailed control of Opentrons OT2 robot

## Overview  
The code is designed to be run from two computers. The controlling computer uses a 
ProtocolExecutor object to run a protocol from googlesheets. The raspberry pi on the robot runs 
a recieving code to take commands from the executor and runs them on the robot.

## Usage Guide  
This guide is intended for a nontechnical audience.  
1. Create a reaction sheet in accordance the the preconditions explained in
excel\_spreadsheet\_precoditions.txt  
2. copy the key of your worksheet into the reaction key spreadsheet.  
3. Make sure that the PlateReader software is not running. If it is, close the window.  
4. move into the git directory and run the command `python controller.py`. (Note, for advanced
users, controller.py has a cli to skip later input stages.
run `python controller.py -h` for more information)  
5. You will be prompted to specify the rxn\_sheet\_name. Enter the name of your google sheet  
6. You will be prompted if you want to use cache, select no. The cache can be used if you want to
run the exact same reaction as you ran previously without changing anything in the google sheets.
This is mostly useful for debugging the software.  
7. A precheck simulation will be run. If your spreadsheet fails to meet preconditions, the code
will exit and ask you to fix the errors. If your spreadsheet failed, fix the errors and go to step
3.  
8. Once the simulation completes, you will be asked if you would like to run on the robot and
platereader. Enter 'y'.  
9. All data from the reaction is stored in Controller\_Out/\<Ouput Dir\>.  

*Note*: The above steps assume that the ot2\_robot code is running, which is almost always the
case. This code is designed to be very robust, and should only need to be rerun when updated or
when the robot is powered off. To run the code, ssh into the robot and move into the directory,
`OT2_Control`, and run the command `python ot2_robot.py`. (There's also usually a tmux session
named *run* that is already in the directory)

## Contents  
1. controller.py: code for the controller  
2. ot2\_robot.py: code for the robot  
3. Armchair/: code for interapplication communication. Start with armchair\_spec.txt  
4. df\_utils: assorted helper functions  
5. LabwareDefs/: Opentrons Definitions for the platereader  
6. excel\_spreadsheet\_preconditions.txt: specs for google sheets  
7. ml\_models.py: template code and implementations for ml models  

## Documentation
Documentation for the code can be found here https://science356lab.github.io/OT2Control/

## Guide For Contributing
### Style
The code follows an object oriented model, and global functions should be used only for the
driver code in controller.py and ot2\_robot.py, or in df\_utils for simple helper functions that 
are 
clearly not a part of a larger class. Global variables should not be used. Constants are ok if 
documented.  
### Comments
All docstrings are formatted in markdown, (so you must add two spaces to the end of the line to
terminate it)  
All Classes, Modules, and methods/functions should have a docstring. The docstring format of a 
module is freeform. The formats for the other types are shown below.  
specified below.  
#### Classes
'''  
Description of class  
ATTRIBUTES:  

- <type\> <name\>: <description\>  
- ...  
- <type\> <name\>: <description\>  

CONSTANTS:  

- <type\> <name\>: <description\>  

METHODS:  

- <signature\> <return type\>: description  

INHERITED METHODS: (optional)  

- <signature/name\> <return type\>, ..., <signature/name\> <return type\>  

INHERITED ATTRIBUTES: (optional)  

- <type\> <name\>, ..., <type\> <name\>  

'''  
#### Methods/Functions  
'''  
Description  
params:  
 
- <type\> <name\>: <description\>  
- ...  
- <type\> <name\>: <description\>  
 
returns:  

- <type\>: <description\>  

'''  
### Abstractions
The code implements a number of useful abstractions that should be preserved as new features are
added.  
1. The code is designed to be run with the ot2\_robot server on another machine. Therefore, all
input/output should take place on the controller.  
2. The only systems that the ot2\_robot code interacts with are the os it runs on (to dump local files), and the controller. i.e. All google sheets i/o is handled by the controller.  
3. The exact locations of chemicals on the robot are not visible to the controller.
--This abstraction is very leaky (because the platereader needs access to exact locations)
, and may need to be replaced later.  
4. The controller does not have access to any opentrons API information, or details about labware, e.g. volumes, what locations they have.  
5. The recipt of ready commands is an Armchair level process, and is not visible to client code.  
### Example Process For Feature implementation
This section guides you through the process of implementing a new feature to transfer a fraction
of the contents of a chemical into another well. e.g. move half of the liquid in 'R1' to 'P2'.
The steps are presented sequentially in an intuitive order that more or less follows the control
flow of the program in execution, but, of course, it make no difference in what order these are
implemented.
Most features will adhere to a similar pattern.  

1. **Define a new operation** in the excel sheet, lets call it 'move\_reagent'.  
2. **Define Arguments**: This command will need three arguments, the src wellname, the dst wellname
it to, and the ratio of the volume to move. It is intuitive to use the existing structure of the
excel sheet for the
src and destination. Lets specify the destination by putting a 1 in the appropriate column, and
we'll specify the src in the reagent column. We also need to specify the amount to transfer, so
let's make a new column to hold that parameter left of the reagent column. --Note we could also
use a float instead of 1 in order to specify what percentage to transfer, but for demonstration
purposes, we'll add the argument.  
3. **Update excel\_specs**: As soon as you change the rxn sheet, you should update the 
excel\_specification with the changes
you made.  
4. **Update parsing**: Since we added a new argument, we'll need to change a little about how
it's parsed. In our case
this is as simple as changing \_load\_rxn\_df(input\_data) in the controller to rename your new
column to something that's less wordy and doesn't have spaces (you don't need to do this. It's
convenience so you can have different names in code and on sheets)  
4. **Think about the information you need**: We now have a new type of row,
but we need to think about what information needs to be extracted
to send to the robot. In this case it's very simple. The robot will need to know, the src, the dst,
and the ratio of volume to transfer.  
5. **Make a new armchair packet type**:
    1. Go into the armchair\_spec and add a new packet type. This requires a string name, an 
    unused single byte bytecode, a desciption, and description of arguments. In our case, the
    args will be str src, str dst, float ratio.  
    2. Go into the armchair code and update the bidict with a mapping from string name
    to bytecode.  
6. **Create a helper function** with args (row, i). This function will need to parse a 
'move\_reagent' row of the reaction dataframe and extract the three parameters mentioned above.
These parameters should then be sent over the Armchair portal using send\_pack.  
7. **Update execute\_protocol\_df** in the controller to check if the operation is 'move\_reagent'
if it is, you should call your helper.  
8. **Create a helper function for the robot**. This is where we'll implement the helper we called
under execute, self.\_exec\_move\_reagent(\*arguments). You should unpack arguments into the
parameters you need for this function, and then write the code to make the robot do what
you need. There are likely already helper functions defined in ot2\_robot that you should use to
help, as well as other class attributes. In our specific case it will look something like the
following:
    1. Access the containers in self.containers by their chemical name.  
    2. From src, check the volume attribute.  
    3. Call self.\_liquid\_transfer with $1\over 2$ the volume of src (or whatever fraction you chose)
from src to dst.  
9. **Register your helper**: In order for the robot to invoke your helper function when it recieves
an armchair command, you must register it with the decorator, `exec_func(str name, int exit_code,
bool send_ready, dict_exec_funcs)`. Going through these arguments:
    1. str name: the armchair command type, for us, 'move\_reagent'  
    2. int exit\_code: the exit code, almost always 1. 0 is used for an exit  
    3. bool send\_ready: If False, function will omit sending a ready. This is usually True.  
    4. dict exec\_funcs: The argument to this is always exec\_funcs, the registry dict.  
-- Note: your helper functions should have no return value, and should not send a ready command,
or take in the cid of the armchair command. These things are all handled by the decorated function,
*but* the original function is left as is and can still be accessed as a helper for other functions
 without sending ready commands or returning exit\_code.  
10. **That's it!** To test you can run everything locally, and just enter 'n' after the simulation,
or you can run it on the robot and platereader without physically moving anything by runing
the script with the -s flag, and entering 'y' after the simulation.  
