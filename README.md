# OT2Control
software to provide detailed control of Opentrons OT2 robot
## Overview  
The code is designed to be run from two computers. The controlling computer uses a 
ProtocolExecutor object to run a protocol from googlesheets. The raspberry pi on the robot runs 
a recieving code to take commands from the executor and runs them on the robot.
## Contents  
1. controller.py: code for the controller  
2. ot2\_robot.py: code for the robot  
3. Armchair/: code for interapplication communication. Start with armchair\_spec.txt  
4. df\_utils: assorted helper functions  
5. LabwareDefs/: Opentrons Definitions for the platereader  
6. excel\_spreadsheet\_preconditoins.txt: specs for google sheets  

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
    <type\> <name\>: <description\>  
    ...  
    <type\> <name\>: <description\>  
CONSTANTS:  
    <type\> <name\>: <description\>  
METHODS:  
    <signature\> <return type\>: description  
INHERITED METHODS: (optional)  
    <signature/name\> <return type\>, ..., <signature/name\> <return type\>  
INHERITED ATTRIBUTES: (optional)  
    <type\> <name\>, ..., <type\> <name\>  
'''  
#### Methods/Functions  
'''  
Description  
params:  
    <type\> <name\>: <description\>  
    ...  
    <type\> <name\>: <description\>  
returns:  
    <type\>: <description\>  
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

1. *Define a new operation* in the excel sheet, lets call it 'move\_reagent'.  
2. *Define Arguments*: This command will need three arguments, the src wellname, the dst wellname
it to, and the ratio of the volume to move. It is intuitive to use the existing structure of the
excel sheet for the
src and destination. Lets specify the destination by putting a 1 in the appropriate column, and
we'll specify the src in the reagent column. We also need to specify the amount to transfer, so
let's make a new column to hold that parameter left of the reagent column. --Note we could also
use a float instead of 1 in order to specify what percentage to transfer, but for demonstration
purposes, we'll add the argument.  
3. *Update excel\_specs*: As soon as you change the rxn sheet, you should update the 
excel\_specification with the changes
you made.  
4. Since we added a new argument, we'll need to change a little about how it's parsed. In our case
this is as simple as changing \_load\_rxn\_df(input\_data) in the controller to rename your new
column to something that's less wordy and doesn't have spaces (you don't need to do this. It's
convenience so you can have different names in code and on sheets)  
4. *Think about the information you need*: We now have a new type of row,
but we need to think about what information needs to be extracted
to send to the robot. In this case it's very simple. The robot will need to know, the src, the dst,
and the ratio of volume to transfer.  
5. *Make a new armchair packet type*:
    1. Go into the armchair\_spec and add a new packet type. This requires a string name, an 
    unused single byte bytecode, a desciption, and description of arguments. In our case, the
    args will be str src, str dst, float ratio.  
    2. Go into the armchair code and update the bidict with a mapping from string name
    to bytecode.  
6. *Create a helper function* with args (row, i). This function will need to parse a 
'move\_reagent' row of the reaction dataframe and extract the three parameters mentioned above.
These parameters should then be sent over the Armchair portal using send\_pack.  
7. *Update execute\_protocol\_df* in the controller to check if the operation is 'move\_reagent'
if it is, you should call your helper.  
8. *Create a new command type for the robot* in ot2\_robot.py, in execute. Add a new condition
to check if the command is 'move\_reagent'. Within that condition, call
self.\_exec\_move\_reagent(\*arguments) (we'll implement this private helper in the next step),
and after that, return 1. --Note. You must return 1 because the execute command has a return type
of int, specifying an exit status. 1 indicates ok. 0 indicates closed, i.e. the connection between
this robot instance and controller has been severed, and it is safe to destruct this robot.  
9. *Create a helper function for the robot*. This is where we'll implement the helper we called
under execute, self.\_exec\_move\_reagent(\*arguments). You should unpack arguments into the
parameters you need for this function, and then write the code to make the robot do what
you need. There are likely already helper functions defined in ot2\_robot that you should use to
help, as well as other class attributes. In our specific case it will look something like the
following:
    1. Access the containers in self.containers by their chemical name.  
    2. From src, check the volume attribute.  
    3. Call self.\_liquid\_transfer with $1\over 2$ the volume of src (or whatever fraction you chose)
from src to dst.  
10. *That's it!* To test you can run everything locally, and just enter 'n' after the simulation,
or you can run it on the robot and platereader without physically moving anything by runing
the script with the -s flag, and entering 'y' after the simulation.  
