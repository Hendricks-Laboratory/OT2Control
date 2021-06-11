from abc import ABC
from abc import abstractmethod


class Container(ABC):
    """
    
    Abstract container class to be overwritten for well, tube, etc.
    ABSTRACT ATTRIBUTES:
        str name: the common name we use to refer to this container
        float vol: the volume of the liquid in this container in ul
        Obj Labware: a pointer to the Opentrons Labware object of which this is a part
        str loc: a location on the labware object (e.g. 'A5')
    ABSTRACT METHODS:
        update_height void: updates self.height to hieght at which to pipet (a bit below water line)
    IMPLEMENTED METHODS:
        update_vol(float aspvol) void: updates the volume upon an aspiration
    """

    def __init__(self, name, vol, labware, loc):
        self.name = name
        self.contents = Contents(vol)
        self.labware = labware
        self.loc = loc
        self.height = self.update_height()

    @abstractmethod
    def update_height(self):
        pass

    def update_vol(self, aspvol):
        self.contents.update_vol(aspvol)



    
class small_tube(Container):
    #TODO update class name to be reflective of size
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass15 = 6.6699 # grams
        reagent_mass = mass - avg_tube_mass15 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 

    def update_height(self):
        diameter_15 = 14.0 # mm (V1 number = 14.4504)
        vol_bottom_cylinder = 2000 # uL
        height_bottom_cylinder = 30.5  #mm
        tip_depth = 5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_15/2)**2))+(height_bottom_cylinder - tip_depth)
            
class big_tube(Container):
    #TODO update class name to be reflective of size
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        update_height void, update_vol(float aspvol) void,
    """
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass50 = 13.3950 # grams
        reagent_mass = mass - avg_tube_mass50 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name,vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 
        
    def update_height(self):
        diameter_50 = 26.50 # mm (V1 number = 26.7586)
        vol_bottom_cylinder = 5000 # uL
        height_bottom_cylinder = 21 #mm
        tip_depth = 5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_50/2)**2)) + (height_bottom_cylinder - tip_depth) 

class tube2000ul(Container):
    """
    2000ul tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
         str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_4C = 0.9998395 # g/mL
        avg_tube_mass2 =  1.4        # grams
        reagent_mass = mass - avg_tube_mass2 # N = 1 (in grams) 
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def update_height(self):
        diameter_2 = 8.30 # mm
        vol_bottom_cylinder = 250 #uL
        height_bottom_cylinder = 10.5 #mm
        tip_depth = 4.5 # mm
        self.height = ((self.contents.vol - vol_bottom_cylinder)/(math.pi*(diameter_2/2)**2)) + (height_bottom_cylinder - tip_depth)

class well96(Container):
    """
        a well in a 96 well plate
        INHERITED ATTRIBUTES
             str name, float vol, Obj Labware, str loc
        INHERITED METHODS
            update_height void, update_vol(float aspvol) void,
    """

    def __init__(self, name, labware, loc, vol=0):
        #vol is defaulted here because the well will probably start without anything in it
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def update_height(self):
        #TODO develop an update height method for wells.
        pass

#TODO right now Contents is created inside the constructor of the container, but contents should
#be passed in pre baked. This requires the code for initial stoicheometry to be moved to a main
#in the initialization code of this or maybe the client
class Contents(ABC):
    """
    There is a difference between a reagent and a product, and oftentimes it has to do with
    reagents being dilutions and products being mixtures. Products need information
    about what is in them, dilutions need concentrations. These attributes relate to
    the content of the container, not the container itself. Every container has a
    substance inside it.
    This is a purely architectural class. It exists to preserve the seperation of things.
    ABSTRACT ATTRIBUTES:
        float vol: the amount of liquid in the container
    METHODS:
        update_vol(float aspvol) void: updates the volume of the contents left
    """
    def __init__(self, vol):
        self.vol=vol

    def update_vol(self, aspvol):
        self.vol = self.vol - aspvol

class Dilution(Contents):
    """
    A contents with a concentration meant to represent a stock solution
    INHERITED ATTRIBUTES:
        float vol
    ATTRIBUTES:
        float conc: the concentration
    INHERITED METHODS:
        update_vol(float aspvol)
    """
    def __init__(self, vol, conc):
        self.conc = conc
        super().__init__(vol)

class Mixture(Contents):
    """
    This is probably a product or seed
    It keeps track of what is in it
    INHERITED ATTRIBUTES:
        float vol
    ATTRIBUTES:
        list<tup<str, vol>> components: reagents and volumes in this mixture
    INHERITED METHODS:
        update_vol(float aspvol)
    """
    def __init__(self, vol, components):
        self.components = components
        super().__init__(vol)
        return

    def add(self, name, vol):
        self.vol += vol
        self.components += (name, vol)
        return


class OT2_Controller():
    """
    The big kahuna. This class contains all the functions for controlling the robot
    ATTRIBUTES:
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
        oauth2client.ServiceAccountCredentials credentials: credentials read from a local json
          file that are used in most google sheets i/o
    """
    def __init__(self, simulate):
        #params:
            #bool simulate: if true, the robot will run in simulation mode only
            #credentials: read from a local json. Required for most google sheets i/o
        if simulate:
            protocol = opentrons.simulate.get_protocol_api('2.9')# define version number and define protocol object
        else:
            protocol = opentrons.execute.get_protocol_api('2.9')
            protocol.set_rail_lights(on = True)
            protocol.rail_lights_on 
        protocol.home() # Homes the pipette tip
