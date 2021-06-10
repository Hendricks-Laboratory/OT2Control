from abc import ABC


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
        self.vol = vol
        self.labware = labware
        self.loc = loc
        self.height = self.update_height()

    @abstractmethod
    def update_height(self):
        pass

    def update_vol(self, aspvol):
        self.volume = self.volume - aspvol



    
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
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_15/2)**2))+(height_bottom_cylinder - tip_depth)
            
class big_tube(Tubes):
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
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_50/2)**2)) + (height_bottom_cylinder - tip_depth) 

class tube2000ul(Tubes):
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
        self.height = ((self.vol - vol_bottom_cylinder)/(math.pi*(diameter_2/2)**2)) + (height_bottom_cylinder - tip_depth)

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


class OT2_Controller():
    """
    The big kahuna. This class contains all the functions for controlling the robot
    ATTRIBUTES:
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
    """




class tip_rack():
    """
    Class to represent a rack of pipet tips
    You might want to remove this class. I'ts starting to look like a glorified dictionary
    for opentrons objects
    ATTRIBUTES:
        str name: the name we use to refer to this rack
        Obj labware: the Opentrons labware that this corresponds to
    """
    #NOTE you're going to need to get the first tip from the sheets in order to construct the 
    #labware with the appropriate tip!
    def __init__(self, name, labware):
        self.name = name
        self.labware = labware 


class Tube(Container):
    """
    Tube represents a test tube container, but it is still abstract. 
    INHERITED ATTRIBUTES:
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS:
        update_height void
    METHODS:
        update_vol(float aspvol) void: eupdates the volume upon an aspiration
        
    """
    def __init__(self, name, vol, labware, loc):
        super().__init__(name,vol, labware, loc)
