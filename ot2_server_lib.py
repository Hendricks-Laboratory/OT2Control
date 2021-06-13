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
        _update_height void: updates self.height to hieght at which to pipet (a bit below water line)
    IMPLEMENTED METHODS:
        update_vol(float aspvol) void: updates the volume upon an aspiration
    """

    def __init__(self, name, vol, labware, loc):
        self.name = name
        self.contents = Contents(vol)
        self.labware = labware
        self.loc = loc
        self.height = self._update_height()
        self.vol = vol

    @abstractmethod
    def _update_height(self):
        pass

    def update_vol(self, aspvol):
        self.contents.update_vol(aspvol)
        self._update_height()



    
class small_tube(Container):
    #TODO update class name to be reflective of size
    """
    Spcific tube with measurements taken to provide implementations of abstract methods
    INHERITED ATTRIBUTES
        str name, float vol, Obj Labware, str loc
    INHERITED METHODS
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass15 = 6.6699 # grams
        reagent_mass = mass - avg_tube_mass15 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 

    def _update_height(self):
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
        _update_height void, update_vol(float aspvol) void,
    """
    def __init__(self, name, mass, labware, loc):
        density_water_25C = 0.9970479 # g/mL
        avg_tube_mass50 = 13.3950 # grams
        reagent_mass = mass - avg_tube_mass50 # N = 1 (in grams) 
        vol = (reagent_mass / density_water_25C) * 1000 # converts mL to uL
        super().__init__(name,vol, labware, loc)
       # 15mm diameter for 15 ml tube  -5: Five mL mark is 19 mm high for the base/noncylindrical protion of tube 
        
    def _update_height(self):
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
        _update_height void, update_vol(float aspvol) void,
    """
    
    def __init__(self, name, mass, labware, loc):
        density_water_4C = 0.9998395 # g/mL
        avg_tube_mass2 =  1.4        # grams
        reagent_mass = mass - avg_tube_mass2 # N = 1 (in grams) 
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def _update_height(self):
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
            _update_height void, update_vol(float aspvol) void,
    """

    def __init__(self, name, labware, loc, vol=0):
        #vol is defaulted here because the well will probably start without anything in it
        vol = (self.mass / density_water_4C) * 1000 # converts mL to uL
        super().__init__(name, vol, labware, loc)
           
    def _update_height(self):
        #TODO develop an update height method for wells.
        pass

#TODO right now Contents is created inside the constructor of the container, but contents should
#be passed in pre baked. This requires the code for initial stoicheometry to be moved to a main
#in the initialization code of this or maybe the client
#TODO docs not updated on these
class Contents(ABC):
    """
    There is a difference between a reagent and a product, and oftentimes it has to do with
    reagents being dilutions and products being mixtures. Products need information
    about what is in them, dilutions need concentrations. These attributes relate to
    the content of the container, not the container itself. Every container has a
    substance inside it.
    This is a purely architectural class. It exists to preserve the seperation of things.
    ABSTRACT ATTRIBUTES:
        float conc: the concentration (varies in meaning based on child)
        float mass: the mass of the reagent
    METHODS:
        update_vol(float aspvol) void: updates the volume of the contents left
    """
    def __init__(self, conc, mass):
        self.conc=conc
        self.mass=mass

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
    def __init__(self, conc, mass, name, molecular_weight):
        self.molecular_weight=molecular_weight
        super().__init__(conc,mass)

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
    def __init__(self, conc, mass, components):
        self.components = components
        super().__init__(conc,mass)
        return

    def add(self, name, vol,timestamp=datetime.datetime.now()):
        #TODO you'll wnat to format the timestamp however you like
        self.vol += vol
        self.components += (timestamp, name, vol)
        return

class OT2_Controller():
    """
    The big kahuna. This class contains all the functions for controlling the robot
    ATTRIBUTES:
        Dict<str, str> OPENTRONS_LABWARE_NAMES: This is a constant dictionary that is used to
          translate from human readable names to opentrons labware object names.
        Dict<str, Container> containers: maps from a common name to a Container object
        Dict<str, Obj> tip_racks: maps from a common name to a opentrons tiprack labware object
        Dict<str, Obj> labware: maps from labware common names to opentrons labware objects. tip racks not included?
        Opentrons...ProtocolContext protocol: the protocol object of this session

    """

    self._OPENTRONS_LABWARE_NAMES = {'96_well_plate_1':'corning_96_wellplate_360ul_flat','96_well_plate_2':'corning_96_wellplate_360ul_flat','24_well_plate_1':'corning_24_wellplate_3.4ml_flat','24_well_plate_2':'corning_24_wellplate_3.4ml_flat','48_well_plate_1':'corning_48_wellplate_1.6ml_flat','48_well_plate_2':'corning_48_wellplate_1.6ml_flat','tip_rack_20':'opentrons_96_tiprack_20ul','tip_rack_300':'opentrons_96_tiprack_300ul','tip_rack_1000':'opentrons_96_tiprack_1000ul','tube_holder_10_1':'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical','tube_holder_10_2':'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical','20uL_pipette':'p20_single_gen2','300uL_pipette':'p300_single_gen2','1000uL_pipette':'p1000_single_gen2'}

    def __init__(self, simulate):
        '''
        params:
            bool simulate: if true, the robot will run in simulation mode only
            credentials: read from a local json. Required for most google sheets i/o
        postconditions:
            protocol has been initialzied
            containers and tip_racks have been created
            labware has been initialized
            CAUTION: the values of tip_racks and containers must be sent from the client.
              it is the client's responsibility to make sure that these are initialized prior
              to operating with them
        '''
        self.tip_racks = {}
        self.containers = {}
        self.labware = {}
        if simulate:
            self.protocol = opentrons.simulate.get_protocol_api('2.9')# define version number and define protocol object
        else:
            self.protocol = opentrons.execute.get_protocol_api('2.9')
            self.protocol.set_rail_lights(on = True)
            self.protocol.rail_lights_on 
        self.protocol.home() # Homes the pipette tip

    def make_fixed_objects():
        self.protocol.max_speeds['X'] = 100
        self.protocol.max_speeds['Y'] = 100

        #with open('Cuvette Rack.json') as labware_file1:
         #   labware_def1 = json.load(labware_file1)
        with open('/var/lib/jupyter/notebooks/JSON/plate_reader_4.json') as labware_file2:
            labware_def2 = json.load(labware_file2)
        with open('/var/lib/jupyter/notebooks/JSON/plate_reader_7.json') as labware_file3:
            labware_def3 = json.load(labware_file3)
            
        labware['platereader4'] = protocol.load_labware_from_definition(labware_def2, 4)
        labware['platereader7'] = protocol.load_labware_from_definition(labware_def3, 7)
        
        if temperature_module_response == 'y' or temperature_module_response == 'yes':
            labware['temp_mod'] = protocol.load_module('temperature module gen2', 3)
            labware['temp_mod'].set_temperature(set_temperature_response)
        #labware['cuvette'] = protocol.load_labware_from_definition(labware_def1, 3)
