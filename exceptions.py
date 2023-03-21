'''
This module contains useful Exceptions used in other modules.  
'''

#controller
class ConversionError(RuntimeError):
    '''
    This is an error used by the controller raised during the conversion from molarity to volume.  
    '''
    def __init__(self, reagent, molarity, total_vol, ratio, empty_reagents):
        '''
        This exception really encapsulates two errors, a volume transfer that is too small,
        or a volume transfer that is trying to transfer from a solution you've run out of.
        In the former case, empty_reagents will be empty. In the later case, empty reagents
        will contain a list of reagents that could be refilled in order to make this work.  
        params:  
            str: reagent: the reagent that is being used  
            float molarity: the desired molarity  
            float total_vol: the total volume in the container you're filling  
            float ratio: the ratio of total volume. See convert for details  
            list<str> empty_reagents: a list of chemical names that could be used, but do not
              have sufficient volume left for the transfer.  
        '''
        self.reagent = reagent
        self.molarity = molarity
        self.total_vol = total_vol
        self.ratio = ratio
        self.empty_reagents = empty_reagents
        print(reagent)

#ot2_robot
class EmptyReagent(RuntimeError):
    '''
    This is used for when you run out of a reagent on a container and try to aspirate from it.  
    Specifically it can be used for triggering a make.  
    '''

    def __init__(self, message, chem_name):
        '''
        params:  
            str message: the message associated with the error  
            str chem_name: the chemical name of this reagent
        '''
        self.chem_name = chem_name
        print(chem_name)
        super().__init__(message)
