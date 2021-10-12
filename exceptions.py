'''
This module contains useful Exceptions used in other modules.  
'''

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
            str product: the chemical name of the product that tripped this error  
            list<str> empty_reagents: a list of chemical names that could be used, but do not
              have sufficient volume left for the transfer.  
        '''
        self.reagent = reagent
        self.molarity = molarity
        self.total_vol = total_vol
        self.ratio = ratio
        self.empty_reagents = empty_reagents
