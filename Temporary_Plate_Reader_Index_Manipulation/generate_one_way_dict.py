import numpy as np
import pandas as pd 
import json
import sys
import opentrons
import opentrons.execute
from opentrons import protocol_api, simulate, types
def plate_reader_index_dict():
    '''
    this stuff less interesting because it just maps from integeer
   '''
#p4_labware.wells()[4],8:p4_labware.wells()[3],16:p4_labware.wells()[2],24:p4_labware.wells()[1],32:p4_labware.wells()[0],
#                88:p7_labware.wells()[0],80:p7_labware.wells()[1],72:p7_labware.wells()[2],64:p7_labware.wells()[3],56:p7_labware.wells()[4],
#                48:p7_labware.wells()[5],40:p7_labware.wells()[6],
#                     
#                1:p4_labware.wells()[9],9:p4_labware.wells()[8],17:p4_labware.wells()[7],25:p4_labware.wells()[6],
#                33:p4_labware.wells()[5],41:p7_labware.wells()[13],49:p7_labware.wells()[12],57:p7_labware.wells()[11],65:p7_labware.wells()[10],
#                73:p7_labware.wells()[9],81:p7_labware.wells()[8],89:p7_labware.wells()[7],
#                     
#                2:p4_labware.wells()[14],10:p4_labware.wells()[13],18:p4_labware.wells()[12],26:p4_labware.wells()[11],
#                34:p4_labware.wells()[10],42:p7_labware.wells()[20],50:p7_labware.wells()[19],58:p7_labware.wells()[18],66:p7_labware.wells()[17],
#                74:p7_labware.wells()[16],82:p7_labware.wells()[15],90:p7_labware.wells()[14],
#                     
#                3:p4_labware.wells()[19],11:p4_labware.wells()[18],19:p4_labware.wells()[17],27:p4_labware.wells()[16],35:p4_labware.wells()[15],
#                43:p7_labware.wells()[27],51:p7_labware.wells()[26],59:p7_labware.wells()[25],67:p7_labware.wells()[24],75:p7_labware.wells()[23],
#                83:p7_labware.wells()[22],91:p7_labware.wells()[21],
#                
#                4:p4_labware.wells()[24],12:p4_labware.wells()[23],20:p4_labware.wells()[22],28:p4_labware.wells()[21],36:p4_labware.wells()[20],
#                44:p7_labware.wells()[34],52:p7_labware.wells()[33],60:p7_labware.wells()[32],68:p7_labware.wells()[31],76:p7_labware.wells()[30],
#                84:p7_labware.wells()[29],92:p7_labware.wells()[28],
#                      
#                5:p4_labware.wells()[29],13:p4_labware.wells()[28],21:p4_labware.wells()[27],29:p4_labware.wells()[26],37:p4_labware.wells()[25],
#                45:p7_labware.wells()[41],53:p7_labware.wells()[40],61:p7_labware.wells()[39],69:p7_labware.wells()[38],77:p7_labware.wells()[37],
#                85:p7_labware.wells()[36],93:p7_labware.wells()[35],
#                    
#                6:p4_labware.wells()[34],14:p4_labware.wells()[33],22:p4_labware.wells()[32],30:p4_labware.wells()[31],38:p4_labware.wells()[30],
#                46:p7_labware.wells()[48],54:p7_labware.wells()[47],62:p7_labware.wells()[46],70:p7_labware.wells()[45],78:p7_labware.wells()[44],
#                86:p7_labware.wells()[43],94:p7_labware.wells()[42],
#                      
#                7:p4_labware.wells()[39],15:p4_labware.wells()[38],23:p4_labware.wells()[37],31:p4_labware.wells()[36],39:p4_labware.wells()[35],
#                47:p7_labware.wells()[55],55:p7_labware.wells()[54],63:p7_labware.wells()[53],71:p7_labware.wells()[52],79:p7_labware.wells()[51],
#                87:p7_labware.wells()[50],95:p7_labware.wells()[49],t():
    plate_reader_index = {
                     
                'A1':p4_labware.wells()[4],'A2':p4_labware.wells()[3],'A3':p4_labware.wells()[2],'A4':p4_labware.wells()[1],'A5':p4_labware.wells()[0],
                'A12':p7_labware.wells()[0],'A11':p7_labware.wells()[1],'A10':p7_labware.wells()[2],'A9':p7_labware.wells()[3],'A8':p7_labware.wells()[4],
                'A7':p7_labware.wells()[5],'A6':p7_labware.wells()[6],
                     
                'B1':p4_labware.wells()[9],'B2':p4_labware.wells()[8],'B3':p4_labware.wells()[7],'B4':p4_labware.wells()[6],
                'B5':p4_labware.wells()[5],'B6':p7_labware.wells()[13],'B7':p7_labware.wells()[12],'B8':p7_labware.wells()[11],'B9':p7_labware.wells()[10],
                'B10':p7_labware.wells()[9],'B11':p7_labware.wells()[8],'B12':p7_labware.wells()[7],
                     
                'C1':p4_labware.wells()[14],'C2':p4_labware.wells()[13],'C3':p4_labware.wells()[12],'C4':p4_labware.wells()[11],
                'C5':p4_labware.wells()[10],'C6':p7_labware.wells()[20],'C7':p7_labware.wells()[19],'C8':p7_labware.wells()[18],'C9':p7_labware.wells()[17],
                'C10':p7_labware.wells()[16],'C11':p7_labware.wells()[15],'C12':p7_labware.wells()[14],
                     
                'D1':p4_labware.wells()[19],'D2':p4_labware.wells()[18],'D3':p4_labware.wells()[17],'D4':p4_labware.wells()[16],'D5':p4_labware.wells()[15],
                'D6':p7_labware.wells()[27],'D7':p7_labware.wells()[26],"D8":p7_labware.wells()[25],'D9':p7_labware.wells()[24],'D10':p7_labware.wells()[23],
                'D11':p7_labware.wells()[22],'D12':p7_labware.wells()[21],
                
                'E1':p4_labware.wells()[24],'E2':p4_labware.wells()[23],'E3':p4_labware.wells()[22],'E4':p4_labware.wells()[21],'E5':p4_labware.wells()[20],
                'E6':p7_labware.wells()[34],'E7':p7_labware.wells()[33],'E8':p7_labware.wells()[32],'E9':p7_labware.wells()[31],'E10':p7_labware.wells()[30],
                'E11':p7_labware.wells()[29],'E12':p7_labware.wells()[28],
                      
                'F1':p4_labware.wells()[29],'F2':p4_labware.wells()[28],'F3':p4_labware.wells()[27],'F4':p4_labware.wells()[26],'F5':p4_labware.wells()[25],
                'F6':p7_labware.wells()[41],'F7':p7_labware.wells()[40],'F8':p7_labware.wells()[39],'F9':p7_labware.wells()[38],'F10':p7_labware.wells()[37],
                'F11':p7_labware.wells()[36],'F12':p7_labware.wells()[35],
                    
                'G1':p4_labware.wells()[34],'G2':p4_labware.wells()[33],'G3':p4_labware.wells()[32],'G4':p4_labware.wells()[31],'G5':p4_labware.wells()[30],
                'G6':p7_labware.wells()[48],'G7':p7_labware.wells()[47],'G8':p7_labware.wells()[46],'G9':p7_labware.wells()[45],'G10':p7_labware.wells()[44],
                'G11':p7_labware.wells()[43],'G12':p7_labware.wells()[42],
                      
                'H1':p4_labware.wells()[39],'H2':p4_labware.wells()[38],'H3':p4_labware.wells()[37],'H4':p4_labware.wells()[36],'H5':p4_labware.wells()[35],
                'H6':p7_labware.wells()[55],'H7':p7_labware.wells()[54],'H8':p7_labware.wells()[53],'H9':p7_labware.wells()[52],'H10':p7_labware.wells()[51],
                'H11':p7_labware.wells()[50],'H12':p7_labware.wells()[49]}
    return plate_reader_index
  

d = { "deck_pos": { "A1": "p4", "A2": "p4", "A3": "p4", "A4": "p4", "A5": "p4", "A13": "p7", "A11": "p7", "A10": "p7", "A9": "p7", "A8": "p7", "A7": "p7", "A6": "p7", "B1": "p4", "B2": "p4", "B3": "p4", "B4": "p4", "B5": "p4", "B6": "p7", "B7": "p7", "B8": "p7", "B9": "p7", "B10": "p7", "B11": "p7", "B12": "p7", "C1": "p4", "C2": "p4", "C3": "p4", "C4": "p4", "C5": "p4", "C6": "p7", "C7": "p7", "C8": "p7", "C9": "p7", "C10": "p7", "C11": "p7", "C12": "p7", "D1": "p4", "D2": "p4", "D3": "p4", "D4": "p4", "D5": "p4", "D6": "p7", "D7": "p7", "D8": "p7", "D9": "p7", "D10": "p7", "D11": "p7", "D12": "p7", "E1": "p4", "E2": "p4", "E3": "p4", "E4": "p4", "E5": "p4", "E6": "p7", "E7": "p7", "E8": "p7", "E9": "p7", "E10": "p7", "E11": "p7", "E12": "p7", "F1": "p4", "F2": "p4", "F3": "p4", "F4": "p4", "F5": "p4", "F6": "p7", "F7": "p7", "F8": "p7", "F9": "p7", "F10": "p7", "F11": "p7", "F12": "p7", "G1": "p4", "G2": "p4", "G3": "p4", "G4": "p4", "G5": "p4", "G6": "p7", "G7": "p7", "G8": "p7", "G9": "p7", "G10": "p7", "G11": "p7", "G12": "p7", "H1": "p4", "H2": "p4", "H3": "p4", "H4": "p4", "H5": "p4", "H6": "p7", "H7": "p7", "H8": "p7", "H9": "p7", "H10": "p7", "H11": "p7", "H12": "p7" }, "loc": { "A1": "E1", "A2": "D1", "A3": "C1", "A4": "B1", "A5": "A1", "A13": "A1", "A11": "B1", "A10": "C1", "A9": "D1", "A8": "E1", "A7": "F1", "A6": "G1", "B1": "E2", "B2": "D2", "B3": "C2", "B4": "B2", "B5": "A2", "B6": "G2", "B7": "F2", "B8": "E2", "B9": "D2", "B10": "C2", "B11": "B2", "B12": "A2", "C1": "E3", "C2": "D3", "C3": "C3", "C4": "B3", "C5": "A3", "C6": "G3", "C7": "F3", "C8": "E3", "C9": "D3", "C10": "C3", "C11": "B3", "C12": "A3", "D1": "E4", "D2": "D4", "D3": "C4", "D4": "B4", "D5": "A4", "D6": "G4", "D7": "F4", "D8": "E4", "D9": "D4", "D10": "C4", "D11": "B4", "D12": "A4", "E1": "E5", "E2": "D5", "E3": "C5", "E4": "B5", "E5": "A5", "E6": "G5", "E7": "F5", "E8": "E5", "E9": "D5", "E10": "C5", "E11": "B5", "E12": "A5", "F1": "E6", "F2": "D6", "F3": "C6", "F4": "B6", "F5": "A6", "F6": "G6", "F7": "F6", "F8": "E6", "F9": "D6", "F10": "C6", "F11": "B6", "F12": "A6", "G1": "E7", "G2": "D7", "G3": "C7", "G4": "B7", "G5": "A7", "G6": "G7", "G7": "F7", "G8": "E7", "G9": "D7", "G10": "C7", "G11": "B7", "G12": "A7", "H1": "E8", "H2": "D8", "H3": "C8", "H4": "B8", "H5": "A8", "H6": "G8", "H7": "F8", "H8": "E8", "H9": "D8", "H10": "C8", "H11": "B8", "H12": "A8" } }

#check that you have all the wells
all_wells = np.array(list(d['deck_pos'].keys()))
assert (np.unique(all_wells).shape[0] == 96), "don't have all wells"

#test that the wells are correctly partitioned into 4 and 5
deck_poses  = np.vectorize(d['deck_pos'].get)(all_wells)
p4_wells = all_wells[deck_poses == 'p4']
p7_wells = all_wells[deck_poses == 'p7']
assert (p4_wells.shape[0] % 8 == 0), "p4 wells not even square"
assert (p7_wells.shape[0] % 8 == 0), "p7 wells not even square"
assert (p4_wells.shape[0] + p7_wells.shape[0] == 96), "wrong number total p4+p7 wells"
no_7 = np.vectorize(lambda x: '7' not in x)
has_7 = np.vectorize(lambda x: '7' in x)
has_12 = np.vectorize(lambda x: '12' in x)
assert (no_7(p4_wells).sum() != 0), "there are greater valued rows in p4 than expected"
assert (has_7(p7_wells).sum() != 0), "expeceted to find col 7 in p7"
assert (has_12(p7_wells).sum() != 0), "p7 needs to go up to col 12"
#p7 is cols 7-12
#p4 is cols 1-4

protocol = opentrons.simulate.get_protocol_api('2.9')
with open('../LabwareDefs/plate_reader_4.json', 'r') as labware_def_file:
    labware_def = json.load(labware_def_file)
p4_labware = protocol.load_labware_from_definition(labware_def, 4,label='p4')
with open('../LabwareDefs/plate_reader_7.json', 'r') as labware_def_file:
    labware_def = json.load(labware_def_file)
p7_labware = protocol.load_labware_from_definition(labware_def, 7,label='p7')

new_d = plate_reader_index_dict()
s = pd.Series(new_d)
plate_reader_names = s.apply(lambda x: str(x)[str(x).find('p'):str(x).find('p')+2])
well_names = s.apply(lambda x: x._impl._name)
df = pd.DataFrame({'loc':well_names,'platereader':plate_reader_names})
final_dict = df.to_dict()
with open('platereader_dict.json','w') as json_dump:
    json.dump(final_dict, json_dump)
breakpoint()

