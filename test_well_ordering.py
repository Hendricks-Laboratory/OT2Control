import pandas as pd

wellmap = pd.read_csv('Eve_Files/Eve_Files/translated_wellmap.tsv', '\t')
ind = wellmap.loc[wellmap['chem_name'] == 'WaterC1.0'].index
wellmap.drop(index=ind,inplace=True)
wellmap['ordering'] = wellmap['chem_name'].apply(lambda x: int(x[x.find('P')+1:x.find('C')])).astype(int)
wellmap.sort_values('ordering',inplace=True)
wellmap.set_index('ordering',inplace=True)
print(wellmap[['loc', 'deck_pos']])
