import pandas as pd
path = input("enter csv name:")

df = pd.read_csv(path)

blankwellRow = pd.read_csv("CNH_040_full.csv")
blankwellRow = blankwellRow.loc[blankwellRow['well name'] == 'blank']
blankwellRow = blankwellRow.iloc[0]

blankwellRow[2] = 'BLANK'

print(blankwellRow)
