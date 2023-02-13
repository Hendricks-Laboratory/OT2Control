import pandas as pd

blankwellRow = pd.read_csv("CNH_040_full.csv")
blankwellRow = blankwellRow.loc[blankwellRow['well name'] == 'blank']
blankwellRow = blankwellRow.iloc[0]

blankwellRow[2] = 'BLANK'

df2 = pd.read_csv("CNH_039_full.csv")
df3 = pd.read_csv("CNH_033_full.csv")
df4 = pd.read_csv("CNH_017_full.csv")
df5 = pd.read_csv("CNH_016_full.csv")
df6 = pd.read_csv("CNH_013_full_df.csv")
df7 = pd.read_csv("CNH_008_full.csv")

dataframes = [df2, df3, df4, df5, df6, df7]
names = ["CNH_039.csv", "CNH_033.csv", "CNH_017.csv", "CNH_016.csv", "CNH_013.csv", "CNH_008.csv"]
paths = []
i = 0
for data in dataframes:
	data.loc[len(data.index)] = blankwellRow 
#	print(data.tail())
	data.to_csv("CSVsWithBlankWells/" + names[i], mode="x")
	paths.append("CSVsWithBlankWells/" + names[i])
	i += 1

finaldf = pd.concat(
	map(pd.read_csv, paths), ignore_index=True)

finaldf.to_csv("concatenatedData.csv", mode="x")
