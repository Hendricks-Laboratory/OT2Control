import pandas as pd

dfs = ["CNH_0081.csv","CNH_0161.csv","CNH_0171.csv","CNH_0331.csv","CNH_0391.csv"]

for i in dfs:
	df = pd.read_csv(i)
	df = df[:-1]
	df.to_csv(i[:-4] + 2 + i[-4:], "x")
