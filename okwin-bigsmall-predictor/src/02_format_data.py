# src/02_format_data.py
import pandas as pd

df = pd.read_csv("../data/wingo_batch_results.csv")
df['Period'] = df['Period'].astype(str)

df['Year']   = df['Period'].str[0:4]
df['Month']  = df['Period'].str[4:6]
df['Day']    = df['Period'].str[6:8]
df['Hour']   = df['Period'].str[8:10]
df['Minute'] = df['Period'].str[10:12]
df['Second'] = df['Period'].str[12:14]
df['Serial'] = df['Period'].str[14:]

df['Datetime'] = pd.to_datetime(df['Period'].str[:14], format='%Y%m%d%H%M%S')

cols = ['Period','Datetime','Year','Month','Day','Hour','Minute','Second','Serial','Big_Small']
df[cols].to_csv("../data/wingo_game_filtered1.csv", index=False)
print("Formatted data saved → data/wingo_game_filtered1.csv")