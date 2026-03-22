import pandas as pd
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100000)
df=pd.read_csv("export.csv", sep="\t")

print(df)