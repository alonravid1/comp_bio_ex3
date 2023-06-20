import pandas as pd

# create dataframe from results file, get max accuracy row
with open('results0.csv', 'r') as res_file:
    df = pd.read_csv(res_file)