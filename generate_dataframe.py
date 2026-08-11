import os
import glob
import pandas as pd
import numpy as np
import polars as pl


##I'm kinda assuming here that your kmc outputs are in txt format, feel free to adjust if that is not the case!!!

genome_files = glob.glob("*.txt")

dfs = []

for file in genome_files:
    df = pd.read_csv(file, sep="\t", header=None, names=["kmer", "freq"])
    genome_name = os.path.splitext(os.path.basename(file))[0]
    df["genome"] = genome_name
    dfs.append(df)


combined_long = pd.concat(dfs, ignore_index=True)


matrix_df = combined_long.pivot_table(
    index="genome",
    columns="kmer",
    values="freq",
    fill_value=0
)


#THIS WILL ALSO WORK

matrix_df = (
    combined_long
    .set_index(["genome", "kmer"])["freq"]
    .unstack(fill_value=0)
)


matrix_df.to_csv("kmer_frequency_matrix.csv")

#convert to binary IF and ONLY IF you are using 12mers or higher

binary_df = (matrix_df > 0).astype(int)

##This is where you can import your label variable (MIC, FIC, or whatever u want to predict)
##I used log2 transformed MICs, I would keep this in mind when considering my W1 scores in the model
##Make sure the order aligns with the genome order in your kmer matrix
 
cazavi = pd.read_csv("abx/cazavi.csv")
merovabor = pd.read_csv("abx/merovabor.csv")
imirel = pd.read_csv("abx/imirel.csv")
mero = pd.read_csv("abx/mero.csv")

matrix_df_reset = matrix_df.reset_index()

ca_matrix = matrix_df_reset.merge(cazavi, on="genome", how="left")
mv_matrix = matrix_df_reset.merge(merovabor, on="genome", how="left")
ir_matrix = matrix_df_reset.merge(imirel, on="genome", how="left")
m_matrix = matrix_df_reset.merge(mero, on="genome", how="left")


###This part is really up to u, ideologically. I ended up not using a combined matrix and did everything separately
###You may find a concatenated matrix is for you, however. It led me to better predictions with lower k kmers


final_combined = pd.concat(
    [ca_matrix, mv_matrix, ir_matrix, m_matrix],
    ignore_index=True
)


final_combined = final_combined.fillna(0)


if "label" in final_combined.columns:
    final_combined["label"] = final_combined["label"].round()


final_combined.to_csv("km_final.csv", index=False)
final_combined.to_csv("km_final.csv", index=False)

#or


pl.from_pandas(df).write_csv('output.csv')


#or

df.to_feather("data.feather")

#If you store the data as a feather just be sure to use the following function to import for xgboost :)))))

df = pd.read_feather("data.feather")
