import pandas as pd
from sklearn.model_selection import train_test_split


data = []
for f in ["Data2021/content2019-01-v3.tsv", "Data2021/content2019-02-v3.tsv", "Data2021/content2019-03-v3.tsv"]:
    df = pd.read_csv(open(f, "rb"), sep="\t")
    data.append(df)
data = pd.concat(data, axis=0, ignore_index=True)

train, test = train_test_split(data, test_size=0.2)

train.to_pickle("./train.pkl")
test.to_pickle("./test.pkl")
