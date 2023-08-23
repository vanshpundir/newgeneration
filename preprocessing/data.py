import pandas as pd

class Data:

    def __init__(self):
        self.df = pd.read_csv("data/api_data.csv")
        self.title = self.df["title"]
        self.description = self.df['description']
        self.url = (self.df["url"])
        self.image = (self.df["image"])
        self.allcolumns = (self.df.columns)
