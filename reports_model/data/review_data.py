import pandas as pd

def fetch_reviews():
    data = [
        {"Review": "Great hotel, very clean!", "Rating": 5},
        {"Review": "Bad experience, rude staff.", "Rating": 1},
        {"Review": "Average stay, nothing special.", "Rating": 3},
    ]
    return pd.DataFrame(data)