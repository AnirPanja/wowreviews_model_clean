import pandas as pd
import os

# Paths
data_path = os.path.join(os.path.dirname(__file__), "data", "tripadvisor_hotel_reviews.csv")

# Load the CSV into a DataFrame and limit to first 9000 entries
df = pd.read_csv(data_path).head(20000)  # Limits to first 9000 rows

# Filter for neutral reviews (Rating == 3)
neutral_df = df[df['Rating'] == 3]

# Print the result
print("Total neutral reviews within first 9000 entries:", len(neutral_df))
print("Rating distribution within first 9000 entries:\n", df['Rating'].value_counts().sort_index())
print("Sample neutral review:\n", neutral_df['Review'].iloc[0] if len(neutral_df) > 0 else "No neutral reviews")