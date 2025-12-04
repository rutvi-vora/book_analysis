# Import required libraries
import glob
# import streamlit as st  # Uncomment to use Streamlit for visualization
# import plotly.express as px  # Uncomment to use Plotly for visualization

# Import sentiment analyzer from NLTK
from nltk.sentiment import SentimentIntensityAnalyzer

# Get all diary text file paths
filepaths = sorted(glob.glob("diary/*.txt"))
# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Lists to store negativity and positivity scores
negativity = []
positivity = []

# Loop through each diary file and analyze sentiment
for filepath in filepaths:
    with open(filepath) as file:
        content = file.read()
    # Get sentiment scores for the diary entry
    scores = analyzer.polarity_scores(content)
    positivity.append(scores["pos"])
    negativity.append(scores["neg"])


# Extract dates from filenames for plotting
# Removes '.txt' and 'diary/' from each filepath
dates = [filepath.strip(".txt").strip("diary/") for filepath in filepaths]

# Visualization code (commented out)
# Uncomment below to display results using Streamlit and Plotly
# st.title("Diary Tone")
# st.subheader("Positivity")
# pos_figure = px.line(x=dates, y=positivity,
#                         labels={"x": "Date", "y": "Positivity"})
#
# st.subheader("Negativity")
# neg_figure = px.line(x=dates, y=negativity,
#                         labels={"x": "Date", "y": "Negativity"})
# st.plotly_chart(pos_figure)
# st.plotly_chart(neg_figure)

import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
ax1.plot(dates, positivity, label='Positivity')
ax2.plot(dates, negativity, label='Negativity')
ax1.set_xlabel('Date')
ax1.set_ylabel('Score')
ax2.set_xlabel('Date')
ax2.set_ylabel('Score')
ax1.set_title('Diary Tone Over Time')
ax1.legend()