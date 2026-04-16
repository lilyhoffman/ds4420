# Hot Hand or Illusion? A Data-Driven NBA Analysis

This project looks at whether the “hot hand” in basketball is real or just randomness. The hot hand is the idea that a player is more likely to make a shot after making a few in a row. While this belief is common, past research has shown mixed results, so we wanted to test it using modern data and models.

We use NBA play-by-play data from 1997 to 2023, focusing on shot-by-shot sequences for a single player. Each shot is treated as a binary outcome (made or missed), and we keep the order of shots within each game to study how performance changes over time.

## Methods

We built two main models:

- **Handwritten Logistic Autoregressive Model**  
  This model predicts the probability of making a shot based on recent performance, including:
  - Previous shot outcomes (lags)
  - Rolling shooting percentage
  - Make/miss streaks  

  The model is implemented from scratch using gradient descent to better understand how it learns.

- **Bayesian Logistic Regression Model**  
  This model also predicts shot probability but returns a distribution of possible values instead of a single estimate. This helps capture uncertainty and gives a more complete picture of the results.

## Results

Across both models, recent performance has only a small effect on future shots. While there is a slight increase in probability after a streak, the effect is weak and not consistent.

Overall, the results suggest that shot outcomes are mostly independent, and the hot hand effect is much smaller than people expect.

## Streamlit App

https://ds4420-hot-hands.streamlit.app/
