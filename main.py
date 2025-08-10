import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
num_movies = 100
data = {
    'MovieTitle': [f'Movie {i}' for i in range(1, num_movies + 1)],
    'EarlyReviewSentiment': np.random.normal(loc=0, scale=1, size=num_movies), #sentiment score (-1 to 1)
    'ReviewerInfluence': np.random.uniform(0.1, 1, size=num_movies), #influence score (0.1 to 1)
    'WeightedSentiment': np.random.normal(loc=0, scale=1, size=num_movies), #weighted sentiment
    'BoxOfficeRevenue': np.random.randint(1000000, 100000000, size=num_movies) #revenue in USD
}
df = pd.DataFrame(data)
df['WeightedSentiment'] = df['EarlyReviewSentiment'] * df['ReviewerInfluence']
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
#In a real-world scenario, this section would involve handling missing data, outliers, etc.
# --- 3. Analysis ---
# Calculate the correlation between weighted sentiment and box office revenue
correlation, p_value = pearsonr(df['WeightedSentiment'], df['BoxOfficeRevenue'])
print(f"Correlation between Weighted Sentiment and Box Office Revenue: {correlation:.2f}")
print(f"P-value: {p_value:.3f}")
# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
sns.regplot(x='WeightedSentiment', y='BoxOfficeRevenue', data=df, scatter_kws={'alpha':0.5})
plt.title('Weighted Sentiment vs. Box Office Revenue')
plt.xlabel('Weighted Sentiment Score')
plt.ylabel('Box Office Revenue (USD)')
plt.grid(True)
plt.tight_layout()
# Save the plot to a file
output_filename = 'weighted_sentiment_vs_revenue.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
#Further analysis could involve building a predictive model (linear regression, etc.) using the weighted sentiment as a predictor for box office revenue.  This example focuses on the core data analysis and visualization aspects.