# Predicting Box Office Success using Weighted Sentiment Analysis of Early Reviews

**Overview:**

This project aims to develop a predictive model for box office revenue using sentiment analysis of early movie reviews.  The model incorporates a weighting system to account for the influence of different reviewers, aiming to improve prediction accuracy by prioritizing reviews from more reputable or influential sources. The analysis involves collecting pre-release and opening weekend reviews, performing sentiment analysis, applying the weighting scheme, and finally building a predictive model to forecast box office revenue.  This allows for the identification of potential high-return films before significant financial investment is committed.


**Technologies Used:**

* Python 3
* Pandas
* NumPy
* Scikit-learn
* NLTK (Natural Language Toolkit)
* Matplotlib
* Seaborn


**How to Run:**

1. **Clone the repository:**  `git clone <repository_url>`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the main script:** `python main.py`


**Example Output:**

The script will print key analysis results to the console, including metrics related to model performance (e.g., R-squared, RMSE).  Additionally, the project generates several visualization files (e.g., `sales_trend.png`, `sentiment_distribution.png`) which provide insights into the relationship between sentiment scores and box office revenue. These plots will be saved in the `output` directory.  The specific output may vary depending on the dataset used.


**Data:**

The project requires a dataset containing movie review data (text and reviewer information) and corresponding box office revenue figures.  This data is not included in this repository and needs to be provided separately.  The data should be preprocessed and formatted according to the specifications detailed in the `data_processing.ipynb` notebook (if applicable).

**Future Work:**

* Explore alternative sentiment analysis techniques and weighting schemes.
* Incorporate additional features (e.g., genre, cast, director) to improve prediction accuracy.
* Develop a user-friendly interface for data input and model execution.


**Contributing:**

Contributions are welcome! Please feel free to open issues or submit pull requests.  Please adhere to the coding style guidelines outlined in the `CONTRIBUTING.md` file (if applicable).