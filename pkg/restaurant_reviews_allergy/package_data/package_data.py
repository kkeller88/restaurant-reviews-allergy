import pkg_resources
import pandas as pd

sentiment_training_data_path = pkg_resources.resource_filename(
    __name__,
    'sentiment_training_data.csv'
    )
sentiment_training_data = pd.read_csv(sentiment_training_data_path)
