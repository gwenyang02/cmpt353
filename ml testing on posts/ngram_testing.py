import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Can n-grams in text effectively predict whether a post leans Democratic or Republican?
# Question: Can posts with similar n-grams be grouped into clusters representing different policy topics or issues?
# Model: Use k-means,
def prepare_data(texts, labels):
        """
        Prepare data by vectorizing texts

        Args:
            texts (list): Input texts
            labels (list): Corresponding labels

        Returns:
            tuple: Vectorized features and labels
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts)

        # Convert to dense array for further processing
        X_dense = X.toarray()

        return X_dense, labels

