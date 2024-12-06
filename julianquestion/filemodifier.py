import pandas as pd
import numpy as np

#reading in the data
shifteddata = pd.read_csv('./shifteddiffbig.csv')
data = pd.read_csv('allcommentsbigwithsentiment.csv')

#randomly filter not shifted comments approx 50%

# Identify rows where shifted == sentiment
not_shifted = data[data['shifted'] == data['sentiment']]

# Randomly sample 50% of these rows
sampled_not_shifted = not_shifted.sample(frac=0.5, random_state=42) 

# Keep all rows where shifted != sentiment
shifted = data[data['shifted'] != data['sentiment']]

# Combine the sampled not-shifted rows with the already shifted rows
filtered_data = pd.concat([sampled_not_shifted, shifted])

# Calculate the initial percentage of rows where shifted equals sentiment
percentage_shifted_equal = (data['shifted'] == data['sentiment']).mean() * 100
print(f"Percentage of rows where shifted equals sentiment before filtering: {percentage_shifted_equal:.2f}%")

# Separate rows where shifted equals sentiment
equal_shifted = data[data['shifted'] == data['sentiment']]

# Separate rows where shifted does not equal sentiment
not_equal_shifted = data[data['shifted'] != data['sentiment']]

# Calculate the initial percentage of rows where shifted equals sentiment
percentage_shifted_equal = (data['shifted'] == data['sentiment']).mean() * 100
print(f"Percentage of rows where shifted equals sentiment before filtering: {percentage_shifted_equal:.2f}%")

# Separate rows where shifted equals sentiment
equal_shifted = data[data['shifted'] == data['sentiment']]

# Separate rows where shifted does not equal sentiment
not_equal_shifted = data[data['shifted'] != data['sentiment']]

# Calculate the number of rows needed to make 30% shifted == sentiment
num_desired_equal_shifted = int(0.3 * len(data))

# Downsample rows where shifted equals sentiment to match the desired number
downsampled_equal_shifted = equal_shifted.sample(n=num_desired_equal_shifted, random_state=42)

# Combine the downsampled rows with all rows where shifted != sentiment
balanced_data = pd.concat([downsampled_equal_shifted, not_equal_shifted])

# Recalculate the percentage after filtering
new_percentage_shifted_equal = (balanced_data['shifted'] == balanced_data['sentiment']).mean() * 100
print(f"Percentage of rows where shifted equals sentiment after filtering: {new_percentage_shifted_equal:.2f}%")

# Optionally shuffle the combined data
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced data to a new CSV
balanced_data.to_csv('balanced_comments_70_not_shifted.csv', index=False)

