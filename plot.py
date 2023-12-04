import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
#file_path = '/Users/wangshaocheng/Desktop/python/average_categorical_accuracy_combined.csv'
file_path = '/Users/wangshaocheng/Desktop/python/average_categorical_accuracy_combined.csv'
data = pd.read_csv(file_path)

# Scaling values to fit between 0 and 5
scaler = MinMaxScaler(feature_range=(0, 78))
data_scaled = scaler.fit_transform(data)

# Convert scaled data back to DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
# data_scaled = data
# Plotting the data
plt.figure(figsize=(12, 6))
for col in data_scaled.columns:
    plt.plot(data_scaled[col], label=col)

plt.title('Avg Acc History')
plt.xlabel('Epochs')
plt.ylabel('avg acc')
plt.legend()

# Save the plot to a file
output_file_path = 'fouracc_history.png'
plt.savefig(output_file_path)

# Close the plot to avoid displaying it in the output
plt.close()

# The file is now saved at the specified path
print(f"Plot saved to {output_file_path}")
