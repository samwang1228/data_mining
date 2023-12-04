import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the CSV file
file_path = ''
data = pd.read_csv(file_path)

 
data_scaled = data
# Plotting the data
plt.figure(figsize=(12, 6))
for col in data_scaled.columns:
    plt.plot(data_scaled[col], label=col)

plt.title('loss History')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

# Save the plot to a file
output_file_path = '2-loss_history_no_two.png'
plt.savefig(output_file_path)

# Close the plot to avoid displaying it in the output
plt.close()

# The file is now saved at the specified path
print(f"Plot saved to {output_file_path}")
