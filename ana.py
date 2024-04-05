import pandas as pd
import matplotlib.pyplot as plt
file_path_new = './loss_cbam.csv'
data_new = pd.read_csv(file_path_new)

# Extract the iteration steps and loss values again
iterations_new = list(range(len(data_new.columns) - 1))
cbam_loss_new = data_new.loc[data_new['Unnamed: 0'] == 'cbam'].iloc[0, 1:].values.astype(float)
base_loss_new = data_new.loc[data_new['Unnamed: 0'] == 'base'].iloc[0, 1:].values.astype(float)

# Calculations for the segment 2000 to 5000
start_iter_new, end_iter_new = 200, 1500
cbam_loss_segment_new = cbam_loss_new[start_iter_new:end_iter_new]
base_loss_segment_new = base_loss_new[start_iter_new:end_iter_new]

# Mean and Standard Deviation for the new segment
cbam_mean_new = cbam_loss_segment_new.mean()
cbam_std_new = cbam_loss_segment_new.std()
base_mean_new = base_loss_segment_new.mean()
base_std_new = base_loss_segment_new.std()

print(cbam_mean_new, cbam_std_new, base_mean_new, base_std_new)

# 绘制对应区间的图像

plt.plot(iterations_new[start_iter_new:end_iter_new], cbam_loss_segment_new, label='CBAM')
plt.plot(iterations_new[start_iter_new:end_iter_new], base_loss_segment_new, label='Base')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration for CBAM and Base Model')
plt.legend()
plt.show()
