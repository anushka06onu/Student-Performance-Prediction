from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/student_performance_prediction'
os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(DRIVE_OUTPUT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(DRIVE_OUTPUT_DIR, 'figures'), exist_ok=True)
print('Drive mounted. Outputs will be saved under:', DRIVE_OUTPUT_DIR)