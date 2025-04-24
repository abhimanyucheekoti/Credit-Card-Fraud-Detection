# Credit-Card-Fraud-Detection
# OUTPUT
Original data shape: (555719, 23)
Missing values per column:
 Unnamed: 0               0
trans_date_trans_time    0
cc_num                   0
merchant                 0
category                 0
amt                      0
first                    0
last                     0
gender                   0
street                   0
city                     0
state                    0
zip                      0
lat                      0
long                     0
city_pop                 0
job                      0
dob                      0
trans_num                0
unix_time                0
merch_lat                0
merch_long               0
is_fraud                 0
dtype: int64

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00    110715
           1       0.97      0.57      0.72       429

    accuracy                           1.00    111144
   macro avg       0.99      0.79      0.86    111144
weighted avg       1.00      1.00      1.00    111144

Confusion Matrix:
 [[110708      7]
 [   184    245]]






