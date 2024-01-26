# DDoS-detect

This is a repository that demonstrates a proof of concept paper "Towards Resource-Efficient DDoS Detection in IoT: Leveraging Feature Engineering of System and Network Usage Metrics."

To build your own dataset, do the following:

1. Use Data/db_collector.py to build a dataset.

2. Manually label the dataset upon building it by adding a value to the .csv file called 'Attack-type' and name it 'labeled_db.csv'

3. Make sure the .csv file is in the /Data/ directory and run ml_train.py

4. Run classifier.py to classify current device state. In case of porting to another device, make sure to include the .joblib files in the same directory

If you wish to use this framework in your paper, please cite <>.