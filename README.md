# UAV Degradation Identification (Master Thesis)

This command line tool allows to run k Nearest Neighbour with DTW to train, evaluate and predict the label of flight data. The `data/` folder contains all the data ready to use for playing with this tool.

It is implemented in Python and uses [Click][click]

## How to use

Type this command to see all available commands:

    python droneML.py --help


If there is an issue with importing matplotlib related to the locale, then export the following environment variables:

    export LC_ALL=en_US.UTF-8
    export LANG=en_UTF-8


## Example commands

Create a model with `k = 3` and `w = 200`, train the model and predict the input:

    python droneML.py predict -k 3 -w 200 --train 'data/uav_data_z_train.csv' --test 'data/uav_data_z_test.csv'

Find the best `k` between the values 1,2 and 3 and for `w = 500`:

    python droneML.py findbestk --kmin 1 --kmax 3 -w 500 --train 'data/uav_data_full_z.csv'

Find the best warping window size `w` between the values 50, 100, 150, ..., until 500 and for `k = 1`:

    python droneML.py findbestw --wmin 50 --wmax 500 --step 50 -k 1 --train 'data/uav_data_full_z.csv'


Compute the DTW between one time series (`--data`) and one or more time series (`--datarray`).

    python droneML.py dtw --data 'data/training_x_example.csv' --dataarray 'data/test_x_example.csv' -w 5

After compute all DTW, it will create and save plots in the same folder as the script and then ask you to input the limits for "Good" and "Bad" values. The output is a CSV file with the labels.



[click]: http://click.pocoo.org/5/ "Click Library"
