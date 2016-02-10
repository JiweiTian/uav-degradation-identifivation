import click
from kNNDTW import KnnDtw
from utils import load_labelled, load_test, get_distances, dtw_plots, label_dtws
from trainer import Trainer

@click.group()
def cli():
    pass

@cli.command('findbestk')
@click.option('--kmin', default=1, help='Minimum value for k Nearest Neighbours')
@click.option('--kmax', default=5, help='Maximum value for k Nearest Neighbours')
@click.option('-w', default=4000, help='Maximum Warping Window')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
def find_best_k(kmin, kmax, w, seed, train):
    click.echo('--- Find best k ---')
    
    kmin = int(min(kmin, kmax))
    kmax = int(max(kmin, kmax))

    ks = range(kmin, kmax + 1)

    train_data, train_label = load_labelled(train)

    click.echo('  - ks    : %s ' % str(ks))
    click.echo('  - w     : %d ' % w)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=train_label)
    trainer.find_best_k(ks, w)

    click.echo('\nDone.')

@cli.command('findbestw')
@click.option('--wmin', default=100, help='Minimum value for Warping Window')
@click.option('--wmax', default=4000, help='Maximum value for Warping Window')
@click.option('--step', default=100, help='Step for Warping Window')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('--seed', default=108, help='Seed for random shuffle')
@click.option('--train', help='Training Data as CSV file path')
def find_best_w(wmin, wmax, step, k, seed, train):
    click.echo('--- Find best w ---')
    
    wmin = int(min(wmin, wmax))
    wmax = int(max(wmin, wmax))

    ws = range(wmin, wmax, step)

    train_data, train_label = load_labelled(train)

    click.echo('  - ws    : %s ' % str(ws))
    click.echo('  - k     : %d ' % k)
    click.echo('  - seed  : %d ' % seed)
    click.echo('  - train : %s ' % train)

    click.echo('  - Training data size: %d' % len(train_data))
    click.echo('\nRunning...')

    trainer = Trainer(seed=seed, data=train_data, data_labels=train_label)
    trainer.find_best_w(k, ws)

    click.echo('\nDone.')

@cli.command('predict')
@click.option('-k', default=3, help='k Nearest Neighbours')
@click.option('-w', default=200, help='Maximum Warping Window')
@click.option('--train', help='Training Data as CSV file path')
@click.option('--test', help='Test Data as CSV file path')
def predict(k, w, train, test):
    click.echo('--- Predicting a label ---')
    #click.echo('Predicting with k=%d and w=%d.' % (k,w))

    train_data, train_label = load_labelled(train)
    test_data = load_test(test)

    click.echo('  - k     : %d ' % k)
    click.echo('  - w     : %d ' % w)
    click.echo('  - train : %s ' % train)
    click.echo('  - test  : %s ' % test)

    click.echo('\nRunning...')


    model = KnnDtw(k_neighbours = k, max_warping_window = w)
    model.fit(train_data, train_label)
    
    predicted_label, probability = model.predict(test_data)
    
    click.echo('\nPredicted label : %s ' % str(predicted_label))
    click.echo('\nDone.')

@cli.command('dtw')
@click.option('--data', help='Single timeseries data as CSV file path')
@click.option('--dataarray', help='List of timeseries data as CSV file path')
@click.option('-w', default=200, help='Maximum Warping Window')
def compute_dtw(data, dataarray, w):
    click.echo('--- Compute DTW ---')

    timeseries, timeseries_label = load_labelled(dataarray)
    timeserie_1 = load_test(data)

    click.echo('  - data        : %s ' % data)
    click.echo('  - dataarray   : %s ' % dataarray)
    click.echo('  - w           : %d ' % w)

    click.echo('\nRunning...')

    unsorted_dtws = get_distances(timeserie_1[0], data_array=timeseries, max_warping_window=w)

    # Save plots
    dtw_plots(unsorted_dtws)
    click.echo('Done. Plots have been saved.')

    click.echo('Choose a maximum number for labelling good and bad data based on DTW values.')
    click.echo('Check the plots to take better decsision.')
    click.echo('  Example: If value for Good is 150, all data with DTW 0-150 will be labelled "Good".')
    # Enter limit for 'Good'
    good_value = raw_input(' > Enter a value for "Good" (Ex: 150) : ')
    # Enter limit for 'Bad'
    bad_value = raw_input(' > Enter a value for "Bad" (Ex: 350) : ')
    # Print and save results to CSV
    fileName = raw_input(' > Enter a file name (add .csv at the end) : ')

    label_dtws(unsorted_dtws, int(good_value), int(bad_value), fileName)

    click.echo('\nDone.')

if __name__ == '__main__':
   cli()



