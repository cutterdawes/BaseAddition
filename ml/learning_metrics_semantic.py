import argparse
import pickle
import numpy as np
import addition_data
import torch
import addition_eval
from models import RNN, GRU, LSTM

def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True, help='Specified base')
    parser.add_argument('-u', '--unit', type=int, required=True, help='unit determining ordering of digits')
    parser.add_argument('-t', '--trials', type=int, required=False, help='number of training trials')
    parser.add_argument('-w', '--workers', type=int, required=False, help='number of CPU workers for data preparation')
    parser.add_argument('-d', '--directory', type=str, required=False, help='directory of pickled output')
    args = parser.parse_args()

    # get carry tables
    with open('../pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]
    
    # specify torch device (set to GPU if available), set number of CPU workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    workers = 0 if (args.workers is None) else args.workers

    # train model for each table
    all_learning_metrics = {}
    for dc, table in tables.items():

        # check if table corresponds to a unit
        if len(np.unique(table)) != 2:
            continue
            
        # initialize learning metrics
        num_passes = 2500
        avg_losses = np.zeros(int(num_passes / 10))
        avg_training_accs = np.zeros(int(num_passes / 10))
        avg_testing_accs = np.zeros(int(num_passes / 10))

        # evaluate model multiple times, average metrics
        trials = 10 if (args.trials is None) else args.trials
        for _ in range(trials):

            # initialize model and dataloaders
            model = LSTM(args.base, 1).to(device)
            training_dataloader, testing_dataloader = addition_data.prepare(
                b=args.base, depth=6, table=table, semanticity=True, unit=args.unit,
                batch_size=64, split_type='OOD', split_depth=3, sample=True, num_workers=workers
            )

            # evaluate model and store output
            losses, training_accs, testing_accs = addition_eval.eval(
                model, training_dataloader, testing_dataloader, device, num_passes=num_passes, lr=0.05, print_loss_and_acc=False
            )
            avg_losses += (np.array(losses) / trials)
            avg_training_accs += (np.array(training_accs) / trials)
            avg_testing_accs += (np.array(testing_accs) / trials)
        
        # add to learning metrics
        learning_metrics = {'loss': avg_losses, 'training_acc': avg_training_accs, 'testing_acc': avg_testing_accs}
        all_learning_metrics[dc] = learning_metrics

        # print progress
        print(f'completed trails for table:\n{table}\n')

    # pickle all learning metrics
    directory = '../pickles/learning_metrics' if (args.directory is None) else args.directory
    with open(f'{directory}/learning_metrics{args.base}_semantic{args.unit}.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()