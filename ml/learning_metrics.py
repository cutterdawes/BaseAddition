import argparse
import pickle
import numpy as np
import addition_data
import torch
import addition_eval
from LSTM import LSTM


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='Compute the valid carry tables for specified base')
    parser.add_argument('-b', '--base', type=int, required=True, help='Specified base')
    parser.add_argument('-d', '--directory', type=str, required=False, help='directory of pickled output')
    args = parser.parse_args()

    # get carry tables
    with open('../pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]
    
    # specify torch device (set to GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train model for each table
    all_learning_metrics = {}
    for dc, table in tables.items():
            
        # initialize learning metrics
        num_passes = 2500
        avg_losses = np.zeros(int(num_passes / 10))
        avg_training_accs = np.zeros(int(num_passes / 10))
        avg_testing_accs = np.zeros(int(num_passes / 10))

        # evaluate model multiple times, average metrics
        rollouts = 5
        for _ in range(rollouts):

            # initialize model and dataloaders
            model = LSTM(args.base, 1).to(device)
            training_dataloader, testing_dataloader = addition_data.prepare(args.base, 6, table, split_type='OOD', split_depth=3, sample=True)

            # evaluate model and store output
            losses, training_accs, testing_accs = addition_eval.eval(model, training_dataloader, testing_dataloader, device, num_passes=num_passes, print_loss_and_acc=False)
            avg_losses += (np.array(losses) / rollouts)
            avg_training_accs += (np.array(training_accs) / rollouts)
            avg_testing_accs += (np.array(testing_accs) / rollouts)
        
        # add to learning metrics
        learning_metrics = {'loss': avg_losses, 'training_acc': avg_training_accs, 'testing_acc': avg_testing_accs}
        all_learning_metrics[dc] = learning_metrics
    
    # pickle all learning metrics
    directory = '/home/cdawes/Repo/pickles' if (args.directory is None) else args.directory
    with open(f'{directory}/learning_metrics{args.base}.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()