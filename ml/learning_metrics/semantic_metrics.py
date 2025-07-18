import argparse
import pickle
import numpy as np
import torch

from ml import dataset
from ml import training
from ml.model import RecurrentModel


def main():
    # create and parse arguments
    parser = argparse.ArgumentParser(description='train recurrent model to add with each carry function, under specified semanticity')
    parser.add_argument('-b', '--base', type=int, required=True,
                        help='Specified base')
    parser.add_argument('-u', '--unit', type=int, required=True,
                        help='unit determining ordering of digits')
    parser.add_argument('-m', '--model', type=str, required=False, default='GRU',
                        help='model type (RNN, GRU, or LSTM; default: GRU)')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=2500,
                        help='number of training epochs (default: 2500)')
    parser.add_argument('-t', '--trials', type=int, required=False, default=10,
                        help='number of training trials (default: 10)')
    parser.add_argument('-w', '--workers', type=int, required=False, default=0,
                        help='number of CPU workers for data preparation (default: 0)')
    parser.add_argument('-d', '--directory', type=str, required=False, default='pickles/learning_metrics',
                        help='directory of pickled output')
    args = parser.parse_args()

    # get carry tables
    with open('pickles/carry_tables/all_tables_d1_b2-6.pickle', 'rb') as f:
        all_tables = pickle.load(f)
    tables = all_tables[args.base]
    
    # specify torch device (set to GPU if available), set number of CPU workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize learning rate, hidden dim
    lrs = {'RNN': 0.005, 'GRU': 0.05, 'LSTM': 0.05}
    lr = lrs[args.model]
    hidden_dim = 5*args.base if args.model == 'RNN' else args.base

    # train model for each table
    all_learning_metrics = {}
    for dc, table in tables.items():

        # check if table corresponds to a unit
        if len(np.unique(table)) != 2:
            continue
            
        # initialize learning metrics
        avg_losses = np.zeros(int(args.epochs / 10))
        avg_training_accs = np.zeros(int(args.epochs / 10))
        avg_testing_accs = np.zeros(int(args.epochs / 10))

        # evaluate model multiple times, average metrics
        for _ in range(args.trials):

            # initialize model and dataloaders
            model = RecurrentModel(args.base, hidden_dim, args.model).to(device)
            training_dataloader, testing_dataloader = dataset.prepare(
                b=args.base, depth=6, table=table, semanticity=True, unit=args.unit,
                batch_size=32, split_type='OOD', split_depth=3, sample=True, num_workers=args.workers
            )

            # evaluate model and store output
            losses, training_accs, testing_accs = training.eval(
                model, training_dataloader, testing_dataloader, device, epochs=args.epochs, lr=lr, print_loss_and_acc=False
            )
            avg_losses += (np.array(losses) / args.trials)
            avg_training_accs += (np.array(training_accs) / args.trials)
            avg_testing_accs += (np.array(testing_accs) / args.trials)
        
        # add to learning metrics
        learning_metrics = {'loss': avg_losses, 'training_acc': avg_training_accs, 'testing_acc': avg_testing_accs}
        all_learning_metrics[dc] = learning_metrics

        # print progress
        print(f'completed trials for table:\n{table}\n')

    # pickle all learning metrics
    with open(f'{args.directory}/learning_metrics{args.base}_semantic{args.unit}_{args.model}.pickle', 'wb') as f:
        pickle.dump(all_learning_metrics, f)


if __name__ == '__main__':
    main()