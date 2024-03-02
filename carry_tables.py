import sys
import pickle
import fn

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print('Usage: python carry_tables.py <b>')
        sys.exit(1)
    
    # Get the command-line argument for b
    b = int(sys.argv[1])
    
    # Execute the function f(b) from the functions module
    tables = fn.construct_tables(b)
    
    # Pickle the dictionary and save it to output.pickle
    directory = '/scratch/network/cdawes/Repo/carry_tables'
    with open(f'{directory}/tables{b}.pickle', 'wb') as file:
        pickle.dump(tables, file)
    
    print(f'Function executed successfully.\nOutput saved to {directory}/tables{b}.pickle')

if __name__ == '__main__':
    main()