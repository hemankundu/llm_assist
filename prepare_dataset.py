import os
from datasets import load_dataset, DatasetDict, Dataset

from config import config_dict


def load_text_data(directory):
    # Function to load and preprocess text files

    texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                texts.append(file.read())
    return texts

def main():

    # Prepare dataset
    texts = load_text_data(config_dict['extract_text_from_pdf']['text_directory'])
    dataset = Dataset.from_dict({'text': texts})
    
    # Split the dataset into train, validation, and test sets
    train_test_split = dataset.train_test_split(test_size=0.2)
    test_val_split = train_test_split['test'].train_test_split(test_size=0.5)

    # Create a DatasetDict to hold the splits
    dataset_dict = DatasetDict({
        'train': train_test_split['train'],
        'validation': test_val_split['train'],
        'test': test_val_split['test']
    })

    # Save the dataset to disk
    dataset_dict.save_to_disk(config_dict['prepare_dataset']['dataset_save_path'])

    print('Dataset dict saved')


if __name__ == "__main__":

    main()