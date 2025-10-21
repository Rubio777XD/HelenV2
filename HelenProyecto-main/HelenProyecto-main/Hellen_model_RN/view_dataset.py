import pickle


def view_dataset(dataset_file):
    with open(dataset_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = data_dict['data']
    labels = data_dict['labels']

    print("Dataset Overview:")
    print(f"Number of samples: {len(data)}")
    print(f"Labels: {set(labels)}")
    print("Example data point:")
    print(data[0])
    print(f"Label: {labels[0]}")


if __name__ == '__main__':
    dataset_path = 'data3.pickle'
    view_dataset(dataset_path)
