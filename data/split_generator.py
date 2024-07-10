import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os.path
import pickle


def split_data(dataset, mode = 'cross_val', k=5):
    dataPath = dataset.root
    splitPath = os.path.join(dataPath, 'splits')

    if not os.path.exists(dataPath):
        raise Exception("The path to the dataset is incorrect")

    if not os.path.exists(splitPath):
        os.makedirs(splitPath)
    
    if mode == 'cross_val':
        if os.path.isfile(os.path.join(splitPath, f'cross_validation_val_{k}_splits.pkl')):
            with open(os.path.join(splitPath, f'cross_validation_val_{k}_splits.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            indices = np.arange(len(dataset))
            targets = dataset.y.long().numpy()

            splitter_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=5345)
            in_split = StratifiedKFold(n_splits=k, shuffle=True, random_state=534)

            splits_t_t = splitter_train_test.split(indices, targets)
            train_idx, test_idx =  next(splits_t_t)

            splits = []
            for train_folds_idx, val_fold_idx in in_split.split(train_idx, targets[train_idx]):
                splits.append({"train": np.array(train_idx[train_folds_idx], dtype=np.int64), "val": np.array(train_idx[val_fold_idx], dtype=np.int64), "test": np.array(test_idx, dtype=np.int64)})

            with open(os.path.join(splitPath, f'cross_validation_val_{k}_splits.pkl'), 'wb') as f:
                pickle.dump(splits, f)
    else:
        raise Exception('Wrong split mode')
    
    return splits