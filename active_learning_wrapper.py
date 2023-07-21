from chemprop.train.make_predictions import make_predictions
import os
import shutil
import csv
import json
import pickle
from typing import List, Tuple, Set
from typing_extensions import Literal
from tap import Tap
from tqdm import tqdm
import numpy as np
from chemprop.args import TrainArgs, PredictArgs
from chemprop.data import get_task_names, get_data, MoleculeDataset, split_data, scaffold_split
from chemprop.train import cross_validate, run_training
from chemprop.utils import makedirs
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import math


class ActiveArgs(Tap):  # commands that is needed to run active learning
    active_save_dir: str  # save path
    train_config_path: str  # path to a json containing all the arguments
    # usually included in a training submission, except for path arguments
    train_config_path2: str = None  # path to a json containing all the arguments
    # usually included in a training submission to train the comparison model
    data_path: str  # dataset path
    features_path: List[str] = None
    active_test_path: str = None  # only use if separate from what's
    # in the data file. If a subset, instead use the indices pickle.
    active_test_features_path: List[str] = None
    active_test_indices_path: str = None  # path to pickle file containing a
    # list of indices for the test set out of the whole data path
    initial_trainval_indices_path: str = None  # path to pickle file containing
    # a list of indices for data in data path
    gpu: int = None  # which gpu to use
    no_comparison_model: bool = False  # if True, will not train a comparison model
    search_function: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
    ] = "random"
    """
    which function to use for choosing what molecules to add
    to trainval from the pool it will run a random data selection
    if search_function is not specified.
    note that for "ensemble" search_function 'ensemble_size <n>'
    is needed in config.json file
    """
    search_function2: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
    ] = "ensemble"
    """
    which function to use for comparison model.
    it will do anensemble model if search_function is not specifie.
    note that for ensemble search_function2 'ensemble_size <n>'
    is needed in config2.json file
    """
    test_fraction: float = 0.1  # This is the fraction of data used for test
    # if a separate test set is not provided.
    # initial_trainval_fraction: float = None
    initial_trainval_size: int = None # the number of data points in the initial trainval
    active_batch_size: int = None  # the number of data points added
    # to trainval in each cycle
    active_iterations_limit: int = None  # the max number of
    # training iterations to go through
    train_seed: int = 0 # seed for the initial training split
    test_seed: int = 0 # seed for the test split
    test_split_type: Literal["random", "scaffold","size"] = "random" # the type of test split
    evidential_regularization: float = 0.0  # the regularization parameter for evidential training
    initial_trainval_type: Literal[
        "random", 
        "related_random",
        "related_both",
        "related_high",
        "related_low",
        "related_min",
        "related_max",
        "related_mean",
        "morgan_fp",
    ] = "random"
    """
    how to choose the initial trainval set.
    "random" will choose the initial trainval set randomly
    "related_random" will choose a random number and choose random data close to that number from nontest dataset
    "related_both" will choose a random number and choose numbers before and after that random number from nontest dataset
    "related_high" will choose a random number and choose numbers after that random number from nontest dataset
    "related_low" will choose a random number and choose numbers before that random number from nontest dataset
    "related_min" will choose the numbers with lowest value from nontest dataset
    "related_max" will choose the numbers with highest value from nontest dataset
    "related_mean" will choose the numbers with average value from nontest dataset
    "morgan_fp" will choose a random molecule and choose molecules with highest morgan fingerprint similarities from nontest dataset
    """
    initial_trainval_seed: int = None # random number for choosing the initial trainval set
    initial_trainval_fraction: float = None # the fraction of data in the initial trainval set
    moprganfp_similarity: Literal["tanimoto", "dice","cosine", "sokal", "russel", "kulczynski", "mcconnaughey"] = "tanimoto" # the similarity metric for morgan fingerprint


def active_learning(active_args: ActiveArgs):
    train_args = get_initial_train_args(
        train_config_path=active_args.train_config_path,
        data_path=active_args.data_path,
        search_function=active_args.search_function,
        gpu=active_args.gpu,
        evidential_regularization=active_args.evidential_regularization,
    )
    if not active_args.no_comparison_model:
        train_args2 = get_initial_train_args(
            train_config_path=active_args.train_config_path2,
            data_path=active_args.data_path,
            search_function=active_args.search_function2,
            gpu=active_args.gpu,
            evidential_regularization=active_args.evidential_regularization,
        )
    active_args.split_type = train_args.split_type
    active_args.task_names = train_args.task_names
    active_args.smiles_columns = train_args.smiles_columns
    active_args.features_generator = train_args.features_generator
    active_args.feature_names = get_feature_names(active_args=active_args)

    makedirs(active_args.active_save_dir)
    whole_data, nontest_data, test_data = get_test_split(
        active_args=active_args,
        save_test_nontest=False,
        save_indices=True
    )
    trainval_data, remaining_data = initial_trainval_split(
        active_args=active_args,
        nontest_data=nontest_data,
        whole_data=whole_data,
        save_data=False,
        save_indices=True,
    )
    spearman, cv, rmses, rmses2, sharpness = [], [], [], [], []
    nll, miscalibration_area, ence, sharpness_root = [], [], [], []
    print(active_args.train_sizes)
    for i in range(len(active_args.train_sizes)):
        active_args.iter_save_dir = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
            f"selection{active_args.train_sizes[i]}",
        )
        if not active_args.no_comparison_model:
            active_args.iter_save_dir2 = os.path.join(
                active_args.active_save_dir,
                f"train{active_args.train_sizes[i]}",
                f"comparison{active_args.train_sizes[i]}",
            )
        active_args.run_save_dir = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
        )
        makedirs(active_args.iter_save_dir)
        if not active_args.no_comparison_model:
            makedirs(active_args.iter_save_dir2)
        """
        creates a folder by name of "train + number of data points"
        and inside that folder there are 2 folders
        one for selection model ("selection + number of data points")
        another one for comparison model ("comparison + number of data points")
        """
        if i != 0:
            trainval_data, remaining_data = update_trainval_split(
                new_trainval_size=active_args.train_sizes[i],
                active_args=active_args,
                previous_trainval_data=trainval_data,
                previous_remaining_data=remaining_data,
                save_new_indices=True,
                save_full_indices=True,
                iteration=i,
            )
        save_datainputs(
            active_args=active_args,
            trainval_data=trainval_data,
            remaining_data=remaining_data,
            test_data=test_data,
        )
        update_train_args(active_args=active_args, train_args=train_args)
        if not active_args.no_comparison_model:
            update_train_args2(active_args=active_args, train_args=train_args2)
        cross_validate(args=train_args, train_func=run_training)
        if not active_args.no_comparison_model:
            cross_validate(args=train_args2, train_func=run_training)
        run_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu)
        test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu)
        if not active_args.no_comparison_model:
            run_predictions2(active_args=active_args, train_args=train_args2,gpu=active_args.gpu)
        get_pred_results(
            active_args=active_args,
            whole_data=whole_data,
            iteration=i,
            save_error=True,
        )
        if not active_args.no_comparison_model:
            get_pred_results2(
                active_args=active_args,
                whole_data=whole_data,
                iteration=i,
            )
        save_results(
            active_args=active_args,
            test_data=test_data,
            nontest_data=nontest_data,
            whole_data=whole_data,
            iteration=i,
            save_whole_results=True,
            save_error=True,
        )
        rmses.append(get_rmse(active_args=active_args))
        if not active_args.no_comparison_model:
            rmses2.append(get_rmse2(active_args=active_args))
        else:
            rmses2.append(None)
        (
            spearman1,
            nll1,
            miscalibration_area1,
            ence1,
            sharpness1,
            shar_root1,
            cv1,
        ) = get_evaluation_scores(active_args=active_args)
        spearman.append(spearman1)
        nll.append(nll1)
        miscalibration_area.append(miscalibration_area1)
        ence.append(ence1)
        sharpness.append(sharpness1)
        sharpness_root.append(shar_root1)
        cv.append(cv1)
        save_evaluations(
            active_args,
            spearman,
            cv,
            rmses,
            rmses2,
            sharpness,
            nll,
            miscalibration_area,
            ence,
            sharpness_root,
        )
        cleanup_active_files(
            active_args=active_args,
            train_args=train_args,
            remove_models=True,
            remove_datainputs=False,
            remove_preds=False,
            remove_indices=False,
        )
        if not active_args.no_comparison_model:
            cleanup_active_files2(
                active_args=active_args,
                train_args2=train_args2,
                remove_models=True,
                remove_datainputs=False,
                remove_preds=False,
                remove_indices=False,
            )


# extract config settings from config json file
def get_initial_train_args(
    train_config_path: str,
    data_path: str,
    search_function,
    gpu,
    evidential_regularization
):
    with open(train_config_path) as f:
        config_dict = json.load(f)
    config_keys = config_dict.keys()
    if any(["path" in key for key in config_keys]):
        raise ValueError(
            "All path arguments should be determined by \
                          the active_learning wrapper and not supplied \
                         in the config file."
        )
    dataset_type = config_dict["dataset_type"]
    commandline_inputs = [
        "--data_path",
        data_path,
        "--config_path",
        train_config_path,
        "--dataset_type",
        dataset_type,
    ]
    if search_function == "mve":
        commandline_inputs.extend(["--loss_function", "mve"])
    elif search_function == "mve_ensemble":
        commandline_inputs.extend(["--loss_function", "mve"])
    elif (
        search_function == "evidential"
        or search_function == "evidential_total"
        or search_function == "evidential_aleatoric"
        or search_function == "evidential_epistemic"
    ):
        commandline_inputs.extend(["--loss_function", "evidential"])
    
    if gpu is not None:
        commandline_inputs.extend(["--gpu", str(gpu)])
    if evidential_regularization is not None:
        commandline_inputs.extend(["--evidential_regularization", str(evidential_regularization)])
    initial_train_args = TrainArgs().parse_args(commandline_inputs)



    initial_train_args.task_names = get_task_names(
        path=data_path,
        smiles_columns=initial_train_args.smiles_columns,
        target_columns=initial_train_args.target_columns,
        ignore_columns=initial_train_args.ignore_columns,
    )
    assert initial_train_args.num_tasks == 1

    return initial_train_args


def get_test_split(
    active_args: ActiveArgs,save_test_nontest: bool = True, save_indices: bool = True
) -> Tuple[MoleculeDataset]:
    data = get_data(
        path=active_args.data_path,
        features_path=active_args.features_path,
        smiles_columns=active_args.smiles_columns,
        target_columns=active_args.task_names,
        features_generator=active_args.features_generator,
    )

    if active_args.active_test_path is not None:
        assert (active_args.active_test_features_path is None) == (
            active_args.features_path is None
        )
        assert active_args.active_test_indices_path is None
        test_data = get_data(
            path=active_args.active_test_path,
            features_path=active_args.active_test_features_path,
            smiles_columns=active_args.smiles_columns,
            target_columns=active_args.task_names,
            features_generator=active_args.features_generator,
        )
        nontest_data = data
        whole_data = MoleculeDataset([d for d in nontest_data] + [d for d in test_data])
        for i, d in enumerate(tqdm(whole_data)):
            d.index = i
        if save_indices:
            nontest_indices = set(range(len(nontest_data)))
            test_indices = set(
                range(len(nontest_data), len(nontest_data) + len(test_data))
            )

    elif active_args.active_test_indices_path is not None:
        with open(active_args.active_test_indices_path, "rb") as f:
            test_indices = pickle.load(f)
            num_data = len(data)
            nontest_indices = {i for i in range(num_data) if i not in test_indices}
            test_data = MoleculeDataset([data[i] for i in test_indices])
            nontest_data = MoleculeDataset([data[i] for i in nontest_indices])
        whole_data = data
        for i, d in enumerate(tqdm(whole_data)):
            d.index = i

    elif active_args.test_split_type == "random":
        for i, d in enumerate(tqdm(data)):
            d.index = i
        sizes = (1 - active_args.test_fraction, active_args.test_fraction, 0)
        nontest_data, test_data, _ = split_data(
            data=data, split_type=active_args.split_type, sizes=sizes,seed=active_args.test_seed,
        )
        if save_indices:
            nontest_indices = {d.index for d in nontest_data}
            test_indices = {d.index for d in test_data}
        whole_data = data
    elif active_args.test_split_type == "scaffold":
        for i, d in enumerate(tqdm(data)):
            d.index = i
        sizes = (1 - active_args.test_fraction, active_args.test_fraction, 0)
        nontest_data, test_data, _ = scaffold_split(
            data=data, sizes=sizes,seed=active_args.test_seed,
        )
        if save_indices:
            nontest_indices = {d.index for d in nontest_data}
            test_indices = {d.index for d in test_data}
        whole_data = data
    elif active_args.test_split_type == "size":
        whole_data_smiles = MoleculeDataset.smiles(data) #list of smiles
        num_atoms=[]
        for [smiles] in whole_data_smiles:
            mol=Chem.MolFromSmiles(smiles)
            num_atoms.append(mol.GetNumAtoms())        
        sorted_smiles=sorted(zip(num_atoms, whole_data_smiles),reverse=True,)
        test_smiles=[item[1] for item in sorted_smiles[0:math.ceil(len(whole_data_smiles)*active_args.test_fraction)]]
        test_indices=[whole_data_smiles.index(value) for value in test_smiles] # get the indices of the test smiles
        for i, d in enumerate(tqdm(data)):
            d.index = i
        nontest_indices = {i for i in range(len(whole_data_smiles)) if i not in test_indices} # get the indices of the nontest smiles
        test_data = MoleculeDataset([data[i] for i in test_indices]) # get the test data
        nontest_data = MoleculeDataset([data[i] for i in nontest_indices]) # get the nontest data
        if save_indices:
            nontest_indices = {d.index for d in nontest_data}
            test_indices = {d.index for d in test_data}
        whole_data = data
        
        

    for d in whole_data:
        d.output = dict()
        for s, smiles in enumerate(active_args.smiles_columns):
            d.output[smiles] = d.smiles[s]
        for t, target in enumerate(active_args.task_names):
            d.output[target] = d.targets[t]
    save_dataset(
        data=whole_data,
        save_dir=active_args.active_save_dir,
        filename_base="whole",
        active_args=active_args,
    )
    if save_test_nontest:
        save_dataset(
            data=nontest_data,
            save_dir=active_args.active_save_dir,
            filename_base="nontest",
            active_args=active_args,
        )
        save_dataset(
            data=test_data,
            save_dir=active_args.active_save_dir,
            filename_base="test",
            active_args=active_args,
        )
    if save_indices:
        save_dataset_indices(
            indices=nontest_indices,
            save_dir=active_args.active_save_dir,
            filename_base="nontest",
        )
        save_dataset_indices(
            indices=test_indices,
            save_dir=active_args.active_save_dir,
            filename_base="test",
        )

    return whole_data, nontest_data, test_data


def save_dataset(
    data: MoleculeDataset, save_dir: str, filename_base: str, active_args: ActiveArgs
) -> None:
    save_smiles(
        data=data,
        save_dir=save_dir,
        filename_base=filename_base,
        active_args=active_args,
    )
    with open(
        os.path.join(save_dir, f"{filename_base}_full.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(active_args.smiles_columns + active_args.task_names)
        dataset_targets = data.targets()
        for i, smiles in enumerate(data.smiles()):
            writer.writerow(smiles + dataset_targets[i])
    if active_args.features_path is not None:
        with open(
            os.path.join(save_dir, f"{filename_base}_features.csv"), "w", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(active_args.feature_names)
            for d in data:
                writer.writerow(d.features)


def save_smiles(
    data: MoleculeDataset, save_dir: str, filename_base: str, active_args: ActiveArgs
) -> None:
    with open(
        os.path.join(save_dir, f"{filename_base}_smiles.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        if active_args.smiles_columns[0] == "":
            writer.writerow(["smiles"])
        else:
            writer.writerow(active_args.smiles_columns)
        writer.writerows(data.smiles())




def print_max_duplicates(lst):
    flat_list = [item for sublist in lst for item in sublist]
    frequency = {}
    max_frequency = 0

    # Count the frequency of each value
    for value in flat_list:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1

        # Update max_frequency if necessary
        if frequency[value] > max_frequency:
            max_frequency = frequency[value]

    return max_frequency



def initial_trainval_split(
    active_args: ActiveArgs,
    nontest_data: MoleculeDataset,
    whole_data: MoleculeDataset,
    save_data: bool = False,
    save_indices: bool = False,
) -> Tuple[MoleculeDataset]:
    num_data = len(whole_data)
    num_nontest = len(nontest_data)
    if active_args.initial_trainval_size is not None:
        active_args.initial_trainval_fraction = active_args.initial_trainval_size / num_data


    if active_args.active_batch_size is None:  # default: 10 steps
        active_args.active_batch_size = (num_nontest // 10) + 1
    if (
        active_args.initial_trainval_fraction is None
        and active_args.initial_trainval_indices_path is None
    ):
        active_args.initial_trainval_fraction = active_args.active_batch_size / num_data
    if active_args.initial_trainval_size is None:
        active_args.initial_trainval_size = int(active_args.initial_trainval_fraction * num_data)
    if active_args.initial_trainval_indices_path is not None:
        with open(active_args.initial_trainval_indices_path, "rb") as f:
            trainval_indices = pickle.load(f)
        active_args.initial_trainval_fraction = len(trainval_indices) / num_data
        trainval_data = MoleculeDataset([whole_data[i] for i in trainval_indices])
        remaining_data = MoleculeDataset(
            [d for d in nontest_data if d.index not in trainval_indices]
        )
        remaining_indices = {d.index for d in remaining_data}
        if save_indices:
            save_dataset_indices(
                indices=trainval_indices,
                save_dir=active_args.active_save_dir,
                filename_base="initial_trainval",
            )
            save_dataset_indices(
                indices=remaining_indices,
                save_dir=active_args.active_save_dir,
                filename_base="initial_remaining",
            )
        fraction_trainval = (
            active_args.initial_trainval_fraction * num_data / num_nontest
        )


    #  define related trainval set
    if active_args.initial_trainval_type == "morgan_fp":
        sorted_indices=morgan_fingerprint(nontest_data=nontest_data,active_args=active_args)
        sorted_trainval_indices=sorted_indices[0:active_args.initial_trainval_size]
        trainval_data = MoleculeDataset([nontest_data[i] for i in sorted_trainval_indices])
        remaining_data = MoleculeDataset([d for d in nontest_data if d.index not in trainval_data])
        save_dataset_indices(
            indices=trainval_data,
            save_dir=active_args.active_save_dir,
            filename_base="trainval",
        )  
    if active_args.initial_trainval_type == "related_max":
        target_nontest =MoleculeDataset.targets(nontest_data)
        smiles_nontest=MoleculeDataset.smiles(nontest_data) 
        sorted_target=sorted(zip(target_nontest, smiles_nontest),reverse=True,)
        test_smiles=[item[1] for item in sorted_target[0:active_args.initial_trainval_size]]
        sorted_trainval_indices=[smiles_nontest.index(test_smiles[i]) for i in range(len(test_smiles))]
        trainval_data = MoleculeDataset([nontest_data[i] for i in sorted_trainval_indices])
        remaining_data = MoleculeDataset([d for d in nontest_data if d.index not in trainval_data])
        save_dataset_indices(
            indices=trainval_data,
            save_dir=active_args.active_save_dir,
            filename_base="trainval",
        )  
    elif active_args.initial_trainval_type == "related_mean":
        target_nontest =MoleculeDataset.targets(nontest_data)
        smiles_nontest=MoleculeDataset.smiles(nontest_data) 
        sorted_target=sorted(zip(target_nontest, smiles_nontest),reverse=True,)
        test_smiles=[item[1] for item in sorted_target[(int(len(smiles_nontest)/2))-int(active_args.initial_trainval_size/2):(int(len(smiles_nontest)/2))+int(active_args.initial_trainval_size/2)]]
        sorted_trainval_indices=[smiles_nontest.index(test_smiles[i]) for i in range(len(test_smiles))]
        trainval_data = MoleculeDataset([nontest_data[i] for i in sorted_trainval_indices])
        remaining_data = MoleculeDataset([d for d in nontest_data if d.index not in trainval_data])
        save_dataset_indices(
            indices=trainval_data,
            save_dir=active_args.active_save_dir,
            filename_base="trainval",
        )
    elif active_args.initial_trainval_type == "related_min":
        target_nontest =MoleculeDataset.targets(nontest_data)
        smiles_nontest=MoleculeDataset.smiles(nontest_data) 
        sorted_target=sorted(zip(target_nontest, smiles_nontest))
        test_smiles=[item[1] for item in sorted_target[0:active_args.initial_trainval_size]]
        sorted_trainval_indices=[smiles_nontest.index(test_smiles[i]) for i in range(len(test_smiles))]
        trainval_data = MoleculeDataset([nontest_data[i] for i in sorted_trainval_indices])
        remaining_data = MoleculeDataset([d for d in nontest_data if d.index not in trainval_data])
        save_dataset_indices(
            indices=trainval_data,
            save_dir=active_args.active_save_dir,
            filename_base="trainval",
        )
    elif active_args.initial_trainval_type == "related_high":
            
        nontest_indices={d.index for d in nontest_data}  
        if active_args.initial_trainval_seed is  None:    
                rand=random.randint(0, len(nontest_indices)-active_args.initial_trainval_size)
        else:
                rand= active_args.initial_trainval_seed
        nontest_data_pickle_list=list(nontest_indices)
        random.shuffle(nontest_data_pickle_list)
        rand_nontest_data_list=nontest_data_pickle_list[rand:rand+active_args.initial_trainval_size]
        rand_nontest_data=set(rand_nontest_data_list)
        assert active_args.initial_trainval_size == len(rand_nontest_data), f"seed can be in this range:(0,{len(nontest_data)-active_args.initial_trainval_size})!"
            
        trainval_data = MoleculeDataset([whole_data[i] for i in rand_nontest_data])
        remaining_data = MoleculeDataset(
                [d for d in nontest_data if d.index not in trainval_data]
        )
        save_dataset_indices(
                indices=trainval_data,
                save_dir=active_args.active_save_dir,
                filename_base="trainval",
        )            
    elif active_args.initial_trainval_type == "related_low":
        nontest_indices={d.index for d in nontest_data}  
        if active_args.initial_trainval_seed is  None:    
                rand=random.randint(0+active_args.initial_trainval_size, len(nontest_indices))
        else:
                rand= active_args.initial_trainval_seed
        nontest_data_pickle_list=list(nontest_indices)
        random.shuffle(nontest_data_pickle_list)
        rand_nontest_data_list=nontest_data_pickle_list[rand-active_args.initial_trainval_size:rand]
        rand_nontest_data=set(rand_nontest_data_list)
        assert active_args.initial_trainval_size == len(rand_nontest_data), f"seed can be in this range:({active_args.initial_trainval_size},{len(nontest_data)})!"
        trainval_data = MoleculeDataset([whole_data[i] for i in rand_nontest_data])
        remaining_data = MoleculeDataset(
                [d for d in nontest_data if d.index not in trainval_data]
        )
        save_dataset_indices(
                indices=trainval_data,
                save_dir=active_args.active_save_dir,
                filename_base="trainval",
        )
    elif active_args.initial_trainval_type == "related_both":
        nontest_indices={d.index for d in nontest_data}  
        if active_args.initial_trainval_seed is  None:    
                rand=random.randint(0+active_args.initial_trainval_size/2, len(nontest_indices)-active_args.initial_trainval_size/2)
        else:
                rand= active_args.initial_trainval_seed
        nontest_data_pickle_list=list(nontest_indices)
        random.shuffle(nontest_data_pickle_list)
        rand_nontest_data_list=nontest_data_pickle_list[rand-int(active_args.initial_trainval_size/2):int(rand+active_args.initial_trainval_size/2)]
        rand_nontest_data=set(rand_nontest_data_list)
        assert active_args.initial_trainval_size == len(rand_nontest_data), f"seed can be in this range:({active_args.initial_trainval_size/2},{len(nontest_data)-int(active_args.initial_trainval_size/2)})!"
        trainval_data = MoleculeDataset([whole_data[i] for i in rand_nontest_data])
        remaining_data = MoleculeDataset(
                [d for d in nontest_data if d.index not in trainval_data]
        )
        save_dataset_indices(
                indices=trainval_data,
                save_dir=active_args.active_save_dir,
                filename_base="trainval",
        )
    elif active_args.initial_trainval_type == "related_random":
        nontest_indices={d.index for d in nontest_data}  
        if active_args.initial_trainval_seed is  None:    
                rand=random.randint(0+active_args.initial_trainval_size, len(nontest_indices)-active_args.initial_trainval_size)
        else:
                rand= active_args.initial_trainval_seed
        nontest_data_pickle_list=list(nontest_indices)
        random.shuffle(nontest_data_pickle_list)
        rand_nontest_data_list=nontest_data_pickle_list[rand-active_args.initial_trainval_size:rand+active_args.initial_trainval_size]
        assert active_args.initial_trainval_size == len(rand_nontest_data_list), f"seed can be in this range:({active_args.initial_trainval_size},{len(nontest_data)-active_args.initial_trainval_size})!"
        trainval_data_rand=random.sample(rand_nontest_data_list,active_args.initial_trainval_size)
        rand_nontest_data=set(trainval_data_rand)
        trainval_data = MoleculeDataset([whole_data[i] for i in rand_nontest_data])
        remaining_data = MoleculeDataset(
                [d for d in nontest_data if d.index not in trainval_data]
        )
        save_dataset_indices(
                indices=trainval_data,
                save_dir=active_args.active_save_dir,
                filename_base="trainval",
        )
    elif active_args.initial_trainval_type == "random":       
        fraction_trainval = (
            active_args.initial_trainval_fraction * num_data / num_nontest
        )
        sizes = (fraction_trainval, 1 - fraction_trainval, 0)
        trainval_data, remaining_data, _ = split_data(
            data=nontest_data, split_type=active_args.split_type, sizes=sizes,seed=active_args.train_seed,
        )
        if save_indices:
            trainval_indices = {d.index for d in trainval_data}
            remaining_indices = {d.index for d in remaining_data}
            save_dataset_indices(
                indices=trainval_indices,
                save_dir=active_args.active_save_dir,
                filename_base="initial_trainval",
            )
            save_dataset_indices(
                indices=remaining_indices,
                save_dir=active_args.active_save_dir,
                filename_base="initial_remaining",
            )

    active_args.train_sizes = list(
        range(len(trainval_data), num_nontest + 1, active_args.active_batch_size)
    )
    if active_args.train_sizes[-1] != num_nontest:
        active_args.train_sizes.append(num_nontest)
    if active_args.active_iterations_limit is not None:
        assert active_args.active_iterations_limit > 1
        if active_args.active_iterations_limit < len(active_args.train_sizes):
            active_args.train_sizes = active_args.train_sizes[
                : active_args.active_iterations_limit
            ]

    if save_data:
        save_dataset(
            data=trainval_data,
            save_dir=active_args.active_save_dir,
            filename_base="initial_trainval",
            active_args=active_args,
        )
        save_dataset(
            data=remaining_data,
            save_dir=active_args.active_save_dir,
            filename_base="initial_remaining",
            active_args=active_args,
        )

    return trainval_data, remaining_data


def get_feature_names(active_args: ActiveArgs) -> List[str]:
    if active_args.features_path is not None:
        features_header = []
        for feat_path in active_args.features_path:
            with open(feat_path, "r") as f:
                reader = csv.reader(f)
                feat_header = next(reader)
                features_header.extend(feat_header)
        return features_header
    else:
        return None


def get_indices(whole_data: MoleculeDataset, subset: MoleculeDataset) -> Set[int]:
    subset_hashes = set()
    subset_indices = set()
    for d in subset:
        subset_hashes.add(
            (tuple(d.smiles), tuple(d.targets), tuple(d.features))
        )  # smiles, targets, features
    for index, d in enumerate(tqdm(whole_data)):
        hash = (tuple(d.smiles), tuple(d.targets), tuple(d.features))
        if hash in subset_hashes:
            subset_indices.add(index)
    return subset_indices


def save_dataset_indices(indices: Set[int], save_dir: str, filename_base: str) -> None:
    with open(os.path.join(save_dir, f"{filename_base}_indices.pckl"), "wb") as f:
        pickle.dump(indices, f)


# train args that will use to train the selection model
def update_train_args(active_args: ActiveArgs, train_args: TrainArgs) -> None:
    train_args.save_dir = active_args.iter_save_dir
    train_args.data_path = os.path.join(active_args.run_save_dir, "trainval_full.csv")
    train_args.separate_test_path = os.path.join(
        active_args.run_save_dir, "test_full.csv"
    )
    if active_args.features_path is not None:
        train_args.features_path = [
            os.path.join(active_args.run_save_dir, "trainval_features.csv")
        ]
        train_args.separate_test_features_path = [
            os.path.join(active_args.run_save_dir, "test_features.csv")
        ]


# train args that will use to train the comparison model
def update_train_args2(active_args: ActiveArgs, train_args: TrainArgs) -> None:
    train_args.save_dir = active_args.iter_save_dir2
    train_args.data_path = os.path.join(active_args.run_save_dir, "trainval_full.csv")
    train_args.separate_test_path = os.path.join(
        active_args.run_save_dir, "test_full.csv"
    )
    if active_args.features_path is not None:
        train_args.features_path = [
            os.path.join(active_args.run_save_dir, "trainval_features.csv")
        ]
        train_args.separate_test_features_path = [
            os.path.join(active_args.run_save_dir2, "test_features.csv")
        ]


def save_datainputs(
    active_args: ActiveArgs,
    trainval_data: MoleculeDataset,
    remaining_data: MoleculeDataset,
    test_data: MoleculeDataset,
) -> None:
    save_dataset(
        data=trainval_data,
        save_dir=active_args.run_save_dir,
        filename_base="trainval",
        active_args=active_args,
    )
    save_dataset(
        data=remaining_data,
        save_dir=active_args.run_save_dir,
        filename_base="remaining",
        active_args=active_args,
    )
    save_dataset(
        data=test_data,
        save_dir=active_args.run_save_dir,
        filename_base="test",
        active_args=active_args,
    )

def test_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu) -> None:
    argument_input = [
        "--test_path",
        os.path.join(active_args.run_save_dir, "test_full.csv"),
        "--checkpoint_dir",
        active_args.iter_save_dir,
        "--preds_path",
        os.path.join(active_args.iter_save_dir, "test_preds.csv"),
        "--evaluation_scores_path",
        os.path.join(active_args.iter_save_dir, "evaluation_scores.csv"),

    ]
    if active_args.search_function != "random":
        argument_input.extend(
            [
                "--evaluation_methods",
                "nll",
                "miscalibration_area",
                "ence",
                "spearman",
                "sharpness",
                "sharpness_root",
                "cv",
            ]
        )
    if active_args.features_path is not None:
        argument_input.extend(
            [
                "--features_path",
                os.path.join(active_args.active_save_dir, "whole_features.csv"),
            ]
        )
    if gpu is not None:
        argument_input.extend(["--gpu", str(gpu)])
    # if isinstance(train_args.gpu, int):
    #     argument_input.extend(["--gpu", train_args.gpu])
    if active_args.search_function == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif active_args.search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif active_args.search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        active_args.search_function == "evidential_total"
        or active_args.search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif active_args.search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif active_args.search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif active_args.search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {active_args.search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)

# run predictions for selection model
def run_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu) -> None:
    argument_input = [
        "--test_path",
        os.path.join(active_args.active_save_dir, "whole_full.csv"),
        "--checkpoint_dir",
        active_args.iter_save_dir,
        "--preds_path",
        os.path.join(active_args.iter_save_dir, "whole_preds.csv"),
        "--evaluation_scores_path",
        os.path.join(active_args.iter_save_dir, "evaluation_scores2.csv"),

    ]
    if active_args.search_function != "random":
        argument_input.extend(
            [
                "--evaluation_methods",
                "nll",
                "miscalibration_area",
                "ence",
                "spearman",
                "sharpness",
                "sharpness_root",
                "cv",
            ]
        )
    if active_args.features_path is not None:
        argument_input.extend(
            [
                "--features_path",
                os.path.join(active_args.active_save_dir, "whole_features.csv"),
            ]
        )
    if gpu is not None:
        argument_input.extend(["--gpu", str(gpu)])
    # if isinstance(train_args.gpu, int):
    #     argument_input.extend(["--gpu", train_args.gpu])
    if active_args.search_function == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif active_args.search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif active_args.search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        active_args.search_function == "evidential_total"
        or active_args.search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif active_args.search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif active_args.search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif active_args.search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {active_args.search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)


# run predictions for comparison model
def run_predictions2(active_args: ActiveArgs, train_args: TrainArgs,gpu) -> None:
    argument_input = [
        "--test_path",
        os.path.join(active_args.active_save_dir, "whole_full.csv"),
        "--checkpoint_dir",
        active_args.iter_save_dir2,
        "--preds_path",
        os.path.join(active_args.iter_save_dir2, "whole_preds.csv"),
    ]
    if active_args.features_path is not None:
        argument_input.extend(
            [
                "--features_path",
                os.path.join(active_args.active_save_dir, "whole_features2.csv"),
            ]
        )
    if gpu is not None:
        argument_input.extend(["--gpu", str(gpu)])
    # if isinstance(train_args.gpu, int):
    #     argument_input.extend(["--gpu", train_args.gpu])
    if active_args.search_function2 == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
    pred_args2 = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args2)


# extract predicted results by selection model
def get_pred_results(
    active_args: ActiveArgs,
    whole_data: MoleculeDataset,
    iteration: int,
    save_error=False,
) -> None:
    with open(os.path.join(active_args.iter_save_dir, "whole_preds.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                whole_data[i].output[
                    j + f"_{active_args.train_sizes[iteration]}"
                ] = float(
                    line[j]
                )  # exp_#
                if active_args.search_function == "ensemble":
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(
                        line[j + "_ensemble_uncal_var"]
                    )  # exp_unc_#
                elif active_args.search_function == "mve":
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_mve_uncal_var"])
                elif active_args.search_function == "mve_ensemble":
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_mve_uncal_var"])
                elif (
                    active_args.search_function == "evidential_total"
                    or active_args.search_function == "evidential"
                ):
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_total_uncal_var"])
                elif active_args.search_function == "evidential_aleatoric":
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_aleatoric_uncal_var"])
                elif active_args.search_function == "evidential_epistemic":
                    whole_data[i].output[
                        j + f"_unc_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_epistemic_uncal_var"])
                if save_error:
                    whole_data[i].output[
                        j + f"_error_{active_args.train_sizes[iteration]}"
                    ] = abs(float(line[j]) - whole_data[i].output[j])


# extract predicted results by comparison model
def get_pred_results2(
    active_args: ActiveArgs,
    whole_data: MoleculeDataset,
    iteration: int,
    save_error=False,
) -> None:
    with open(os.path.join(active_args.iter_save_dir2, "whole_preds.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                whole_data[i].output[
                    j + f"_{active_args.train_sizes[iteration]}"
                ] = float(
                    line[j]
                )  # exp_#


def save_results(
    active_args: ActiveArgs,
    test_data: MoleculeDataset,
    nontest_data: MoleculeDataset,
    whole_data: MoleculeDataset,
    iteration: int,
    save_whole_results: bool = False,
    save_error=False,
) -> None:
    fieldnames = []
    fieldnames.extend(active_args.smiles_columns)
    fieldnames.extend(active_args.task_names)
    for i in range(iteration + 1):
        for j in active_args.task_names:
            fieldnames.append(j + f"_{active_args.train_sizes[i]}")
    if save_error:
        for i in range(iteration + 1):
            for j in active_args.task_names:
                fieldnames.append(j + f"_error_{active_args.train_sizes[i]}")
    if active_args.search_function != "random":
        for i in range(iteration + 1):
            for j in active_args.task_names:
                fieldnames.append(j + f"_unc_{active_args.train_sizes[i]}")
    with open(os.path.join(active_args.active_save_dir, "test_results.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for d in test_data:
            writer.writerow(d.output)
    with open(
        os.path.join(active_args.active_save_dir, "nontest_results.csv"), "w"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for d in nontest_data:
            writer.writerow(d.output)
    if save_whole_results:
        with open(
            os.path.join(active_args.active_save_dir, "whole_results.csv"), "w"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            for d in whole_data:
                writer.writerow(d.output)


def update_trainval_split(
    new_trainval_size: int,
    iteration: int,
    active_args: ActiveArgs,
    previous_trainval_data: MoleculeDataset,
    previous_remaining_data: MoleculeDataset,
    save_new_indices: bool = True,
    save_full_indices: bool = False,
) -> Tuple[MoleculeDataset]:
    num_additional = new_trainval_size - len(previous_trainval_data)
    if num_additional <= 0:
        raise ValueError(
            "Previous trainval size is larger than the next "
            + f"trainval size at iteration {iteration}"
        )
    if num_additional > len(previous_remaining_data):
        raise ValueError(
            "Increasing trainval size to "
            + f"{new_trainval_size} at iteration {iteration} "
            + "requires more data than is in the remaining pool, "
            + f"{len(previous_remaining_data)}"
        )
    if active_args.search_function != "random":  # only for a single task
        priority_values = [
            d.output[
                active_args.task_names[0]
                + f"_unc_{active_args.train_sizes[iteration-1]}"
            ]
            for d in previous_remaining_data
        ]
    elif active_args.search_function == "random":
        priority_values = [np.random.rand() for d in previous_remaining_data]
    sorted_remaining_data = [
        d
        for _, d in sorted(
            zip(priority_values, previous_remaining_data),
            reverse=True,
            key=lambda x: (x[0], np.random.rand()),
        )
    ]
    new_data = sorted_remaining_data[:num_additional]
    new_data_indices = {d.index for d in new_data}
    updated_trainval_data = MoleculeDataset(
        [d for d in previous_trainval_data] + new_data
    )
    updated_remaining_data = MoleculeDataset(
        [d for d in previous_remaining_data if d.index not in new_data_indices]
    )

    if save_new_indices:
        save_dataset_indices(
            indices=new_data_indices,
            save_dir=active_args.iter_save_dir,
            filename_base="new_trainval",
        )
    if save_full_indices:
        updated_trainval_indices = {d.index for d in updated_trainval_data}
        updated_remaining_indices = {d.index for d in updated_remaining_data}
        save_dataset_indices(
            indices=updated_trainval_indices,
            save_dir=active_args.iter_save_dir,
            filename_base="updated_trainval",
        )
        save_dataset_indices(
            indices=updated_remaining_indices,
            save_dir=active_args.iter_save_dir,
            filename_base="updated_remaining",
        )
    return updated_trainval_data, updated_remaining_data


# trainval and remaining, model files, preds
def cleanup_active_files(
    active_args: ActiveArgs,
    train_args: TrainArgs,
    remove_models: bool = True,
    remove_datainputs: bool = False,
    remove_preds: bool = False,
    remove_indices: bool = False,
) -> None:
    if remove_models:
        for i in range(train_args.num_folds):
            fold_dir = os.path.join(active_args.iter_save_dir, f"fold_{i}")
            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)
    if remove_datainputs:
        for dataset in ("trainval", "remaining", "test"):
            for file_suffix in ("_full.csv", "_smiles.csv", "_features.csv"):
                path = os.path.join(active_args.iter_save_dir, dataset + file_suffix)
                if os.path.exists(path):
                    os.remove(path)
    if remove_preds:
        for file in (
            "whole_preds.csv",
            "verbose.log",
            "quiet.log",
            "args.json",
            "test_scores.csv",
        ):
            path = os.path.join(active_args.iter_save_dir, file)
            if os.path.exists(path):
                os.remove(path)
    if remove_indices:
        for file in (
            "new_trainval_indices.pckl",
            "updated_remaining_indices.pckl",
            "updated_trainval_indices.pckl",
        ):
            path = os.path.join(active_args.iter_save_dir, file)
            if os.path.exists(path):
                os.remove(path)


# clean unnecessary files of comparison model
def cleanup_active_files2(
    active_args: ActiveArgs,
    train_args2: TrainArgs,
    remove_models: bool = True,
    remove_datainputs: bool = False,
    remove_preds: bool = False,
    remove_indices: bool = False,
) -> None:
    if remove_models:
        for i in range(train_args2.num_folds):
            fold_dir = os.path.join(active_args.iter_save_dir2, f"fold_{i}")
            if os.path.exists(fold_dir):
                shutil.rmtree(fold_dir)
    if remove_datainputs:
        for dataset in ("trainval", "remaining", "test"):
            for file_suffix in ("_full.csv", "_smiles.csv", "_features.csv"):
                path = os.path.join(active_args.iter_save_dir2, dataset + file_suffix)
                if os.path.exists(path):
                    os.remove(path)
    if remove_preds:
        for file in (
            "whole_preds.csv",
            "verbose.log",
            "quiet.log",
            "args.json",
            "test_scores.csv",
        ):
            path = os.path.join(active_args.iter_save_dir2, file)
            if os.path.exists(path):
                os.remove(path)
    if remove_indices:
        for file in (
            "new_trainval_indices.pckl",
            "updated_remaining_indices.pckl",
            "updated_trainval_indices.pckl",
        ):
            path = os.path.join(active_args.iter_save_dir2, file)
            if os.path.exists(path):
                os.remove(path)


# extract rmse of selection model
def get_rmse(active_args):
    with open(os.path.join(active_args.iter_save_dir, "test_scores.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                rmse = float(line["Mean rmse"])
    return rmse


# extract rmse of comparison model
def get_rmse2(active_args):
    with open(os.path.join(active_args.iter_save_dir2, "test_scores.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                rmse2 = float(line["Mean rmse"])
        return rmse2


# extract calculated evaluation scores by chemprop
def get_evaluation_scores(active_args):
    if active_args.search_function != "random":
        with open(
            os.path.join(active_args.iter_save_dir, "evaluation_scores.csv"), "r"
        ) as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            transposed_rows = list(zip(*rows))
        with open(
            os.path.join(active_args.iter_save_dir, "evaluation_scores.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(transposed_rows)

        with open(
            os.path.join(active_args.iter_save_dir, "evaluation_scores.csv"), "r"
        ) as f:
            reader = csv.DictReader(f)
            for i, line in enumerate(tqdm(reader)):
                for j in active_args.task_names:
                    spearmans = float(line["spearman"])
                    nlls = float(line["nll"])
                    miscalibration_areas = float(line["miscalibration_area"])
                    ences = float(line["ence"])
                    sharpness = float(line["sharpness"])
                    sharpness_root = float(line["sharpness_root"])
                    cv = float(line["cv"])
    else:
            spearmans, nlls, miscalibration_areas, ences, sharpness, sharpness_root, cv= 'nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan'
    return spearmans, nlls, miscalibration_areas, ences, sharpness, sharpness_root, cv


# save uncertainty evaluations in one file
def save_evaluations(
    active_args,
    spearmans,
    cv,
    rmses,
    rmses2,
    sharpness,
    nll,
    miscalibration_area,
    ence,
    sharpness_root,
):
    with open(
        os.path.join(active_args.active_save_dir, "uncertainty_evaluations.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        header = [
            "data_points",
            "spearman",
            "cv",
            "sharpness",
            "sharpness_root",
            "rmse",
            "rmse2",
            "nll",
            "miscalibration_area",
            "ence",
        ]
        writer.writerow(header)
        for i in range(len(cv)):
            new_row = [
                active_args.train_sizes[i],
                spearmans[i],
                cv[i],
                sharpness[i],
                sharpness_root[i],
                rmses[i],
                rmses2[i],
                nll[i],
                miscalibration_area[i],
                ence[i],
            ]
            writer.writerow(new_row)

def morgan_fingerprint(nontest_data:MoleculeDataset,active_args:ActiveArgs) -> Tuple[MoleculeDataset]:
    nontest_smiles = nontest_data.smiles()
    morgan_fps = []
    rand=random.randint(0, len(nontest_smiles))
    for [smiles] in nontest_smiles:
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,useChirality=True, radius=5, nBits = 1024)  # 5 is the radius of the fingerprint
        morgan_fps.append(fp)

    similarities = []
    for fp1 in morgan_fps:
        if active_args.moprganfp_similarity == "cosine":
            similarity_scores = [Chem.DataStructs.CosineSimilarity(fp1, fp2) for fp2 in morgan_fps] # measure similarity of each fingerptint with Tanimoto method 
        elif active_args.moprganfp_similarity == "dice":
            similarity_scores = [Chem.DataStructs.DiceSimilarity(fp1, fp2) for fp2 in morgan_fps]
        elif active_args.moprganfp_similarity == "russel":
            similarity_scores = [Chem.DataStructs.RusselSimilarity(fp1, fp2) for fp2 in morgan_fps]
        elif active_args.moprganfp_similarity == "kulczynski":
            similarity_scores = [Chem.DataStructs.KulczynskiSimilarity(fp1, fp2) for fp2 in morgan_fps]
        elif active_args.moprganfp_similarity == "tanimoto":
            similarity_scores = [Chem.DataStructs.TanimotoSimilarity(fp1, fp2) for fp2 in morgan_fps]
        elif active_args.moprganfp_similarity == "sokal":
            similarity_scores = [Chem.DataStructs.SokalSimilarity(fp1, fp2) for fp2 in morgan_fps]
        elif active_args.moprganfp_similarity == "mcconnaughey":
            similarity_scores = [Chem.DataStructs.McConnaugheySimilarity(fp1, fp2) for fp2 in morgan_fps]
        similarities.append(similarity_scores)
    similarities_array = np.array(similarities)
    scores=similarities_array[:,rand] # choose similarity scores of random smiles
    rounded_scores = [round(num, 9) for num in scores] # round to 9 decimal places
    sorted_scores=sorted(zip(rounded_scores, nontest_smiles))
    test_smiles=[item[1] for item in sorted_scores]
    sorted_scores_indices=[nontest_smiles.index(value) for value in test_smiles] # get indices of sorted similarity scores
    return sorted_scores_indices
    








if __name__ == "__main__":
    active_learning(ActiveArgs().parse_args())
