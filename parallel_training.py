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
from chemprop.args import TrainArgs, PredictArgs,FingerprintArgs
from chemprop.train.molecule_fingerprint import molecule_fingerprint
from chemprop.data import get_task_names, get_data, MoleculeDataset, split_data, scaffold_split
from chemprop.train import cross_validate, run_training
from chemprop.utils import makedirs
import random
from rdkit import Chem
from rdkit.Chem import AllChem
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


class ActiveArgs(Tap):
    features_path: List[str] = None
    active_save_dir: str 
    train_config_path: str
    train_config_path2: str = None
    train_config_path3: str = None
    train_config_path4: str = None
    train_config_path_comparison: str = None
    data_path: str 
    gpu: int = None
    test_fraction: float = 0.1 
    initial_trainval_size: int = None
    active_batch_size: int = None  
    active_iterations_limit: int = None 
    train_seed: int = 0 
    test_seed: int = 0 
    evidential_regularization: float = 0.0
    initial_trainval_seed: int = None 
    initial_trainval_fraction: float = None 
    search_function: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "dropout"
    ] = "random"
    search_function2: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "dropout"
    ] = "random"
    search_function3: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "dropout"
    ] = "random"
    search_function4: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "dropout"
    ] = "random"
    no_comparison_model: bool = False 
    search_function_comparison: Literal[
        "ensemble",
        "random",
        "mve",
        "mve_ensemble",
        "evidential",
        "evidential_epistemic",
        "evidential_aleatoric",
        "evidential_total",
        "dropout"
    ] = "ensemble"
    selection_method: Literal["rmse","spearman","hybrid","nll","miscal","mix","multi"] = "rmse"
    hybrid_method: Literal["rmse","spearman","nll","miscal"] = "rmse"
    hybrid_count: int = 2

def parallel_training(active_args: ActiveArgs):
    train_args = get_initial_train_args(
        train_config_path=active_args.train_config_path,
        data_path=active_args.data_path,
        search_function=active_args.search_function,
        gpu=active_args.gpu,
        evidential_regularization=active_args.evidential_regularization,
    )
    train_args2 = get_initial_train_args(
        train_config_path=active_args.train_config_path2,
        data_path=active_args.data_path,
        search_function=active_args.search_function2,
        gpu=active_args.gpu,
        evidential_regularization=active_args.evidential_regularization,
    )
    train_args3 = get_initial_train_args(
        train_config_path=active_args.train_config_path3,
        data_path=active_args.data_path,
        search_function=active_args.search_function3,
        gpu=active_args.gpu,
        evidential_regularization=active_args.evidential_regularization,
    )
    train_args4 = get_initial_train_args(
        train_config_path=active_args.train_config_path4,
        data_path=active_args.data_path,
        search_function=active_args.search_function4,
        gpu=active_args.gpu,
        evidential_regularization=active_args.evidential_regularization,
    )
    if not active_args.no_comparison_model:
        train_args_comparison1 = get_initial_train_args(
            train_config_path=active_args.train_config_path_comparison,
            data_path=active_args.data_path,
            search_function=active_args.search_function_comparison,
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
    rmses1, rmses2, rmses3, rmses4 = [], [], [], []
    rmses21, rmses22, rmses23, rmses24 = [], [], [], []
    sp1, sp2, sp3, sp4 = [], [], [], []
    nl1, nl2, nl3, nl4 = [], [], [], []
    mis1, mis2, mis3, mis4 = [], [], [], []
    best_function=[]
    for i in range(len(active_args.train_sizes)):
        if active_args.selection_method == "hybrid" and i ==active_args.hybrid_count:
            active_args.selection_method = active_args.hybrid_method
            
        active_args.model_save_dir1 = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
            f"{active_args.search_function}_{active_args.train_sizes[i]}",
        )
        active_args.model_save_dir2 = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
            f"{active_args.search_function2}_{active_args.train_sizes[i]}",
        )
        active_args.model_save_dir3 = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
            f"{active_args.search_function3}_{active_args.train_sizes[i]}",
        )
        active_args.model_save_dir4 = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
            f"{active_args.search_function4}_{active_args.train_sizes[i]}",
        )
        active_args.run_save_dir1 = os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i]}",
        )
        active_args.iter_save_dir1 = os.path.join(
                active_args.model_save_dir1,
                f"selection",)
        active_args.iter_save_dir2 = os.path.join(
                active_args.model_save_dir2,
                f"selection",)
        active_args.iter_save_dir3 = os.path.join(
                active_args.model_save_dir3,
                f"selection",)
        active_args.iter_save_dir4 = os.path.join(
                active_args.model_save_dir4,
                f"selection",)
        if not active_args.no_comparison_model:
            active_args.iter_save_dir1_comp = os.path.join(
                active_args.model_save_dir1,
                f"comparison",)
        makedirs(active_args.model_save_dir1)
        makedirs(active_args.model_save_dir2)
        makedirs(active_args.model_save_dir3)
        makedirs(active_args.model_save_dir4)
        makedirs(active_args.iter_save_dir1)
        makedirs(active_args.iter_save_dir2)
        makedirs(active_args.iter_save_dir3)
        makedirs(active_args.iter_save_dir4)
        if not active_args.no_comparison_model:
            makedirs(active_args.iter_save_dir1_comp)
        if i != 0:
            trainval_data, remaining_data = update_trainval_split(
                new_trainval_size=active_args.train_sizes[i],
                active_args=active_args,
                previous_trainval_data=trainval_data,
                previous_remaining_data=remaining_data,
                save_new_indices=True,
                save_full_indices=True,
                iteration=i,
                best_function=best_function,
                search_function=best_function[i-1]
                
            )
        validation_set, train_set,_  = split_data(
            data=trainval_data,
            sizes=(0.2, 0.8, 0),
        )
        save_datainputs(
            active_args=active_args,
            trainval_data=trainval_data,
            remaining_data=remaining_data,
            test_data=test_data,
            save_dir=active_args.run_save_dir1,
        )
        save_dataset(
            data=validation_set,
            save_dir=active_args.run_save_dir1,
            filename_base="validation_set",
            active_args=active_args,
        )
        save_dataset(
            data=train_set,
            save_dir=active_args.run_save_dir1,
            filename_base="training_set",
            active_args=active_args,
        )
        update_train_args(active_args=active_args, train_args=train_args,save_dir=active_args.iter_save_dir1,save_dir2=active_args.run_save_dir1)
        update_train_args(active_args=active_args, train_args=train_args2,save_dir=active_args.iter_save_dir2,save_dir2=active_args.run_save_dir1)
        update_train_args(active_args=active_args, train_args=train_args3,save_dir=active_args.iter_save_dir3,save_dir2=active_args.run_save_dir1)
        update_train_args(active_args=active_args, train_args=train_args4,save_dir=active_args.iter_save_dir4,save_dir2=active_args.run_save_dir1)
        if not active_args.no_comparison_model:
            update_train_args(active_args=active_args, train_args=train_args_comparison1,save_dir=active_args.iter_save_dir1_comp,save_dir2=active_args.run_save_dir1,)        
        cross_validate(args=train_args, train_func=run_training)
        cross_validate(args=train_args2, train_func=run_training)       
        cross_validate(args=train_args3, train_func=run_training)
        cross_validate(args=train_args4, train_func=run_training)
        if not active_args.no_comparison_model:
            cross_validate(args=train_args_comparison1, train_func=run_training)

        run_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1,search_function=active_args.search_function)
        run_predictions(active_args=active_args, train_args=train_args2,gpu=active_args.gpu,save_dir=active_args.iter_save_dir2,search_function=active_args.search_function2)       
        run_predictions(active_args=active_args, train_args=train_args3,gpu=active_args.gpu,save_dir=active_args.iter_save_dir3,search_function=active_args.search_function3)
        run_predictions(active_args=active_args, train_args=train_args4,gpu=active_args.gpu,save_dir=active_args.iter_save_dir4,search_function=active_args.search_function4)
        if not active_args.no_comparison_model:
            run_predictions(active_args=active_args, train_args=train_args_comparison1,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1_comp,search_function=active_args.search_function_comparison)

        test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function)
        test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir2,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function2)
        test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir3,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function3)
        test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir4,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function4)
        if not active_args.no_comparison_model:
            test_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1_comp,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function_comparison)
        cal_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function)
        cal_predictions(active_args=active_args, train_args=train_args2,gpu=active_args.gpu,save_dir=active_args.iter_save_dir2,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function2)
        cal_predictions(active_args=active_args, train_args=train_args3,gpu=active_args.gpu,save_dir=active_args.iter_save_dir3,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function3)
        cal_predictions(active_args=active_args, train_args=train_args4,gpu=active_args.gpu,save_dir=active_args.iter_save_dir4,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function4)
        val_cal_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function)
        val_cal_predictions(active_args=active_args, train_args=train_args2,gpu=active_args.gpu,save_dir=active_args.iter_save_dir2,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function2)
        val_cal_predictions(active_args=active_args, train_args=train_args3,gpu=active_args.gpu,save_dir=active_args.iter_save_dir3,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function3)
        val_cal_predictions(active_args=active_args, train_args=train_args4,gpu=active_args.gpu,save_dir=active_args.iter_save_dir4,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function4)
        val_predictions(active_args=active_args, train_args=train_args,gpu=active_args.gpu,save_dir=active_args.iter_save_dir1,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function)
        val_predictions(active_args=active_args, train_args=train_args2,gpu=active_args.gpu,save_dir=active_args.iter_save_dir2,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function2)
        val_predictions(active_args=active_args, train_args=train_args3,gpu=active_args.gpu,save_dir=active_args.iter_save_dir3,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function3)
        val_predictions(active_args=active_args, train_args=train_args4,gpu=active_args.gpu,save_dir=active_args.iter_save_dir4,save_dir2=active_args.run_save_dir1,search_function=active_args.search_function4)        
        get_pred_results(
            active_args=active_args,
            whole_data=whole_data,
            iteration=i,
            save_error=True,
            save_dir=active_args.iter_save_dir1,
            search_function=active_args.search_function,
        )
        get_pred_results(
            active_args=active_args,
            whole_data=whole_data,
            iteration=i,
            save_error=True,
            save_dir=active_args.iter_save_dir2,
            search_function=active_args.search_function2,
        )
        get_pred_results(
            active_args=active_args,
            whole_data=whole_data,
            iteration=i,
            save_error=True,
            save_dir=active_args.iter_save_dir3,
            search_function=active_args.search_function3,
        )
        get_pred_results(
            active_args=active_args,
            whole_data=whole_data,
            iteration=i,
            save_error=True,
            save_dir=active_args.iter_save_dir4,
            search_function=active_args.search_function4,
        )
        if not active_args.no_comparison_model:
            get_pred_results(
                active_args=active_args,
                whole_data=whole_data,
                iteration=i,
                save_error=True,
                save_dir=active_args.iter_save_dir1_comp,
                search_function=active_args.search_function_comparison,
            )
        save_results(
            active_args=active_args,
            test_data=test_data,
            nontest_data=nontest_data,
            whole_data=whole_data,
            iteration=i,
            save_dir=active_args.run_save_dir1,
            save_whole_results=True,
            save_error=True,
        )
        
        rmses1.append(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir1))
        rmses2.append(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir2))
        rmses3.append(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir3))
        rmses4.append(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir4))
        rm1=(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir1))
        rm2=(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir2))
        rm3=(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir3))
        rm4=(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir4))
        if not active_args.no_comparison_model:
            rmses21.append(get_rmse(active_args=active_args,save_dir=active_args.iter_save_dir1_comp))
        spearman1,nll1,miscal1=get_scores(active_args=active_args,save_dir=active_args.iter_save_dir1)
        spearman2,nll2,miscal2=get_scores(active_args=active_args,save_dir=active_args.iter_save_dir2)
        spearman3,nll3,miscal3=get_scores(active_args=active_args,save_dir=active_args.iter_save_dir3)
        spearman4,nll4,miscal4=get_scores(active_args=active_args,save_dir=active_args.iter_save_dir4)
        if active_args.selection_method == "spearman":
            best_function.append(highest_spearman(active_args=active_args,sp1=spearman1,sp2=spearman2,sp3=spearman3,sp4=spearman4))
        elif active_args.selection_method == "rmse":
            best_function.append(lowest_rmse(active_args=active_args,score1=rm1,score2=rm2,score3=rm3,score4=rm4))
        elif active_args.selection_method == "hybrid":
            best_function.append('kmeans')
        elif active_args.selection_method == "multi":
            best_function.append('multi')
        elif active_args.selection_method == "nll":
            best_function.append(lowest_rmse(active_args=active_args,score1=nll1,score2=nll2,score3=nll3,score4=nll4))
        elif active_args.selection_method == "miscal":
            best_function.append(lowest_rmse(active_args=active_args,score1=miscal1,score2=miscal2,score3=miscal3,score4=miscal4))
        elif active_args.selection_method == "mix":
            best_function.append(mix_score(sp1=spearman1,sp2=spearman2,sp3=spearman3,sp4=spearman4,nll1=nll1,nll2=nll2,nll3=nll3,nll4=nll4
                                           ,search_function1=active_args.search_function,search_function2=active_args.search_function2,
                                           search_function3=active_args.search_function3,search_function4=active_args.search_function4))
        
        sp1.append(spearman1)
        sp2.append(spearman2)
        sp3.append(spearman3)
        sp4.append(spearman4)
        nl1.append(nll1)
        nl2.append(nll2)
        nl3.append(nll3)
        nl4.append(nll4)
        mis1.append(miscal1)
        mis2.append(miscal2)
        mis3.append(miscal3)
        mis4.append(miscal4)

        save_evaluations(active_args=active_args,rmse1=rmses1, rmse2=rmses2, rmse3=rmses3,
                          rmse4=rmses4,best_function=best_function,rmse21=rmses21,sp1=sp1,sp2=sp2,sp3=sp3,sp4=sp4
                          ,nl1=nl1,nl2=nl2,nl3=nl3,nl4=nl4,mis1=mis1,mis2=mis2,mis3=mis3,mis4=mis4)

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
    # initial_train_args.task_names = [initial_train_args.task_names[i] +f"{ActiveArgs.search_function}" for i in range(initial_train_args.num_tasks)]
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

    for i, d in enumerate(tqdm(data)):
        d.index = i
    sizes = (1 - active_args.test_fraction, active_args.test_fraction, 0)
    nontest_data, test_data, _ = split_data(
        data=data, split_type=active_args.split_type, sizes=sizes,seed=active_args.test_seed,
    )
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
     
    fraction_trainval = (
            active_args.initial_trainval_fraction * num_data / num_nontest
        )
    sizes = (fraction_trainval, 1 - fraction_trainval, 0)
    trainval_data, remaining_data, _ = split_data(
        data=nontest_data, split_type=active_args.split_type, sizes=sizes,seed=active_args.train_seed,
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

def save_dataset_indices(indices: Set[int], save_dir: str, filename_base: str) -> None:
    with open(os.path.join(save_dir, f"{filename_base}_indices.pckl"), "wb") as f:
        pickle.dump(indices, f)

def update_train_args(active_args: ActiveArgs, train_args: TrainArgs,save_dir,save_dir2) -> None:
    train_args.save_dir = save_dir
    # train_args.seed = 0
    train_args.data_path = os.path.join(save_dir2, "trainval_full.csv")
    train_args.separate_test_path = os.path.join(
        save_dir2, "test_full.csv"
    )
    if active_args.features_path is not None:
        train_args.features_path = [
            os.path.join(save_dir2, "trainval_features.csv")
        ]
        train_args.separate_test_features_path = [
            os.path.join(save_dir2, "test_features.csv")
        ]

def save_datainputs(
    active_args: ActiveArgs,
    trainval_data: MoleculeDataset,
    remaining_data: MoleculeDataset,
    test_data: MoleculeDataset,
    save_dir
) -> None:
    save_dataset(
        data=trainval_data,
        save_dir=save_dir,
        filename_base="trainval",
        active_args=active_args,
    )
    save_dataset(
        data=remaining_data,
        save_dir=save_dir,
        filename_base="remaining",
        active_args=active_args,
    )
    save_dataset(
        data=test_data,
        save_dir=save_dir,
        filename_base="test",
        active_args=active_args,
    )

def run_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu,save_dir,search_function) -> None:
    argument_input = [
        "--test_path",
        os.path.join(active_args.active_save_dir, "whole_full.csv"),
        "--checkpoint_dir",
        save_dir,
        "--preds_path",
        os.path.join(save_dir, f"whole_preds_{search_function}.csv"),
        "--evaluation_scores_path",
        os.path.join(save_dir, "evaluation_scores2.csv"),]
    if search_function != "random":
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
    if search_function == "ensemble":
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        search_function == "evidential_total"
        or search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif search_function == "dropout":
        argument_input.extend(["--uncertainty_method", "dropout"])
    elif search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)

def cal_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu,save_dir,save_dir2,search_function) -> None:
    argument_input = [
        "--test_path",
        os.path.join(save_dir2, "test_full.csv"),
        "--checkpoint_dir",
        save_dir,
        "--preds_path",
        os.path.join(save_dir, "test_pred_cal.csv"),
        "--evaluation_scores_path",
        os.path.join(save_dir, "evaluation_scores_cal.csv"),
        "--calibration_method", "zscaling",
        "--calibration_path", os.path.join(save_dir2, f"validation_set_full.csv"),

    ]
    if search_function != "random":
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
                "rmse",
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
    if search_function == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        search_function == "evidential_total"
        or search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif search_function == "dropout":
        argument_input.extend(["--uncertainty_method", "dropout"])
    elif search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)

def val_cal_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu,save_dir,save_dir2,search_function) -> None:
    argument_input = [
        "--test_path",
        os.path.join(save_dir2, f"validation_set_full.csv"),
        "--checkpoint_dir",
        save_dir,
        "--preds_path",
        os.path.join(save_dir, "val_pred_cal.csv"),
        "--evaluation_scores_path",
        os.path.join(save_dir, "evaluation_scores_cal_val.csv"),
        "--calibration_method", "zscaling",
        "--calibration_path", os.path.join(save_dir2, f"validation_set_full.csv"),

    ]
    if search_function != "random":
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
                "rmse",
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
    if search_function == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        search_function == "evidential_total"
        or search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif search_function == "dropout":
        argument_input.extend(["--uncertainty_method", "dropout"])
    elif search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)
    
#@profile
def val_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu,save_dir,save_dir2,search_function) -> None:
    argument_input = [
        "--test_path",
        os.path.join(save_dir2, f"validation_set_full.csv"),
        "--checkpoint_dir",
        save_dir,
        "--preds_path",
        os.path.join(save_dir, "val_pred.csv"),
        "--evaluation_scores_path",
        os.path.join(save_dir, "evaluation_scores_val.csv"),

    ]
    if search_function != "random":
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
                "rmse",
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
    if search_function == "ensemble":
        assert (train_args.ensemble_size != 1) or (train_args.num_folds != 1)
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        search_function == "evidential_total"
        or search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif search_function == "dropout":
        argument_input.extend(["--uncertainty_method", "dropout"])
    elif search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)
    
def test_predictions(active_args: ActiveArgs, train_args: TrainArgs,gpu,save_dir,save_dir2,search_function) -> None:
    argument_input = [
        "--test_path",
        os.path.join(save_dir2, "test_full.csv"),
        "--checkpoint_dir",
        save_dir,
        "--preds_path",
        os.path.join(save_dir, "test_preds.csv"),
        "--evaluation_scores_path",
        os.path.join(save_dir, "evaluation_scores.csv"),

    ]
    if search_function != "random":
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
    if search_function == "ensemble":
        argument_input.extend(["--uncertainty_method", "ensemble"])
    elif search_function == "mve":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif search_function == "mve_ensemble":
        argument_input.extend(["--uncertainty_method", "mve"])
    elif (
        search_function == "evidential_total"
        or search_function == "evidential"
    ):
        argument_input.extend(["--uncertainty_method", "evidential_total"])
    elif search_function == "evidential_aleatoric":
        argument_input.extend(["--uncertainty_method", "evidential_aleatoric"])
    elif search_function == "evidential_epistemic":
        argument_input.extend(["--uncertainty_method", "evidential_epistemic"])
    elif search_function == "dropout":
        argument_input.extend(["--uncertainty_method", "dropout"])
    elif search_function == "random":
        pass
    else:
        raise ValueError(
            f"The search function {search_function}" + "is not supported."
        )
    pred_args = PredictArgs().parse_args(argument_input)
    make_predictions(pred_args)

def get_pred_results(
    active_args: ActiveArgs,
    whole_data: MoleculeDataset,
    iteration: int,
    save_dir,
    search_function,
    save_error=False,
) -> None:
    with open(os.path.join(save_dir, f"whole_preds_{search_function}.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                whole_data[i].output[
                    j + f"_{active_args.train_sizes[iteration]}"
                ] = float(
                    line[j]
                )  # exp_#
                if search_function == "ensemble":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(
                        line[j + "_ensemble_uncal_var"]
                    )  # exp_unc_#
                elif search_function == "mve":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_mve_uncal_var"])
                elif search_function == "mve_ensemble":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_mve_uncal_var"])
                elif (
                    search_function == "evidential_total"
                    or search_function == "evidential"
                ):
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_total_uncal_var"])
                elif search_function == "evidential_aleatoric":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_aleatoric_uncal_var"])
                elif search_function == "evidential_epistemic":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_evidential_epistemic_uncal_var"])
                elif search_function == "dropout":
                    whole_data[i].output[
                        j + f"_unc_{search_function}_{active_args.train_sizes[iteration]}"
                    ] = float(line[j + "_dropout_uncal_var"])
                if save_error:
                    whole_data[i].output[
                        j + f"_error_{active_args.train_sizes[iteration]}"
                    ] = abs(float(line[j]) - whole_data[i].output[j])

def save_results(
    active_args: ActiveArgs,
    test_data: MoleculeDataset,
    nontest_data: MoleculeDataset,
    whole_data: MoleculeDataset,
    iteration: int,
    save_dir,
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
    with open(os.path.join(save_dir, "test_results.csv"), "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for d in test_data:
            writer.writerow(d.output)
    with open(
        os.path.join(save_dir, "nontest_results.csv"), "w"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for d in nontest_data:
            writer.writerow(d.output)
    if save_whole_results:
        with open(
            os.path.join(save_dir, "whole_results.csv"), "w"
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
    search_function,
    best_function,
    save_new_indices: bool = True,
    save_full_indices: bool = False,
) -> Tuple[MoleculeDataset]:
    print('-------------------------------------------------')
    print(iteration)
    print(search_function)
    print(active_args.search_function)
    print(active_args.hybrid_count)
    print(active_args.selection_method)
    print(best_function)
    print('-------------------------------------------------')
    if iteration >= active_args.hybrid_count and (best_function[0] == "kmeans" or best_function[0]  == "multi"):
        search_function = active_args.search_function
    print(search_function)
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
    if active_args.selection_method != "hybrid" and active_args.selection_method != "multi":
        if active_args.search_function != "random":  # only for a single task
            priority_values = [
                d.output[
                    active_args.task_names[0]
                    + f"_unc_{search_function}_{active_args.train_sizes[iteration-1]}"
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
    elif active_args.selection_method == "hybrid":
        smiles=get_fingerprint(previous_remaining_data=previous_remaining_data,active_args=active_args,gpu=active_args.gpu,i=iteration)
        smiles_=MoleculeDataset.smiles(previous_remaining_data) 
        new_indices=[smiles_.index(smiles[i]) for i in range(len(smiles))]
        new_data=MoleculeDataset([previous_remaining_data[i] for i in new_indices])


        new_data_indices = new_indices
        updated_trainval_data = MoleculeDataset(new_data + previous_trainval_data)
        updated_remaining_data = MoleculeDataset(
            [d for d in previous_remaining_data if d.index not in new_indices]
        )
    elif active_args.selection_method == "multi":
        priority_values = [
                d.output[
                    active_args.task_names[0]
                    + f"_unc_{active_args.search_function}_{active_args.train_sizes[iteration-1]}"
                ]
                for d in previous_remaining_data
            ]
        priority_values2 = [
                d.output[
                    active_args.task_names[0]
                    + f"_unc_{active_args.search_function2}_{active_args.train_sizes[iteration-1]}"
                ]
                for d in previous_remaining_data
            ]
        priority_values3 = [
                d.output[
                    active_args.task_names[0]
                    + f"_unc_{active_args.search_function3}_{active_args.train_sizes[iteration-1]}"
                ]
                for d in previous_remaining_data
            ]
        priority_values4 = [
                d.output[
                    active_args.task_names[0]
                    + f"_unc_{active_args.search_function4}_{active_args.train_sizes[iteration-1]}"
                ]
                for d in previous_remaining_data
            ]
        smiles_=MoleculeDataset.smiles(previous_remaining_data)
        sorted_data1=sorted(zip(priority_values,smiles_),reverse=True)
        sorted_data2=sorted(zip(priority_values2,smiles_),reverse=True)
        sorted_data3=sorted(zip(priority_values3,smiles_),reverse=True)
        sorted_data4=sorted(zip(priority_values4,smiles_),reverse=True)
        indices1,indices2,indices3,indices4=[],[],[],[]
        for value in sorted_data1[0:active_args.active_batch_size]:
            indices1.append(smiles_.index(value[1]))
        for value in sorted_data2[0:active_args.active_batch_size]:
            indices2.append(smiles_.index(value[1]))
        for value in sorted_data3[0:active_args.active_batch_size]:
            indices3.append(smiles_.index(value[1]))
        for value in sorted_data4[0:active_args.active_batch_size]:
            indices4.append(smiles_.index(value[1]))
        
        unique_combined_indices = []
        seen_numbers = set()

        for sublist in zip(indices1, indices2, indices3, indices4):
            for item in sublist:
                if item not in seen_numbers:
                    unique_combined_indices.append(item)
                    seen_numbers.add(item)
        new_indices=unique_combined_indices[0:active_args.active_batch_size]
        new_data=MoleculeDataset([previous_remaining_data[i] for i in new_indices])
        new_data_indices = unique_combined_indices
        updated_trainval_data = MoleculeDataset(new_data + previous_trainval_data)
        updated_remaining_data = MoleculeDataset(
            [d for d in previous_remaining_data if d.index not in new_indices]
        )
        # assert False

    return updated_trainval_data, updated_remaining_data



    




def get_rmse(active_args,save_dir):
    with open(os.path.join(save_dir, "test_scores.csv"), "r") as f:
        reader = csv.DictReader(f)
        for i, line in enumerate(tqdm(reader)):
            for j in active_args.task_names:
                rmse = float(line["Mean rmse"])
    return rmse

def get_scores(active_args,save_dir):
    with open(
            os.path.join(save_dir, "evaluation_scores_cal.csv"), "r"
        ) as file:
            csv_reader = csv.reader(file)
            rows = list(csv_reader)
            transposed_rows = list(zip(*rows))
    with open(
            os.path.join(save_dir, "evaluation_scores_cal.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(transposed_rows)

    with open(
            os.path.join(save_dir, "evaluation_scores_cal.csv"), "r"
        ) as f:
            reader = csv.DictReader(f)
            for i, line in enumerate(tqdm(reader)):
                for j in active_args.task_names:
                    spearmans = float(line["spearman"])
                    nlls = float(line["nll"])
                    miscalibration_areas = float(line["miscalibration_area"])
    return spearmans,nlls,miscalibration_areas

def highest_spearman(active_args,sp1,sp2,sp3,sp4):
    functions=[active_args.search_function,active_args.search_function2,active_args.search_function3,active_args.search_function4]
    sp=[sp1,sp2,sp3,sp4]
    combined = list(zip(functions, sp))
    combined.sort(key=lambda x: x[1], reverse=True)
    highest_function, highest_sp = combined[0]
    return highest_function

def lowest_rmse(active_args,score1,score2,score3,score4):
    functions=[active_args.search_function,active_args.search_function2,active_args.search_function3,active_args.search_function4]
    rmse=[score1,score2,score3,score4]
    combined = list(zip(functions, rmse))
    combined.sort(key=lambda x: x[1], reverse=False)
    lowest_function, lowest_score = combined[0]
    return lowest_function

def mix_score(sp1, sp2, sp3, sp4, nll1, nll2, nll3, nll4, search_function1, search_function2, search_function3, search_function4):
    # Combine sp values into a list
    sp_values = [sp1, sp2, sp3, sp4]

    # Combine nll values into a list
    nll_values = [nll1, nll2, nll3, nll4]

    # Combine search functions into a list
    search_functions = [search_function1, search_function2, search_function3, search_function4]

    # Zip and sort based on sp values
    sorted_sp_data = sorted(zip(sp_values, nll_values, search_functions), key=lambda x: x[0], reverse=True)

    # Get the highest sp value
    highest_sp = sorted_sp_data[0][0]

    # Get all search functions with sp values close to the highest sp by one unit
    close_sp_functions = [func for sp, _, func in sorted_sp_data if abs(highest_sp - sp) <= 0.05]

    # If there's only one function close to the highest sp, return it
    if len(close_sp_functions) == 1:
        return close_sp_functions[0]

    # If there are multiple functions close to the highest sp, compare based on nll values
    # Zip and sort based on nll values in ascending order (lower is better)
    sorted_nll_data = sorted(zip(nll_values, search_functions), key=lambda x: x[0])

    # Get the lowest nll value
    lowest_nll = sorted_nll_data[0][0]

    # Get all search functions with nll values close to the lowest nll by one unit
    close_nll_functions = [func for nll, func in sorted_nll_data if abs(lowest_nll - nll) <= 1]

    # Return the first function in the close_nll_functions list (you can modify this logic if needed)
    return close_nll_functions[0] if close_nll_functions else None





def save_evaluations(active_args,rmse1,rmse2,rmse3,rmse4,best_function,rmse21,sp1,sp2,sp3,sp4,nl1,nl2,nl3,nl4,mis1,mis2,mis3,mis4):
    with open(
        os.path.join(active_args.active_save_dir, "uncertainty_evaluations.csv"),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        header = ["data_points",f"rmse_{active_args.search_function}",f"rmse_{active_args.search_function2}",
                  f"rmse_{active_args.search_function3}",f"rmse_{active_args.search_function4}",
                  f"rmse2",
                  f"spearman_{active_args.search_function}",f"spearman_{active_args.search_function2}",
                  f"spearman_{active_args.search_function3}",f"spearman_{active_args.search_function4}",
                  f"nll_{active_args.search_function}",f"nll_{active_args.search_function2}",
                  f"nll_{active_args.search_function3}",f"nll_{active_args.search_function4}",
                  f"miscalibration_area_{active_args.search_function}",f"miscalibration_area_{active_args.search_function2}",
                  f"miscalibration_area_{active_args.search_function3}",f"miscalibration_area_{active_args.search_function4}",
                  "best_function"]
    
        writer.writerow(header)
        for i in range(len(rmse1)):
            new_row = [
                active_args.train_sizes[i],
                rmse1[i],rmse2[i],rmse3[i],rmse4[i],
                rmse21[i],
                sp1[i],sp2[i],sp3[i],sp4[i],
                nl1[i],nl2[i],nl3[i],nl4[i],
                mis1[i],mis2[i],mis3[i],mis4[i],
                best_function[i],
                
            ]
            writer.writerow(new_row)

def get_fingerprint(previous_remaining_data:MoleculeDataset,active_args:ActiveArgs,gpu,i) -> Tuple[MoleculeDataset]:
    previous_remaining_data = previous_remaining_data.smiles() # it has to change to remaining data
    argument_input = [
        "--test_path",
        os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i-1]}","remaining_full.csv"),
            # Results\train10\mve_10\selection
        "--checkpoint_dir",
        os.path.join(
            active_args.active_save_dir,
            f"train{active_args.train_sizes[i-1]}",
            f"{active_args.search_function}_{active_args.train_sizes[i-1]}",
            f"selection"),
        "--preds_path",
        os.path.join(active_args.active_save_dir, "finger_print.csv"),
        "--num_workers",0
    ]
    if gpu is not None:
        argument_input.extend(["--gpu", str(gpu)])
    fp_args = FingerprintArgs().parse_args(argument_input)
    x=molecule_fingerprint(fp_args)
    del x

    lists_per_row = []
    with open(os.path.join(active_args.active_save_dir, "finger_print.csv")) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        lists_per_row = [list(map(float, row[1:])) for row in csvreader]
    lists_per_row=np.array(lists_per_row)
    mean = np.mean(lists_per_row, axis=0)
    std_dev = np.std(lists_per_row, axis=0)
    standardized_data = (lists_per_row - mean) / std_dev

    # replace nan with 0
    nan_indices = np.isnan(standardized_data[0])

    standardized_data = standardized_data[:, ~nan_indices]
    kmeans = KMeans(n_clusters=active_args.active_batch_size,random_state=0)
    cluster_labels = kmeans.fit_predict(standardized_data)
    cluster_assignments = kmeans.predict(standardized_data)
    distances = cdist(standardized_data, kmeans.cluster_centers_, 'euclidean')
    closest_points_indices = [distances[:, i].argmin() for i in range(active_args.active_batch_size)]
    closest_points = [standardized_data[i] for i in closest_points_indices] 
    smiles=[]
    adding_fp=[]
    smiles=[previous_remaining_data[i] for i in closest_points_indices]
    adding_fp=[standardized_data[i] for i in closest_points_indices]
    with open(
            os.path.join(active_args.iter_save_dir1, "added_fp.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(adding_fp)
    with open(
            os.path.join(active_args.iter_save_dir1, "added_smiles.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(smiles)
    with open(
            os.path.join(active_args.iter_save_dir1, "closest_points.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(closest_points)
    with open(
            os.path.join(active_args.iter_save_dir1, "scaled_data.csv"),
            "w",
            newline="",
        ) as file:
            csv_writer = csv.writer(file)
            csv_writer.writerows(standardized_data)
    del standardized_data
    del closest_points
    del adding_fp
    del lists_per_row

    assert len(smiles)==active_args.active_batch_size, f"Smiles: {len(smiles)}, Active batch size: {active_args.active_batch_size}"
    return smiles
if __name__ == "__main__":
    parallel_training(ActiveArgs().parse_args())

