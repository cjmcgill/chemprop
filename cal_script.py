from chemprop.data import get_data,MoleculeDataset, split_data
import numpy as np
import pandas as pd
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from tap import Tap
import os
from scipy.stats import  spearmanr
from scipy.special import erfinv
# class ActiveArgs(Tap):

#     main_path: str  # where all the data is
#     method_name: str  # name of the method
#     number_seed: int = 10 # number of seeds

num_seeds=10
trainval_csv_filename = "trainval_full.csv"
preds_csv_filename = "whole_preds.csv"
test_preds_filename = "test_preds.csv"
test_taget_filename = "test_results.csv"
uncertainty_evaluations_filename = "uncertainty_evaluations.csv"
# methods=['Dropout','Ensemble5','Evidential_Alea','Evidential_Alea_Ensemble',
#          'Evidential_Alea02','Evidential_Alea02_Ensemble','Evidential_Ensemble_Epi','Evidential_Ensemble_Tot',
#         'Evidential_Ensemble_Tot02','Evidential_Epi','Evidential_Epi02','Evidential_Epi02_Ensemble','Evidential_Tot','Evidential_Tot02'
#         ,'MVE','MVE_Ensemble5']
methods=['Evidential_Tot02']


################################################################################################
def process_folders(root_path, trainval_csv_filename, preds_csv_filename):
    cal_spearman,cal_cv,cal_sharpness,cal_nll,cal_ence,cal_miscal=[],[],[],[],[],[]
    test_targets=pd.read_csv(os.path.join(root_path, test_taget_filename))#,usecols=[1])
    test_targets=test_targets.iloc[:, 1]
    evaluation=pd.read_csv(os.path.join(root_path, uncertainty_evaluations_filename))
    for folder_name in sorted(os.listdir(root_path)):
        folder_path = os.path.join(root_path, folder_name)
        
        
        

        if os.path.isdir(folder_path):
            trainval_path = find_csv_in_folder(folder_path, trainval_csv_filename)

            if trainval_path:
                # Process the main CSV file
                # trainval = pd.read_csv(trainval_path)
                
                create_calibration_folder(folder_path)

            selection_folder_path = find_selection_folder(folder_path)
            if selection_folder_path:
                preds_csv_path = find_csv_in_folder(selection_folder_path, preds_csv_filename)
                test_preds_path = find_csv_in_folder(selection_folder_path, test_preds_filename)
                if test_preds_path:
                    test_preds = pd.read_csv(test_preds_path)
                    
                if preds_csv_path:
                    # Process the 'whole_preds.csv' file
                    whole_preds = pd.read_csv(preds_csv_path)

            # trainval_preds = pd.merge(trainval[['smiles']], whole_preds, on='smiles', how='left')
            trainval_data = get_data(
            path=trainval_path,)
            validation_set, train_set,_  = split_data(
                        data=trainval_data,
                        sizes=(0.2, 0.8, 0),)

            val_smiles=validation_set.smiles()
            val_smiles = [item for sublist in val_smiles for item in sublist]
            val_preds = pd.DataFrame(columns=whole_preds.columns)
            for i,s in enumerate(val_smiles):
                s_preds = whole_preds[whole_preds['smiles'] == s]
                val_preds.loc[i] = s_preds.iloc[0]
            val_pred=val_preds.iloc[:, 1]
            val_vars=val_preds.iloc[:, 2]
            test_pred=test_preds.iloc[:, 1]
            test_vars=test_preds.iloc[:, 2]
            val_targets=validation_set.targets()
            val_pred = np.array(list(val_pred))
            val_vars = np.array(list(val_vars))
            val_targets = np.array(list(val_targets))
            num_tasks=validation_set.num_tasks()
            scaling = np.zeros(num_tasks)
            for i in range(num_tasks):
                        # shape_og=val_pred.shape
                        # task_targets = targets
                        val_pred = val_pred.reshape(val_targets.shape)
                        val_vars = val_vars.reshape(val_targets.shape)
                        task_errors = val_pred - val_targets
                        task_zscore = task_errors / np.sqrt(val_vars)
                        def objective(scaler_value: float):
                            scaled_vars = val_vars * scaler_value ** 2
                            nll = np.log(2 * np.pi * scaled_vars) / 2 \
                                + (task_errors) ** 2 / (2 * scaled_vars)
                            return nll.sum()
                        initial_guess = np.std(task_zscore)
                        sol = fmin(objective, initial_guess)
                        scaling[i] = sol


            ############################# ERROR FRACTION ##########################################
            error_val_after=[]
            error_val_before=[]
            error_test_before=[]
            error_test_after=[]
            for i in range(len(val_pred)):
                error_val_after.append((abs(val_pred[i] - val_targets[i]))/(np.sqrt(val_vars[i])*np.sqrt(scaling[0])))
                error_val_before.append((abs(val_pred[i] - val_targets[i]))/(np.sqrt(val_vars[i])))
            for i in range(len(test_pred)):
                error_test_after.append((abs(test_pred[i] - test_targets[i]))/(np.sqrt(test_vars[i])*np.sqrt(scaling[0])))
                error_test_before.append((abs(test_pred[i] - test_targets[i]))/(np.sqrt(test_vars[i])))
            val_after=(sum(1 for num in error_val_after if num < 1) / len(error_val_after)) 
            val_before=(sum(1 for num in error_val_before if num < 1) / len(error_val_before)) 
            test_after=(sum(1 for num in error_test_after if num < 1) / len(error_test_after))
            test_before=(sum(1 for num in error_test_before if num < 1) / len(error_test_before))
            with open(os.path.join(folder_path, 'Calibration','error_fraction.txt'), "w") as file:
            # Write the numbers to the file
                file.write(f"Validation: \nerror fraction before: {val_before}\nerror fraction after: {val_after}\nTest: \nerror fraction before: {test_before}\nerror fraction after: {test_after}\n")



            ############################# PLOTTING ##########################################
            val_vars = [item for sublist in val_vars for item in sublist]
            val_preds = [item for sublist in val_pred for item in sublist]
            val_targets = [item for sublist in val_targets for item in sublist]
            test_targets=np.array(list(test_targets))
            test_vars=np.array(list(test_vars))
            test_pred=np.array(list(test_pred))
            plt.figure()
            plt.errorbar(val_targets, val_preds, yerr=np.sqrt(val_vars),
                        fmt='o', markersize=8, capsize=5, color='b', ecolor='red', label='Data with Error Bars')
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], '--', color='gray', label='Perfect Agreement')
            plt.scatter(val_targets,val_preds, color='red')
            plt.savefig(os.path.join(folder_path, 'Calibration', 'parity_val_before.png'),dpi=300)
            plt.figure()
            plt.errorbar(val_targets, val_preds, yerr=np.sqrt(val_vars)*scaling,
                        fmt='o', markersize=8, capsize=5, color='b', ecolor='red', label='Data with Error Bars')
            plt.plot([min(val_targets), max(val_targets)], [min(val_targets), max(val_targets)], '--', color='gray', label='Perfect Agreement')
            plt.scatter(val_targets,val_preds, color='red')
            plt.savefig(os.path.join(folder_path, 'Calibration', 'parity_val_after.png'),dpi=300)
            plt.figure()
            plt.errorbar(test_targets, test_pred, yerr=np.sqrt(test_vars),
                        fmt='o', markersize=8, capsize=5, color='b', ecolor='red', label='Data with Error Bars')
            plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], '--', color='gray', label='Perfect Agreement')
            plt.scatter(test_targets,test_pred, color='red')
            plt.savefig(os.path.join(folder_path, 'Calibration', 'parity_test_before.png'),dpi=300)
            plt.figure()
            plt.errorbar(test_targets, test_pred, yerr=np.sqrt(test_vars)*scaling,
                        fmt='o', markersize=8, capsize=5, color='b', ecolor='red', label='Data with Error Bars')
            plt.plot([min(test_targets), max(test_targets)], [min(test_targets), max(test_targets)], '--', color='gray', label='Perfect Agreement')
            plt.scatter(test_targets,test_pred, color='red')
            plt.savefig(os.path.join(folder_path, 'Calibration', 'parity_test_after.png'),dpi=300)
            cal_spearman.append(spearman(test_targets,test_pred,test_vars*scaling))
            cal_cv.append(coefficient_variance(test_targets,test_pred,test_vars*scaling))
            cal_sharpness.append(sharpness(test_targets,test_pred,test_vars*scaling))
            cal_nll.append(nll(test_targets,test_pred,test_vars*scaling))
            cal_miscal.append(miscal(test_targets,test_pred,test_vars*scaling,num_tasks))
            cal_ence.append(enh_ence(test_targets,test_pred,test_vars*scaling))
            
    evaluation['spearman'] = cal_spearman
    evaluation['cv'] = cal_cv
    evaluation['sharpness'] = cal_sharpness
    evaluation['sharpness_root'] = [np.sqrt(i) for i in cal_sharpness]
    evaluation['nll'] = cal_nll
    evaluation['miscalibration_area'] = cal_miscal
    evaluation['ence'] = cal_ence
    evaluation.to_csv(os.path.join(root_folder_path,'uncertainty_evaluation_cal.csv'), index=False)



def find_csv_in_folder(folder_path, trainval_csv_filename):
    for file_name in os.listdir(folder_path):
        if file_name == trainval_csv_filename:
            return os.path.join(folder_path, file_name)

    return None

def find_selection_folder(folder_path):
    for subfolder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder_name)
        if os.path.isdir(subfolder_path) and subfolder_name.lower().startswith('selection'):
            return subfolder_path

def create_calibration_folder(main_folder_path):

    calibration_folder_path = os.path.join(main_folder_path, 'Calibration')
    
    # Check if 'Calibration' folder already exists
    if not os.path.exists(calibration_folder_path):
        os.mkdir(calibration_folder_path)


    return None

def spearman(target,pred,var):
    error = np.abs(pred - target)
    spmn = spearmanr(var, error).correlation
    return spmn

def coefficient_variance(target,pred,var):
    cv=(np.sqrt((sum(np.array((np.sqrt(np.array(var))-np.mean(var))**2))/(len(np.array((np.sqrt(np.array(var))-np.mean(var))**2))-1))))/(np.mean(var))
    return cv

def sharpness(target,pred,var):
    sharpness=np.mean(var)
    return sharpness

def nll(target,pred,var):
    error = np.abs(pred - target)
    nll = np.log(2 * np.pi * var) / 2 + (error) ** 2 / (2 * var)
    return nll.mean()


######### ENCANCED OG
def enh_ence(true_values, predicted_values, variance, num_bins=100):

    # Calculate the absolute errors
    absolute_errors = np.abs(predicted_values - true_values)

    # Sort the data based on variance
    sort_idx = np.argsort(variance)
    sorted_var = variance[sort_idx].copy()
    sorted_errors = absolute_errors[sort_idx].copy()

    # Split data into bins
    split_vars = np.array_split(sorted_var, num_bins)
    split_errors = np.array_split(sorted_errors, num_bins)

    # Calculate root mean variance and root mean squared error for each bin
    root_mean_vars = np.sqrt([np.mean(bin_vars) for bin_vars in split_vars])
    rmses = np.sqrt([np.mean(np.square(bin_errors)) for bin_errors in split_errors])

    # Calculate ENCE
    ence_score = np.mean(np.abs(root_mean_vars - rmses) / (root_mean_vars + 1e-10))  # Add a small constant to avoid division by zero

    return ence_score

######### MY ENCE
# def my_ence(target,pred,var,num_tasks):
#     sort_idx = np.argsort(var)
#     var = var[sort_idx]
#     pred = pred[sort_idx]
#     target = target[sort_idx]
#     var_bin=np.array_split(var,100)
#     target_bin=np.array_split(target,100)
#     pred_bin=np.array_split(pred,100)
#     ence=[]
#     for i in range(100):
#         rmse = np.sqrt(np.mean((np.array(target_bin[i]) - np.array(pred_bin[i]))**2))
#         root_mean_var=np.sqrt(np.mean(np.array(var_bin[i])))
#         ence.append(np.abs(root_mean_var-rmse)/root_mean_var)
#     return np.mean(ence)


def miscal(target,pred,var,num_tasks):
    fractions = np.zeros([num_tasks, 101])  # shape(tasks, 101)
    fractions[:, 100] = 1
    bin_scaling = [0]

    for i in range(1, 100):
        bin_scaling.append(erfinv(i / 100) * np.sqrt(2))

    error = np.abs(pred - target)

    for i in range(1, 100):
        bin_unc = np.sqrt(var) * bin_scaling[i]
        bin_fraction = np.mean(bin_unc >= error)
        fractions[0, i] = bin_fraction

        # trapezoid rule
        auce = np.sum(
            0.01 * np.abs(fractions - np.expand_dims(np.arange(101) / 100, axis=0)),
            axis=1,
        )
    return auce[0]
    


if __name__ == "__main__":
    for method in methods:
            # i=2
        for i in range(num_seeds):
            # root_folder_path=rf"/lustre/home/rafeik/Results/Serious/atom/Active_Learning/{method}_{i+1}"
            root_folder_path = rf"F:\Results\Qm9_U0_atom\Active_Learning\{method}_{i+1}"
            process_folders(root_folder_path, trainval_csv_filename, preds_csv_filename)

