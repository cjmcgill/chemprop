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
from chemprop.data import get_task_names, get_data, MoleculeDataset, split_data
from chemprop.train import cross_validate, run_training
from chemprop.utils import makedirs

import matplotlib.pyplot as plt


class ActiveArgs(Tap):

    save_dir: str  # where to save the plots
    evaluation_path: str  # location of uncertainty_evaluations.csv file
    no_spearman: bool = False  # will not plot spearman if it is True
    no_sharpness: bool  = False  # will not plot sharpness if it is True
    no_sharpness_root: bool = False  # will not plot sharpness_root if it is True
    no_cv: bool = False  # will not plot cv if it is True
    no_rmse: bool = False  # will not plot rmse if it is True
    no_rmse2: bool = False # will not plot rmse if it is True
    no_ence: bool =False  # will not plot ence if it is True
    no_miscalibration_area: bool = False # will not plot calibration_area if it is True
    no_true_vs_predicted: bool = False  # will not plot true vs predicted value if it is True
    no_subplot: bool = False  # will not plot subplots if it is True
    no_nll: bool = False  # will not plot nll if it is True


spearmans, cv, rmses, rmses2, sharpness = [], [], [], [], []
nlls, miscalibration_areas, ences, sharpness_root = [], [], [], []
rmses, rmses2, data_points = [], [], []
def plot_results(active_args: ActiveArgs):
    spearmans1, cv1, rmses1, rmses21, sharpness1, data_point1 = [], [], [], [], [], []
    nlls1, miscalibration_areas1, ences1, sharpness_root1 = [], [], [], []

    spearmans1, nlls1, miscalibration_areas1, ences1, sharpness1, sharpness_root1, cv1,  rmses1, rmses21, data_point1 = read_result(active_args=active_args)
    
    # rmses.append(rmses1)
    # rmses2.append(rmses21)
    # sharpness_root.append(sharpness_root1)
    # spearmans.append(spearmans1)
    # nlls.append(nlls1)
    # miscalibration_areas.append(miscalibration_areas1)
    # ences.append(ences1)
    # sharpness.append(sharpness1)
    # sharpness_root.append(sharpness_root1)
    # cv.append(cv1)
    # data_points.append(data_point1)
    active_args.plot_save_dir=os.path.join(active_args.save_dir,'plot')
    makedirs(active_args.plot_save_dir)
    if not active_args.no_rmse:
        plot_rmse(active_args=active_args,rmses=rmses1,data_points=data_point1)
    if not active_args.no_rmse2:
        plot_rmse2(active_args=active_args,rmses=rmses21,data_points=data_point1)        
    if not active_args.no_spearman:
        plot_spearman(active_args=active_args,spearmans=spearmans1,data_points=data_points)
    if not active_args.no_sharpness:
        plot_sharpness(active_args=active_args,sharpness=sharpness1,data_points=data_points)
    if not active_args.no_sharpness_root:
        plot_sharpness_root(active_args=active_args,sharpness_root=sharpness_root1,data_points=data_points)
    if not active_args.no_rmse2:
        plot_cv(active_args=active_args,cv=cv1,data_points=data_points)
    if not active_args.no_miscalibration_area:
        plot_miscalibration_area(active_args=active_args,miscalibration_area=miscalibration_areas1,data_points=data_points)
    if not active_args.no_rmse2:
        plot_ence(active_args=active_args,ences=ences1,data_points=data_points)   
    if not active_args.no_subplot:
         sub_plot(active_args=active_args,rmses=rmses1,rmses2=rmses21,spearman=spearmans1,cv=cv1)
    if not active_args.no_nll:
         plot_nll(active_args=active_args,nll=nlls1,data_points=data_points)
    # if not active_args.no_true_vs_predicted:
    #     plot_true_predicted(active_args=active_args,true=true,predicted=predicted)
    
        

    
    

    # plot_true_predicted()


def read_result(active_args: ActiveArgs):
        with open(active_args.evaluation_path, "r") as f:
            
            reader = csv.DictReader(f)
            for i, line in enumerate(tqdm(reader)):
                    spearmans.append(float(line['spearman']))
                    nlls.append(float(line['nll']))
                    miscalibration_areas.append(float(line['miscalibration_area']))
                    ences.append(float(line['ence']))
                    sharpness.append(float(line['sharpness']))
                    sharpness_root.append(float(line['sharpness_root']))
                    cv.append(float(line['cv']))
                    rmses.append(float(line['rmse']))
                    rmses2.append(float(line['rmse2']))
                    data_points.append(float(line['data_points']))

        return spearmans, nlls, miscalibration_areas, ences, \
        sharpness, sharpness_root, cv, rmses, rmses2, data_points
    
def plot_rmse(active_args, rmses, data_points):
        if len(rmses)==len(data_points):
            plt.scatter(data_points, rmses)
            plt.plot(data_points, rmses, '-')
            plt.xlabel("Data Points")
            plt.ylabel("RMSE")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'rmse_plot.png'))
            plt.clf()

def plot_rmse2(active_args, rmses, data_points):
        if len(rmses)==len(data_points):
            plt.scatter(data_points, rmses)
            plt.plot(data_points, rmses, '-')
            plt.xlabel("Data Points")
            plt.ylabel("RMSE2")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'rmse2_plot.png'))
            plt.clf()


def plot_spearman(active_args, spearmans, data_points):
        if len(rmses)==len(data_points):
            plt.scatter(data_points, spearmans)
            plt.plot(data_points, spearmans, '-')
            plt.xlabel("Data Points")
            plt.ylabel("Spearman")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'spearman_plot.png'))
            plt.clf()

def plot_sharpness(active_args, sharpness, data_points):
        if len(sharpness)==len(data_points):
            plt.scatter(data_points, sharpness)
            plt.plot(data_points, sharpness, '-')
            plt.grid(True)
            plt.xlabel("Data Points")
            plt.ylabel("Sharpness")
            plt.savefig(os.path.join(active_args.plot_save_dir,'sharpness_plot.png'))
            plt.clf()


def plot_sharpness_root(active_args, sharpness_root, data_points):
        if len(sharpness_root)==len(data_points):
            plt.scatter(data_points, sharpness_root)
            plt.plot(data_points, sharpness_root, '-')
            plt.xlabel("Data Points")
            plt.ylabel("Sharpness_Root")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'sharpness_root_plot.png'))
            plt.clf()

def plot_cv(active_args, cv, data_points):
        if len(cv)==len(data_points):
            plt.scatter(data_points, cv)
            plt.plot(data_points, cv, '-')
            plt.xlabel("Data Points")
            plt.ylabel("CV")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'cv_plot.png'))
            plt.clf()


def plot_miscalibration_area(active_args, miscalibration_area, data_points):
        if len(miscalibration_area)==len(data_points):
            plt.scatter(data_points, miscalibration_area)
            plt.plot(data_points, miscalibration_area, '-')
            plt.xlabel("Data Points")
            plt.ylabel("Miscalibration_Area")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'miscalibration_area_plot.png'))
            plt.clf()

def plot_ence(active_args, ences, data_points):
        if len(ences)==len(data_points):
            plt.scatter(data_points, ences)
            plt.plot(data_points, ences, '-')
            plt.xlabel("Data Points")
            plt.ylabel("ENCE")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'ence_plot.png'))
            plt.clf()



def sub_plot(active_args, rmses, rmses2, spearman, cv):
        fig, axs = plt.subplots(4)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(data_points, rmses, linewidth=3, color='g')
        axs[1].plot(data_points, rmses2,linewidth=3, color='r')
        axs[2].plot(data_points, spearman,linewidth=3, color='m')
        axs[3].plot(data_points, cv,linewidth=3, color='b')
        # axs[0].set_title('Plot 1')
        # axs[1].set_title('Plot 2')
        # axs[2].set_title('Plot 3')
        axs[3].set_xlabel('Data Points')
        axs[0].set_ylabel('RMSE')
        axs[1].set_ylabel('RMSE2')
        axs[2].set_ylabel('SpearMan')
        axs[3].set_ylabel('CV')
        for ax in axs:
            ax.grid(True)
        for ax in axs:
            ax.spines['top'].set_linewidth(2)    
            ax.spines['bottom'].set_linewidth(2) 
            ax.spines['left'].set_linewidth(2)   
            ax.spines['right'].set_linewidth(2)
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(os.path.join(active_args.plot_save_dir,'sub_plot.png'))
        plt.clf()
     

def plot_nll(active_args,nll,data_points):
     if len(nll)==len(data_points):
            plt.scatter(data_points, nll)
            plt.plot(data_points, nll, '-')
            plt.xlabel("Data Points")
            plt.ylabel("NLL")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'nll_plot.png'))
            plt.clf()

if __name__ == '__main__':
    plot_results(ActiveArgs().parse_args())
