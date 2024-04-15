import os
import csv
from tap import Tap
from tqdm import tqdm
import numpy as np
import pandas as pd
from chemprop.utils import makedirs
import matplotlib.pyplot as plt


class ActiveArgs(Tap):

    save_dir: str  # where to save the plots
    evaluation_path: str  # location of uncertainty_evaluations.csv file
    results_path: str  # location of test_results.csv file
    value_name: str  # name of the value as it shows in test_results.csv
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
    subplot_size: list  # size of the subplot, number of rows x number of columns for vertical subplot enter one number
    subplot_x: list  # x axis of subplots
    subplot_y: list  # y axis of subplots
    """
    available inputs for x and y:
    [ '1rmse','2rmse','nll','spearman','data_point','miscalibration_area',
    'cv','ence','sharpness','sharpness_root','first','last','true' ] 
    """




def plot_results(active_args: ActiveArgs):
    spearmans1, nlls1, miscalibration_areas1, ences1, sharpness1, sharpness_root1, cv1,  rmses1, rmses21, data_point1, first, last, true = read_result(active_args=active_args)
    active_args.plot_save_dir=os.path.join(active_args.save_dir,'plot')
    makedirs(active_args.plot_save_dir)
    if not active_args.no_rmse:
        plot_rmse(active_args=active_args,rmses=rmses1,data_points=data_point1)
    if not active_args.no_rmse2:
        plot_rmse2(active_args=active_args,rmses=rmses21,data_points=data_point1)        
    if not active_args.no_spearman:
        plot_spearman(active_args=active_args,spearmans=spearmans1,data_points=data_point1)
    if not active_args.no_sharpness:
        plot_sharpness(active_args=active_args,sharpness=sharpness1,data_points=data_point1)
    if not active_args.no_sharpness_root:
        plot_sharpness_root(active_args=active_args,sharpness_root=sharpness_root1,data_points=data_point1)
    if not active_args.no_rmse2:
        plot_cv(active_args=active_args,cv=cv1,data_points=data_point1)
    if not active_args.no_miscalibration_area:
        plot_miscalibration_area(active_args=active_args,miscalibration_area=miscalibration_areas1,data_points=data_point1)
    if not active_args.no_rmse2:
        plot_ence(active_args=active_args,ences=ences1,data_points=data_point1)   
    if not active_args.no_subplot:
         sub_plot(active_args=active_args,rmses=rmses1,rmses2=rmses21,spearmans=spearmans1,cv=cv1,data_points=data_point1,sharpness=sharpness1,sharpness_root=sharpness_root1,nlls=nlls1,ences=ences1,miscalibration_areas=miscalibration_areas1,first=first,last=last,true=true)
    if not active_args.no_nll:
         plot_nll(active_args=active_args,nll=nlls1,data_points=data_point1)
    if not active_args.no_true_vs_predicted:
        plot_true_predicted(active_args=active_args,true=true,last=last, first=first)
    


def read_result(active_args: ActiveArgs):
        with open(active_args.evaluation_path, "r") as f:
            spearmans, cv, rmses, rmses2, sharpness = [], [], [], [], []
            nlls, miscalibration_areas, ences, sharpness_root = [], [], [], []
            rmses, rmses2, data_points = [], [], []
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
                    if 'data_points' in reader.fieldnames:
                        data_points.append(float(line['data_points']))

        # with open(active_args.results_path, "r") as f:
        #     reader = csv.reader(f)
        #     # next(reader)
        #     value = []
        #     for row in reader:
        #         if len(row) > 1:  
        #             value.append(row[1])  
        data = pd.read_csv(active_args.results_path)
        true=(data[f"{active_args.value_name}"])
        first=(data[f"{active_args.value_name}_{int(data_points[0])}"])
        last=(data[f"{active_args.value_name}_{int(data_points[-1])}"])
                
        
        return spearmans, nlls, miscalibration_areas, ences, \
        sharpness, sharpness_root, cv, rmses, rmses2, data_points, first, last, true
    
def plot_rmse(active_args, rmses, data_points):
        if len(rmses)==len(data_points):
            plt.scatter(data_points, rmses)
            plt.plot(data_points, rmses, '-',linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("RMSE")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'rmse_plot.png'), dpi=300)
            plt.clf()

def plot_rmse2(active_args, rmses, data_points):
        if len(rmses)==len(data_points):
            plt.scatter(data_points, rmses)
            plt.plot(data_points, rmses, '-',linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("RMSE2")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'rmse2_plot.png'), dpi=300)
            plt.clf()


def plot_spearman(active_args, spearmans, data_points):
        if len(spearmans)==len(data_points):
            plt.scatter(data_points, spearmans)
            plt.plot(data_points, spearmans, '-',linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("Spearman")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'spearman_plot.png'), dpi=300)
            plt.clf()

def plot_sharpness(active_args, sharpness, data_points):
        if len(sharpness)==len(data_points):
            plt.scatter(data_points, sharpness)
            plt.plot(data_points, sharpness, '-', linewidth=1.5, color='g')
            plt.grid(True)
            plt.xlabel("Data Points")
            plt.ylabel("Sharpness")
            plt.savefig(os.path.join(active_args.plot_save_dir,'sharpness_plot.png'), dpi=300)
            plt.clf()


def plot_sharpness_root(active_args, sharpness_root, data_points):
        if len(sharpness_root)==len(data_points):
            plt.scatter(data_points, sharpness_root)
            plt.plot(data_points, sharpness_root, '-', linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("Sharpness_Root")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'sharpness_root_plot.png'), dpi=300)
            plt.clf()

def plot_cv(active_args, cv, data_points):
        if len(cv)==len(data_points):
            plt.scatter(data_points, cv)
            plt.plot(data_points, cv, '-', linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("CV")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'cv_plot.png'), dpi=300)
            plt.clf()


def plot_miscalibration_area(active_args, miscalibration_area, data_points):
        if len(miscalibration_area)==len(data_points):
            plt.scatter(data_points, miscalibration_area)
            plt.plot(data_points, miscalibration_area, '-', linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("Miscalibration_Area")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'miscalibration_area_plot.png'), dpi=300)
            plt.clf()

def plot_ence(active_args, ences, data_points):
        if len(ences)==len(data_points):
            plt.scatter(data_points, ences)
            plt.plot(data_points, ences, '-', linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("ENCE")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'ence_plot.png'), dpi=300)
            plt.clf()



def sub_plot(active_args, rmses, rmses2, cv,data_points,spearmans, nlls, miscalibration_areas, ences, sharpness, sharpness_root, first, last, true):
        x = [0 for _ in range(len(active_args.subplot_x))]
        y=[0 for _ in range(len(active_args.subplot_x))]
        k=0
        for i in range(len(active_args.subplot_x)):
            if 'data_point' in active_args.subplot_x[i]:
                  x[i]=data_points
            elif '1rmse' in active_args.subplot_x[i]:
                  x[i]=rmses
            elif 'spearman' in active_args.subplot_x[i]:
                  x[i]=spearmans
            elif '2rmse' in active_args.subplot_x[i]:
                  x[i]=rmses2  
            elif 'nll' in active_args.subplot_x[i]:
                  x[i]=nlls
            elif 'miscalibration_area' in active_args.subplot_x[i]:
                  x[i]=miscalibration_areas
            elif 'ence' in active_args.subplot_x[i]:
                  x[i]=ences
            elif 'sharpness' in active_args.subplot_x[i]:
                  x[i]=sharpness
            elif 'sharpness_root' in active_args.subplot_x[i]:
                  x[i]=sharpness_root
            elif 'first' in active_args.subplot_x[i]:
                  x[i]=first
            elif 'last' in active_args.subplot_x[i]:
                  x[i]=last
            elif 'true' in active_args.subplot_x[i]:
                  x[i]=true
                             

        for i in range(len(active_args.subplot_y)):
            if 'cv' in active_args.subplot_y[i]:
                  y[i]=cv
            elif '1rmse' in active_args.subplot_y[i]:
                  y[i]=rmses
            elif 'spearman' in active_args.subplot_y[i]:
                  y[i]=spearmans
            elif '2rmse' in active_args.subplot_y[i]:
                  y[i]=rmses2    
            elif 'nll' in active_args.subplot_x[i]:
                  y[i]=nlls
            elif 'miscalibration_area' in active_args.subplot_x[i]:
                  y[i]=miscalibration_areas
            elif 'ence' in active_args.subplot_x[i]:
                  y[i]=ences
            elif 'sharpness' in active_args.subplot_x[i]:
                  y[i]=sharpness
            elif 'sharpness_root' in active_args.subplot_x[i]:
                  y[i]=sharpness_root
            elif 'first' in active_args.subplot_x[i]:
                  y[i]=first
            elif 'last' in active_args.subplot_x[i]:
                  y[i]=last
            elif 'true' in active_args.subplot_x[i]:
                  y[i]=true           

        if len(active_args.subplot_x)==2 or len(active_args.subplot_size)==1:
                fig, axs = plt.subplots(len(active_args.subplot_x))
                fig.suptitle('Vertically stacked subplots')
                for i in range(len(active_args.subplot_x)):
                    axs[i].plot(x[i], y[i], linewidth=1.5, color='g')
                    axs[i].set_xlabel(active_args.subplot_x[i])
                    axs[i].set_ylabel(active_args.subplot_y[i])
                    axs[i].set_title(f'{active_args.subplot_x[i]} vs {active_args.subplot_y[i]}')
                plt.subplots_adjust(hspace=2)
                plt.savefig(os.path.join(active_args.plot_save_dir,'sub_plot.png'), dpi=300)
                plt.clf()
                

  
        else:
            fig, axs = plt.subplots(nrows=int(active_args.subplot_size[0]), ncols=int(active_args.subplot_size[1]))
            fig.suptitle('Vertically stacked subplots')            
            for i in range(int(active_args.subplot_size[0])):
                for j in range(int(active_args.subplot_size[1])):
                    axs[i,j].plot(x[k], y[k], linewidth=1.5, color='g')
                    k+=1
            for i in range(int(active_args.subplot_size[0])):
                for j in range(int(active_args.subplot_size[1])):
                    axs[i,j].set_title(f'{active_args.subplot_x[i]} vs {active_args.subplot_y[i]}')        
            for i in range(int(active_args.subplot_size[0])):
                for j in range(int(active_args.subplot_size[1])):
                    axs[i,j].set_xlabel(active_args.subplot_x[i])
                    axs[i,j].set_ylabel(active_args.subplot_y[i])
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            
            plt.savefig(os.path.join(active_args.plot_save_dir,'sub_plot.png'), dpi=300)
            plt.clf()


def plot_nll(active_args,nll,data_points):
     if len(nll)==len(data_points):
            plt.scatter(data_points, nll)
            plt.plot(data_points, nll, '-', linewidth=1.5, color='g')
            plt.xlabel("Data Points")
            plt.ylabel("NLL")
            plt.grid(True)
            plt.savefig(os.path.join(active_args.plot_save_dir,'nll_plot.png'), dpi=300)
            plt.clf()


def plot_true_predicted(active_args,true,last,first):
     if len(true)==len(first):
            fig, ax = plt.subplots()
            plt.scatter(first, true,color='green')
            plt.xlabel("Predicted Value")
            plt.ylabel("True Value")
            plt.grid(True)
            lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            plt.title('First Step')
            plt.savefig(os.path.join(active_args.plot_save_dir,'true_pred1_plot.png'), dpi=300)
            plt.clf()
     if len(true)==len(last):
            fig, ax = plt.subplots()
            plt.scatter(last, true, color='red')
            plt.xlabel("Predicted Value")
            plt.ylabel("True Value")
            plt.grid(True)
            lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
            ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            plt.title('Last Step')
            plt.savefig(os.path.join(active_args.plot_save_dir,'true_pred2_plot.png'), dpi=300)
            plt.clf()


if __name__ == '__main__':
    plot_results(ActiveArgs().parse_args())
