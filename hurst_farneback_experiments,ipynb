# import
import numpy as np
import pandas as pd
import cv2 as cv
import seaborn as sns
from StatTools.generators.ndfnoise_generator import ndfnoise
from StatTools.visualization import plot_ff
from StatTools.analysis.dfa import DFA, dfa, dfa_worker
from StatTools.analysis import(
    bma,
    f_fcn_without_overflaw,
    f_fcn,
    rev_f_fcn,
    tf_minus_inf,
    tf_plus_inf,
    ff_base_appriximation,
    cross_fcn_sloped,
     ff_params,
     var_estimation,
    analyse_cross_ff,
    analyse_zero_cross_ff

) 

from joblib import Parallel, delayed, cpu_count
import plotly.express as px
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from typing import Optional

# git clone https://github.com/Digiratory/FluctuationAnalysisTools.git
# cd FluctuationAnalysisTools
# mv StatTools/ ../
# defs
def frames_generator(field: np.ndarray):
    frames_num = field.shape[2]
    for i in range(frames_num):
        yield field[:,:,i]
def build_pyr(frame_list: list, levels:int):
    
    if levels==0:
        return frame_list
    else:

        scaled = cv.resize(src=frame_list[-1],
                           dsize=None,
                           dst=None,
                           fx=0.5,
                           fy=0.5,
                           interpolation=cv.INTER_LINEAR)
        frame_list.append(scaled)
        
        return build_pyr(frame_list, levels-1)
def farneback_optical_flows(field:np.ndarray,
                            pyr_level:int,
                            winsize: int = 15,
                            iterations: int = 5,
                            poly_n: int = 5,
                            poly_sigma: float = 1.1):
    
    frames_gen = frames_generator(field)

    ang_result      = [[] for _ in range(pyr_level+1)]
    mag_result      = [[] for _ in range(pyr_level+1)]
    x_shift_result  = [[] for _ in range(pyr_level+1)]
    y_shift_result  = [[] for _ in range(pyr_level+1)]

    try:
        prev_frame = next(frames_gen)
    except StopIteration:
        return None
    
    prev_pyr = build_pyr([prev_frame], pyr_level)

    for curr_frame in tqdm(iterable=frames_gen, total=field.shape[2]-1, desc='Field frames processing', unit='frames'):
        curr_pyr = build_pyr([curr_frame], pyr_level)
        
        for level in range(pyr_level+1):
            flow = cv.calcOpticalFlowFarneback(prev=prev_pyr[level],
                                               next=curr_pyr[level],
                                               flow=None,
                                               pyr_scale=0.5,
                                               levels=1,
                                               winsize=winsize,
                                               iterations=iterations,
                                               poly_n=poly_n,
                                               poly_sigma=poly_sigma,
                                               flags=0)
            
            # levels=1 means that no extra layers are created and only the original images are used.

            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            x_shift = flow[..., 0]
            y_shift = flow[..., 1]

            ang_result[level].append(ang)
            mag_result[level].append(mag)
            x_shift_result[level].append(x_shift)
            y_shift_result[level].append(y_shift)
        
        prev_pyr = curr_pyr
    
    ang_result      = [np.stack(level, axis=-1) for level in ang_result]
    mag_result      = [np.stack(level, axis=-1) for level in mag_result]
    x_shift_result  = [np.stack(level, axis=-1) for level in x_shift_result]
    y_shift_result  = [np.stack(level, axis=-1) for level in y_shift_result]

    return ang_result, mag_result, x_shift_result, y_shift_result

# generate field 
# задаемся параметрами поля
frame_num = 1000
frame_shape = (128, 128)
# получаем поле с заданным показателем херста
np.random.seed(42)
field = ndfnoise(shape=(*frame_shape, frame_num), hurst=0.8, normalize=True)
field = np.diff(field)
pyr_level = 3
# calc optical flow
ang_result, mag_result, x_shift_result, y_shift_result = farneback_optical_flows(field=field, pyr_level=pyr_level)
# dfa calc
n_jobs = cpu_count(only_physical_cores=True)
def worker(series):
    _, F2_s = dfa(dataset=series,
                  degree=2,
                  processes=1,
                  n_integral=1)
    return F2_s

F2_s_field_result = []

s, _ = dfa(dataset=x_shift_result[0][0, 0, :],
            degree=2,
            processes=1,
            n_integral=1)

for level in range(pyr_level + 1):
    data_lvl = x_shift_result[level]
    h, w = data_lvl.shape[:2]
    
    tasks = [data_lvl[i, j, :] for i in range(h) for j in range(w)]
    n_tasks = len(tasks)
    
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(worker)(task) for task in tqdm(tasks, desc=f'Level {level}', unit='series')
    )
    
    n_scales = len(results[0])
    F2_s_lvl = np.array(results).reshape(h, w, n_scales)
    F2_s_field_result.append(F2_s_lvl)
# LLS on ff medians
fig, axes = plt.subplots(nrows=len(F2_s_field_result)//2, ncols=2, figsize=(15, 12))
axes = np.atleast_2d(axes).flatten()  

for level, ax in enumerate(axes):
    if level >= len(F2_s_field_result):
        ax.axis('off')
        continue
        
    data_lvl = F2_s_field_result[level].reshape(-1, len(s)) 
    
    medians = np.median(data_lvl, axis=0)
     
    log_s = np.log(s)
    
    log_med = np.log(medians)
    
    linreg_med = stats.linregress(log_s, log_med)
    
    alpha_med = linreg_med.slope
   
    ax.boxplot(data_lvl, positions=s, patch_artist=True, showfliers=False, widths=s/5)
    
    f2_fit = np.exp(linreg_med.slope * np.log(s) + linreg_med.intercept)
    ax.plot(s, f2_fit, 'r-', linewidth=2, label=f'\u03b1 = {alpha_med:.2f}')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('s')
    ax.set_ylabel('F²(s)')
    ax.set_title(f'Level {level}')
    ax.legend(fontsize=10)
    ax.grid(which='both')
    


plt.show()
# ff crossover analysis
fig, axes = plt.subplots(nrows=len(F2_s_field_result)//2, ncols=2, figsize=(15, 15))
axes = np.atleast_2d(axes).flatten()
for level, ax in enumerate(axes):

    data_lvl = F2_s_field_result[level].reshape(-1, len(s))
    ff_parameters, _ = analyse_cross_ff(data_lvl, s, crossover_amount=1, min_slope_current=0, max_slope_current=3)

    plot_ff(
        data_lvl,
        s,
        ff_parameter=ff_parameters,
        residuals=None,
        ax=ax,
        title=f'Level {level}'
    )
# ff medians crossover analysis
fig, axes = plt.subplots(nrows=len(F2_s_field_result)//2, ncols=2, figsize=(15, 15))
axes = np.atleast_2d(axes).flatten()
for level, ax in enumerate(axes):

    data_lvl = F2_s_field_result[level].reshape(-1, len(s))
    medians = np.median(data_lvl, axis=0).reshape(-1, len(s))
    ff_parameters, _ = analyse_cross_ff(medians, s, crossover_amount=1, min_slope_current=0, max_slope_current=10)

    plot_ff(
        medians,
        s,
        ff_parameter=ff_parameters,
        residuals=None,
        ax=ax,
        title=f'Level {level}'
    )
# zero-cross ff
fig, axes = plt.subplots(nrows=len(F2_s_field_result)//2, ncols=2, figsize=(15, 15))
axes = np.atleast_2d(axes).flatten()
for level, ax in enumerate(axes):

    data_lvl = F2_s_field_result[level].reshape(-1, len(s))
    ff_parameters, _ = analyse_zero_cross_ff(data_lvl, s)

    plot_ff(
        data_lvl,
        s,
        ff_parameter=ff_parameters,
        residuals=None,
        ax=ax,
        title=f'Level {level}'
    )
# field anime
target = x_shift_result[2]
anime = px.imshow(target, color_continuous_scale ='inferno', animation_frame=2, width=500, height=500, zmin=target.min(), zmax=target.max())
anime.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 0
anime.show()
