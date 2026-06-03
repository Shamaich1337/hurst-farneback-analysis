from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import cv2 as cv


def build_pyr(field, n_jobs):
    def worker(frame):
        dstsize=(frame.shape[0]//2, frame.shape[1]//2)
        return cv.pyrDown(frame, dstsize=dstsize)
    
    length = field.shape[-1]
    tasks = [field[:, :, i] for i in range(0, length)]

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(worker)(task) for task in tqdm(tasks, desc=f'build pyr', unit='frames'))
    
    return np.stack(results, axis=-1)


def farneback_optical_flow(field,
                           n_jobs = 4,
                           winsize = 15,
                           iterations = 5,
                           poly_n = 5,
                           poly_sigma = 1.1):
    
    length = field.shape[-1]
    
    def worker(prev_frame, curr_frame):
        
        flow = cv.calcOpticalFlowFarneback(prev=prev_frame,
                                           next=curr_frame,
                                           flow=None,
                                           pyr_scale=0.5,
                                           levels=1,
                                           winsize=winsize,
                                           iterations=iterations,
                                           poly_n=poly_n,
                                           poly_sigma=poly_sigma,
                                           flags=0)
        
        mag, _ = cv.cartToPolar(flow[..., 0], flow[..., 1])
        return mag

    
    tasks = [(field[:, :, i], field[:, :, i + 1]) for i in range(length - 1)]

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(worker)(prev, curr) for prev, curr in tqdm(tasks, desc='Farneback', unit='pairs')
    )

    return np.stack(results, axis=-1)
