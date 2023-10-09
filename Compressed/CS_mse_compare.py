import nibabel as nib
from sklearn.metrics import mean_squared_error
import numpy as np
import os
from matplotlib.pyplot import imshow

CS_folder = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/'

CS_orig = '/Volumes/Data/Badea/Lab/mouse/CS_Recon_Optimization/Bruker_diffusion_test_15.nii.gz'

TV_vals = [0.0025, 0.005, 0.01, 0.025, 0.05]
L1_vals = [0.0025, 0.005, 0.01, 0.025, 0.05]

orig_nii = nib.load(CS_orig)
orig_data = orig_nii.get_fdata()

orig_data = orig_data[orig_data[:,:,:,0]>0, :]
mask = orig_data[:,0]>30000
rmse_dir = {}

rmse_min_vals = {}

CS_vals = ['4','5.9998','8','10.0009','11.9985']

CS_val = CS_vals[4]
vol = 1

rmse_min = 10000000000000000000000
orig_vol_data = orig_data[:,vol]

for TV_val in TV_vals:
    for L1_val in L1_vals:
        path = os.path.join(CS_folder, f'Bruker_diffusion_test_15_0.045_af_{CS_val}x__TV_and_L1_wavelet_{TV_val}_{L1_val}_bart_recon.nii.gz')
        CS_nii = nib.load(path)
        CS_data = CS_nii.get_fdata()
        CS_data = np.ndarray.flatten(CS_data[:,:,:,vol])
        #imshow(np.reshape(orig_vol_data,np.shape(CS_data))[:,:,50])
        #imshow(CS_data[:,:,50])
        #imshow(CS_data[:,:,50] - np.reshape(orig_vol_data,np.shape(CS_data))[:,:,50])

        rmse_result = mean_squared_error(orig_vol_data[mask], CS_data[mask])
        rmse_dir[int(np.round(float(CS_val))), TV_val, L1_val] = rmse_result
        print(f'Result for CS sampling level at {CS_val}, TV val at {TV_val} and L1_val at {L1_val} is {rmse_result}')
        if rmse_result<rmse_min:
            rmse_min_vals[int(np.round(float(CS_val)))] = [TV_val, L1_val, rmse_result]
            rmse_min = rmse_result


print(f'Minimum vals for CS_val {CS_val} are TV_val = {rmse_min_vals[int(np.round(float(CS_val)))][0]} and L1_val = {rmse_min_vals[int(np.round(float(CS_val)))][1]}')