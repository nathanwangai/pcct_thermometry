import os
import mat73

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

""" ----- SPECTRAL SCAN CLASS ----- """ 
class SpectralScan():
    MAG = 1.23 # ratio
    INNER_DIM = 2.54 # cm
    PIXEL_DIM = 0.0055 # cm/pixel
    energies = ["8-33 kvp", "33-45 kvp", "45-60 kvp", "60-110 kvp"]
 
    def __init__(self, open_projs, empty_projs, ROI, verbose=False):
        self.r0 = ROI[0]
        self.c0 = ROI[1]
        self.r1 = ROI[2]
        self.c1 = ROI[3]
 
        self.open_projs = self.avg_projs(open_projs, verbose)
        self.empty_projs = self.avg_projs(empty_projs, verbose)
        self.filled_projs = None
 
    # change filled_projs attribute
    def set_filled_projs(self, filled_projs, verbose=False):
        self.filled_projs = self.avg_projs(filled_projs, verbose)

    # change ROI attribute
    def set_ROI(self, ROI): 
        self.r0 = ROI[0]
        self.c0 = ROI[1]
        self.r1 = ROI[2]
        self.c1 = ROI[3]
 
    """ ----- HELPER METHODS ----- """
    # sets pixels > 3 sd from the mean to NaN
    @staticmethod
    def rm_dead_pix(proj):
        mean, stdev = np.mean(proj), np.std(proj)  
        dead_pixels = np.abs((proj-mean)/stdev) > 3
        proj = proj.astype("float")
        proj[dead_pixels] = np.NaN
        return proj
   
    # uses linear interpolation to fill in NaN values
    @staticmethod
    def interp_nan(proj):
        mask = np.isnan(proj)
        proj[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), proj[~mask])
        return proj
 
    # performs sliding average and median filtering on a 1D array
    @staticmethod
    def filter_profile(profile, extra=True):
        profile[profile < 0] = 0
        profile = SpectralScan.interp_nan(profile)
        profile = medfilt(profile, kernel_size=9)
        profile = np.convolve(np.ones(5)/5, profile, mode="valid")
 
        return profile
 
    # truncates projection to the ROI
    def get_roi(self, proj):
        return proj[self.r0:self.r1, self.c0:self.c1]
    """ ----- END OF HELPER METHODS -----"""
 
    """ ----- MAIN METHODS ----- """
    # stores projections averaged over the ROI for each of the 4 energy bins
    def avg_projs(self, projs, verbose=False):
        width = self.r1 - self.r0
        height = self.c1 - self.c0
        averaged_projs = [np.zeros((width, height))] * 4
        proj_counts = [0] * 4
        thresholds = [2000, 800, 500, 200]
 
        num_projs = projs.shape[0]
        for i in range(num_projs):
            proj = projs[i]
            id_mean = np.mean(proj[100:200, 100:200])
            proj = SpectralScan.rm_dead_pix(self.get_roi(proj))
 
            # add projection to others at the same energy
            for k in range(4):
                if(id_mean > thresholds[k]):
                    # NaN pixel in one projection is NaN in all projections (NaN + anything = NaN)
                    averaged_projs[k] = averaged_projs[k] + proj
                    proj_counts[k] += 1
                    break
 
        # compute averages
        for i in range(4):
            averaged_projs[i] = averaged_projs[i] / proj_counts[i]
 
        # compute bins
        for i in range(3):
            averaged_projs[i] = averaged_projs[i] - averaged_projs[i+1]
       
        if verbose:
            for i in range(4):
                proj = averaged_projs[i]
                print(proj_counts)
                print(f"Average intensity in ROI of {SpectralScan.energies[i]} bin: {str(np.nanmean(proj))}")
            print("")
 
        return averaged_projs # contains NaN values

    # calculates the attenuation in each energy bin
    def cmp_spectral_mus(self, verbose=False, method="weak_perspective"):
        attenuations = []
 
        for i in range(4):
            empty_profile = np.log(np.divide(self.open_projs[i], self.empty_projs[i]))
            filled_profile = np.log(np.divide(self.open_projs[i], self.filled_projs[i]))
            empty_profile[empty_profile==np.inf] = np.NaN
            filled_profile[filled_profile==np.inf] = np.NaN
           
            if(method == "weak_perspective"):
                empty_profile = SpectralScan.filter_profile(empty_profile.flatten())
                filled_profile = SpectralScan.filter_profile(filled_profile.flatten())
                mu = np.sum(filled_profile - empty_profile) * SpectralScan.PIXEL_DIM / (SpectralScan.INNER_DIM**2 * SpectralScan.MAG)
            elif(method == "line_integral"):
                empty_profile = SpectralScan.interp_nan(empty_profile)
                filled_profile = SpectralScan.interp_nan(filled_profile)
                mu = np.nanmean(filled_profile - empty_profile) / SpectralScan.INNER_DIM      
            attenuations.append(mu)
       
            if verbose:
                plt.title(f"{mu:.4f} cm^-1 in {str(SpectralScan.energies[i])} bin")
                if(method == "weak_perspective"):
                    plt.plot(empty_profile, label="empty")
                    plt.plot(filled_profile, label="filled")
                    plt.legend()
                    plt.show()
                elif(method == "line_integral"):
                    plt.hist(empty_profile.flatten(), bins=100, label="empty")
                    plt.hist(filled_profile.flatten(), bins=100, label="filled")
                    plt.xlabel("mu * x")
                    plt.ylabel("frequency")
                    plt.legend()
                    plt.show()
 
        return attenuations
    """ ----- END OF MAIN METHODS -----"""
""" ----- END OF SPECTRAL SCAN CLASS -----"""

"""
load and save attenuation data
"""
# read in raw .mat file as numpy array
def load_proj_mat(fpath):
    return mat73.loadmat(fpath)["picdata"]

def read_csv(fname):
    return pd.read_csv(f"projection_mat\{fname}.csv", header=None).to_numpy()

# returns temperatures (array), mean attenuation (matrix), and standard deviation of attenuation (matrix)
# matrix dimensions: temperature (row), energy (column)
def get_mu_mtx(fpath, scanner):
    projs = [file for file in os.listdir(fpath) if file.startswith("T")]
    temps = [int(proj[1:3]) for proj in projs]
    mu_mtx = []
    mu_std_mtx = []
 
    for proj in projs:
        proj = load_proj_mat(os.path.join(fpath, proj))
        trials_arr = []

        # compute attenuation over several ROIs to find mean and stdev
        for i in range(10):
            scanner.set_ROI([195+i, 200, 196+i, 1200])
            scanner.set_filled_projs(proj)
            trials_arr.append(scanner.cmp_spectral_mus(method="weak_perspective"))
        
        trials_arr = np.array(trials_arr)
        # mean and std of 10 attenuations computed from 10 LOI's
        mu_mtx.append(np.mean(trials_arr, axis=0))
        mu_std_mtx.append(np.std(trials_arr, axis=0))
            
    return temps, np.array(mu_mtx), np.array(mu_std_mtx)