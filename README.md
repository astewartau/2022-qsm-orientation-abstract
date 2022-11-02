# Automatically Resampling Oblique-Acquired MRI to Enable Robust and Accurate QSM Algorithms

Ashley Stewart<sup>1</sup>, Korbinian Eckstein<sup>1</sup>, Thanh Thuy Dao<sup>1</sup>, Kieran O’Brien<sup>2</sup>, Josef Pfeuffer<sup>2</sup>, Jin Jin<sup>2</sup>, Markus Barth<sup>1</sup>, Steffen Bollmann<sup>1</sup>

<sup>1</sup>School of Information Technology and Electrical Engineering, The University of Queensland, Brisbane, Australia; 

<sup>2</sup>Siemens Healthineers Pty Ltd., Brisbane, Australia

## Introduction

In an MRI acquisition, oblique slice orientations are often used to visualise clinically relevant anatomical regions and optimise the scanning field of view and duration. In Quantitative Susceptibility Mapping (QSM), which aims to estimate the magnetic susceptibility of tissues, many emerging algorithms assume that the main magnetic field is oriented axially through the imaging volume, which is not the case for oblique acquisitions. This assumption prevents accurate susceptibility quantification and introduces pervasive artefacts in susceptibility maps, limiting the practical uptake of the method. Recent work has identified resampling slices to an axial orientation prior to QSM processing as a solution<sup>1</sup>. In this work, we implement this rotate-and-resample method to enable automated processing of oblique acquisitions and publish it as part of the QSMxT software toolbox<sup>2,3</sup>. We test the implementation in silico using the QSM challenge head phantom, measuring differences in the effect of orientation on susceptibility quantification across brain regions, identifying that small sources such as veins remain challenging to quantify accurately after resampling.


```python
# Necessary imports
import os
import glob
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import abstract_functions as fn
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
```

## Methods

### Data

The QSM challenge 2.0 *in silico* head phantom<sup>4</sup> was used for this experiment and is based on a segmented MP2RAGE dataset with 0.64 mm<sup>3</sup> isotropic resolution down-sampled to 1 mm<sup>3</sup> using k-space cropping and TEs=4/12/20/28 ms. The magnitude and phase were augmented by rotation about the $x$-axis to produce new datasets with effective B<sub>0</sub> directions at 5° increments up to 45°.


```python
# Download input data
bids_dir = fn.get_bids_data()
print(f"Downloaded in silico head phantom data to {bids_dir}.")
```

    Downloaded in silico head phantom data to /tmp/bids-osf.



```python
# Acquisition and reconstruction parameters
echo_times = [0.004, 0.012, 0.020, 0.028]
magnetic_field_strength = 3
tgvqsm_erosions = 3
bet_fractional_intensity = 0.5
```


```python
# Get relevant paths and find input images
session_path = os.path.join(bids_dir, "sub-1", "ses-1")
anat_path = os.path.join(session_path, "anat")
extra_data_path = os.path.join(session_path, "extra_data")
chi_file = os.path.join(extra_data_path, "sub-1_ses-1_run-01_chi-cropped.nii.gz")
segmentation_file = os.path.join(extra_data_path, "sub-1_ses-1_run-01_segmentation.nii.gz")
mag_files = sorted(glob.glob(os.path.join(anat_path, "sub-1_ses-1_run-01_echo-*_part-mag_MEGRE.nii.gz")))
pha_files = sorted(glob.glob(os.path.join(anat_path, "sub-1_ses-1_run-01_echo-*_part-phase_MEGRE.nii.gz")))
assert(len(mag_files) == len(pha_files))
tmp_dir = tempfile.gettempdir()
print(f"Found {len(mag_files)} magnitude and phase image pairs.")
```

    Found 4 magnitude and phase image pairs.



```python
# View some input data
fn.display_nii(mag_files[0], title=f'Magnitude (TE={echo_times[0]})', cmap='gray', interpolation='nearest')
fn.display_nii(pha_files[0], title=f'Phase (TE={echo_times[0]})', cmap='gray', interpolation='nearest')
```


    
![png](abstract-code_files/abstract-code_7_0.png)
    



    
![png](abstract-code_files/abstract-code_7_1.png)
    



```python
# Create augmented (rotated) data
degrees = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
for d in degrees:
    print(f"Creating rotated input images ({d}°)")
    out_dir = os.path.join(tmp_dir, f"inputs-rotated-{d}-degrees")
    mag_rotated, pha_rotated = fn.rotate_mag_phase(mag_files, pha_files, out_dir, d)
```

    Creating rotated input images (1°)
    Creating rotated input images (2°)
    Creating rotated input images (3°)
    Creating rotated input images (4°)
    Creating rotated input images (5°)
    Creating rotated input images (10°)
    Creating rotated input images (15°)
    Creating rotated input images (20°)
    Creating rotated input images (25°)
    Creating rotated input images (30°)
    Creating rotated input images (35°)
    Creating rotated input images (40°)
    Creating rotated input images (45°)



```python
# View some of the augmented data
fn.display_nii(mag_rotated[0], title=f'Magnitude (TE={echo_times[0]}; rotated {degrees[-1]}°)', cmap='gray', interpolation='nearest')
fn.display_nii(pha_rotated[0], title=f'Phase (TE={echo_times[0]}; rotated {degrees[-1]}°)', cmap='gray', interpolation='nearest')
```


    
![png](abstract-code_files/abstract-code_9_0.png)
    



    
![png](abstract-code_files/abstract-code_9_1.png)
    


### Processing

Magnitude and phase data were resampled to an axial affine through the complex domain using nilearn prior to QSM processing using TGV-QSM5. The susceptibility maps were then resampled back to the original orientation. In QSMxT, this process occurs automatically when the obliquity, measured using nibabel, is greater than a user-defined threshold.


```python
# First process the original (0-degrees rotated) data
print("BET masking for original data...")
mask_file = fn.bet_masking(mag_files[0], bet_fractional_intensity, extra_data_path)
print("QSM reconstruction for original data...")
qsm_average_file = fn.tgv_qsm_me(mask_file, pha_files, echo_times, magnetic_field_strength, tgvqsm_erosions, extra_data_path)
fn.display_nii(mask_file, title='Mask (original)', cmap='gray', interpolation='nearest')
fn.display_nii(qsm_average_file, title='QSM (original)', cmap='gray', vmin=-0.05, vmax=+0.05, interpolation='nearest')
```

    BET masking for original data...
    QSM reconstruction for original data...



    
![png](abstract-code_files/abstract-code_11_1.png)
    



    
![png](abstract-code_files/abstract-code_11_2.png)
    



```python
# Now process the augmented/rotated data
result_files = { }
for d in degrees:
    out_dir = os.path.join(extra_data_path, f"inputs-rotated-{d}-degrees")
    mag_rotated, pha_rotated = fn.rotate_mag_phase(mag_files, pha_files, out_dir, d)
    mag_rotated_axial, pha_rotated_axial = fn.resample_to_axial(mag_rotated, pha_rotated, out_dir)

    print(f"Reconstructing QSM for {d}° rotated images (no corrections / 'NoRot')...")
    mask_rotated = fn.bet_masking(mag_rotated[0], bet_fractional_intensity, out_dir)
    qsm_rotated_average = fn.tgv_qsm_me(mask_rotated, pha_rotated, echo_times, magnetic_field_strength, tgvqsm_erosions, out_dir)
    mask_rotated_resampled_file = fn.resample_like(mask_rotated, mag_files[0], interpolation='nearest')
    qsm_rotated_average_resampled_file = fn.resample_like(qsm_rotated_average, mag_files[0])
    print(f"Reconstructing QSM for {d}° backsampled images (with corrections / 'RotPrior')...")
    mask_rotated_axial = fn.bet_masking(mag_rotated_axial[0], bet_fractional_intensity, out_dir)
    qsm_rotated_axial_average = fn.tgv_qsm_me(mask_rotated_axial, pha_rotated_axial, echo_times, magnetic_field_strength, tgvqsm_erosions, out_dir)
    mask_rotated_axial_resampled_file = fn.resample_like(mask_rotated_axial, mag_files[0], out_dir, interpolation='nearest')
    qsm_rotated_axial_average_resampled_file = fn.resample_like(qsm_rotated_axial_average, mag_files[0])
    
    result_files[d] = [qsm_rotated_axial_average_resampled_file, qsm_rotated_average_resampled_file]
```

    Reconstructing QSM for 1° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 1° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 2° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 2° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 3° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 3° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 4° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 4° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 5° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 5° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 10° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 10° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 15° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 15° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 20° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 20° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 25° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 25° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 30° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 30° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 35° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 35° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 40° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 40° backsampled images (with corrections / 'RotPrior')...
    Reconstructing QSM for 45° rotated images (no corrections / 'NoRot')...
    Reconstructing QSM for 45° backsampled images (with corrections / 'RotPrior')...



```python
fn.display_nii(mask_rotated_resampled_file, title=f'Mask (rotated {degrees[-1]}°; processed with no corrections / NoRot)', cmap='gray', interpolation='nearest', slc=98)
fn.display_nii(qsm_rotated_average_resampled_file, title=f'QSM (rotated {degrees[-1]}°; processed with no corrections / NoRot)', cmap='gray', vmin=-0.05, vmax=+0.05, interpolation='nearest', slc=98)
```


    
![png](abstract-code_files/abstract-code_13_0.png)
    



    
![png](abstract-code_files/abstract-code_13_1.png)
    



```python
fn.display_nii(mask_rotated_axial_resampled_file, title=f'Mask (rotated {degrees[-1]}°; processed with corrections / RotPrior)', cmap='gray', interpolation='nearest', slc=98)
fn.display_nii(qsm_rotated_axial_average_resampled_file, title=f'QSM (rotated {degrees[-1]}°; processed with corrections / RotPrior)', cmap='gray', vmin=-0.05, vmax=+0.05, interpolation='nearest', slc=98)
```


    
![png](abstract-code_files/abstract-code_14_0.png)
    



    
![png](abstract-code_files/abstract-code_14_1.png)
    


### Measurements

RMSE was measured across brain regions defined by segmentations provided with the original head phantom.


```python
# Segmentation labels
labels = { 
    1 : "Caudate",
    2 : "Globus pallidus",
    3 : "Putamen",
    4 : "Red nucleus",
    5 : "Dentate nucleus",
    6 : "SN and STN",
    7 : "Thalamus",
    8 : "White matter",
    9 : "Gray matter",
    10 : "CSF",
    11 : "Blood",
    12 : "Fat",
    13 : "Bone",
    14 : "Air",
    15 : "Muscle",
    16 : "Calcification"
}

# Load ground truth data
segmentation = np.rot90(nib.load(segmentation_file).get_fdata())
chi = np.rot90(nib.load(chi_file).get_fdata())
qsm_average = np.rot90(nib.load(qsm_average_file).get_fdata())
chi = chi * (qsm_average != 0)
mask = qsm_average != 0

# Calculate RMSE
print("Calculating error metrics...")
columns = ["Rotation", "Segmentation", "Corrected", "RMSE"]
rmse = np.array([0, 0, 0, mse(chi[mask], qsm_average[mask], squared=False)]).reshape(1,4)
rmse = np.append(rmse, np.array([0, 0, 1, mse(chi[mask], qsm_average[mask], squared=False)]).reshape(1,4), axis=0)
for label_id in labels.keys():
    if np.any(chi[segmentation == label_id]):
        rmse = np.append(rmse, np.array([0, label_id, 1, mse(chi[segmentation == label_id], qsm_average[segmentation == label_id], squared=False)]).reshape(1,4), axis=0)
        rmse = np.append(rmse, np.array([0, label_id, 0, mse(chi[segmentation == label_id], qsm_average[segmentation == label_id], squared=False)]).reshape(1,4), axis=0)
for d in degrees:
    print(f"Calculating error metrics for d={d}")
    qsm_corrected = np.rot90(nib.load(result_files[d][0]).get_fdata())
    qsm_uncorrected = np.rot90(nib.load(result_files[d][1]).get_fdata())
    rmse = np.append(rmse, np.array([d, 0, 1, mse(chi[mask], qsm_corrected[mask], squared=False)]).reshape(1,4), axis=0)
    rmse = np.append(rmse, np.array([d, 0, 0, mse(chi[mask], qsm_uncorrected[mask], squared=False)]).reshape(1,4), axis=0)
    for label_id in labels.keys():
        if np.any(chi[segmentation == label_id]):
            rmse = np.append(rmse, np.array([d, label_id, 1, mse(chi[segmentation == label_id], qsm_corrected[segmentation == label_id], squared=False)]).reshape(1,4), axis=0)
            rmse = np.append(rmse, np.array([d, label_id, 0, mse(chi[segmentation == label_id], qsm_uncorrected[segmentation == label_id], squared=False)]).reshape(1,4), axis=0)
rmse = pd.DataFrame(data=rmse, columns=columns)
rmse["Corrected"] = rmse["Corrected"].replace(0, "NoRot").replace(1, "RotPrior")
```

    Calculating error metrics...
    Calculating error metrics for d=1
    Calculating error metrics for d=2
    Calculating error metrics for d=3
    Calculating error metrics for d=4
    Calculating error metrics for d=5
    Calculating error metrics for d=10
    Calculating error metrics for d=15
    Calculating error metrics for d=20
    Calculating error metrics for d=25
    Calculating error metrics for d=30
    Calculating error metrics for d=35
    Calculating error metrics for d=40
    Calculating error metrics for d=45


## Results

RMSE measured in the head phantom indicates that the rotate-and-resample method (RotPrior) provides stable RMSEs across brain regions except in blood and CSF (see Figure 1). Conversely, no corrections (NoRot) cause increasing RMSE with the degree of rotation, particularly in deep gray matter (dGM) nuclei.


```python
print("Creating line plot...")
sns.set_style("darkgrid")
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":10,"axes.labelsize":10})
fig, axes = plt.subplots(3,4, figsize=(12,8))
fig.suptitle('RMSE with rotation, with and without rotate-and-resample using TGV-QSM')
fig.supxlabel('Rotation (degrees)')
fig.supylabel('RMSE (ppm)')
index = 0
for label_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    ylim=(0.0, 0.15)
    xlim=(0, 45)
    ax = axes[int(index / 4), index % 4]
    plot = sns.lineplot(ax=ax, data=rmse[rmse.Segmentation == label_id], x="Rotation", y="RMSE", hue="Corrected")
    plot.set(xlabel=None)
    plot.set(ylabel=None)
    plot.set(ylim=ylim, xlim=xlim, title=labels[label_id] if label_id in labels else "Whole brain")
    if index < 8:
        plot.set(xticklabels=[])
    if index % 4 != 0:
        plot.set(yticklabels=[])
    plot.tick_params(axis='both', which='major', labelsize=8)
    plot.tick_params(axis='both', which='minor', labelsize=8)
    plt.xlabel("")
    plot.get_legend().remove()
    index += 1
handles, labels_handle = plot.get_legend_handles_labels()
plot.legend(handles, labels_handle, loc=(1.2, 3.025))
#plt.savefig(f"lineplot.png", dpi=700, bbox_inches='tight')
plt.show()
plt.close()
```

    Creating line plot...



    
![png](abstract-code_files/abstract-code_18_1.png)
    


Difference imaging visualises the spatial distribution of changes in susceptibility quantification and smearing artefacts in dGM regions as the degree of rotation increases (see Figure 2). 


```python
print("Creating diff image figure...")
#plt.figure()
sns.set_style("darkgrid")
sns.set_context("paper", rc={"font.size":13,"axes.titlesize":13,"axes.labelsize":13})
fig, axes = plt.subplots(4, 5, figsize=[10,10])
fig.suptitle('QSM computed using TGV-QSM using simulated data rotated to various angles')
slc = 102
vmin = -0.1
vmax = +0.1

axes[0,0].set_ylabel("QSM at θ", rotation=0, horizontalalignment='right')
axes[0,0].set_title("0°")
axes[0,0].imshow(qsm_average[:,:,slc], cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0,1].set_title("5°")
axes[0,1].imshow(np.rot90(nib.load(result_files[5][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0,2].set_title("10°")
axes[0,2].imshow(np.rot90(nib.load(result_files[10][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0,3].set_title("15°")
axes[0,3].imshow(np.rot90(nib.load(result_files[15][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0,4].set_title("20°")
axes[0,4].imshow(np.rot90(nib.load(result_files[20][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

axes[1,0].set_ylabel("Diff vs. 0°", rotation=0, horizontalalignment='right')
axes[1,0].imshow(qsm_average[:,:,slc] - qsm_average[:,:,slc], cmap='seismic', vmin=vmin, vmax=vmax)
axes[1,1].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[5][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[1,2].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[10][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[1,3].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[15][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[1,4].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[20][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')

axes[2,0].set_title("25°")
axes[2,0].set_ylabel("QSM at θ", rotation=0, horizontalalignment='right')
axes[2,0].imshow(np.rot90(nib.load(result_files[25][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[2,1].set_title("30°")
axes[2,1].imshow(np.rot90(nib.load(result_files[30][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[2,2].set_title("35°")
axes[2,2].imshow(np.rot90(nib.load(result_files[35][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[2,3].set_title("40°")
axes[2,3].imshow(np.rot90(nib.load(result_files[40][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[2,4].set_title("45°")
im1 = axes[2,4].imshow(np.rot90(nib.load(result_files[45][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

axes[3,0].set_ylabel("Diff vs. 0°", rotation=0, horizontalalignment='right')
axes[3,0].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[25][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[3,1].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[30][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[3,2].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[35][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[3,3].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[40][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
im2 = axes[3,4].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[45][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')

for row in axes:
    for ax in row:
        ax.tick_params(which='both', left=False, right=False, labelbottom=False, labelleft=False, grid_alpha=0)

cax1 = fig.add_axes([0.265,0.050,0.5,0.025])
cax2 = fig.add_axes([0.265,0.075,0.5,0.025])
cax1.set_frame_on(True)
cax2.set_frame_on(True)
cbar1 = fig.colorbar(im1, orientation='horizontal', cax=cax1)
cbar2 = fig.colorbar(im2, orientation='horizontal', cax=cax2)
cbar2.set_ticks([])
cbar1.ax.locator_params(nbins=3)
cbar2.ax.locator_params(nbins=3)
cbar1.ax.tick_params(labelsize=10)
cbar2.ax.tick_params(labelsize=10)

#plt.savefig(f"diff-images.png", dpi=700, bbox_inches='tight')
plt.show()
plt.close()
```

    Creating diff image figure...



    
![png](abstract-code_files/abstract-code_20_1.png)
    


## Discussion and conclusions

Our experiment validates that the rotate-and-resample method1 results in stable susceptibility quantification across most brain regions. Further, small, high-susceptibility sources such as blood in veins were found to be more challenging to recover after resampling, which may be explained by reduced phase accuracy after interpolation in the complex domain. Our publicly available implementation of the method in QSMxT makes it available and practical for users wishing to reconstruct QSM for acquisitions with oblique slice orientations.

## References

1] Kiersnowski et al. ISMRM 2021, Abstract 794. 2] Stewart et al., Mag. Reson. Med., 2022. 3] Stewart et al., QSMxT, GitHub, 2022. 4] Marques et al., Mag. Reson. Med., 2021. 6] Langkammer et al., NeuroImage, 2015.
