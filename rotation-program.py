import os
import nibabel as nib
import numpy as np
import nilearn.image
import glob
import warnings
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

def get_extension(filename):
    return ".".join(filename.split(".")[1:])

def get_fname(filename):
    return os.path.split(filename)[1].split(".")[0]

def get_dir(filename):
    return os.path.split(filename)[0]

def fname_append(filename, new_part):
    dir_part = get_dir(filename)
    extension = get_extension(filename)
    fname = get_fname(filename)
    return f"{dir_part}/{fname}{new_part}.{extension}"

def get_base_affine(nii):
    # calculate base affine
    voxel_size = np.array(nii.header.get_zooms())
    resolution = np.array(nii.header.get_data_shape())
    base_affine = np.eye(4)
    np.fill_diagonal(base_affine, voxel_size * np.sign(np.diag(nii.affine))[:3])
    base_affine[3,3] = 1
    base_affine[:3,3] = voxel_size * resolution/2 * -np.sign(np.diag(nii.affine)[:3])
    return base_affine

def resample_to_axial(mag_nii, scaled_pha_nii):
    base_affine = get_base_affine(mag_nii)

    # resample magnitude to base affine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mag_rot_nii = nilearn.image.resample_img(mag_nii, target_affine=base_affine, target_shape=None, interpolation='continuous')

    # compute real and imaginary components from magnitude and phase
    pha = scaled_pha_nii.get_fdata()
    mag = mag_nii.get_fdata()
    real = mag * np.cos(pha)
    imag = mag * np.sin(pha)
    cplx_header = mag_nii.header
    cplx_header.set_data_dtype(np.float32)
    real_nii = nib.Nifti1Image(real, affine=scaled_pha_nii.affine, header=cplx_header)
    imag_nii = nib.Nifti1Image(imag, affine=scaled_pha_nii.affine, header=cplx_header)

    # resample real and imaginary to base affine
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        real_rot_nii = nilearn.image.resample_img(real_nii, target_affine=base_affine, target_shape=None, interpolation='continuous')
        imag_rot_nii = nilearn.image.resample_img(imag_nii, target_affine=base_affine, target_shape=None, interpolation='continuous')

    # convert real and imaginary to phase
    pha_rot = np.arctan2(imag_rot_nii.get_fdata(), real_rot_nii.get_fdata())
    pha_rot_nii = nib.Nifti1Image(pha_rot, affine=real_rot_nii.affine, header=real_rot_nii.header)

    return mag_rot_nii, pha_rot_nii

def scale_to_pi(pha_nii):
    pha = pha_nii.get_fdata()
    if pha_nii.header.get_data_dtype() == np.int16:
        pha = np.array(np.interp(pha, (np.min(pha), np.max(pha)), (-np.pi, +np.pi)), dtype=np.float32)
    scaled_pha_nii_header = pha_nii.header.copy()
    scaled_pha_nii_header.set_data_dtype(np.float32)
    scaled_pha_nii = nib.Nifti1Image(pha, affine=pha_nii.affine, header=scaled_pha_nii_header)
    return scaled_pha_nii

def scale_to_siemens(pha_nii):
    pha = pha_nii.get_fdata()
    if pha_nii.header.get_data_dtype() != np.int16:
        pha = np.array(np.round(np.interp(pha, (np.min(pha), np.max(pha)), (-4096, +4094)), 0), dtype=np.int16)
    scaled_pha_nii_header = pha_nii.header.copy()
    scaled_pha_nii_header.set_data_dtype(np.int16)
    scaled_pha_nii = nib.Nifti1Image(pha, affine=pha_nii.affine, header=scaled_pha_nii_header)
    return scaled_pha_nii

def resample_batch_to_axial(mag_files, pha_files):
    mag_files = sorted(mag_files)
    pha_files = sorted(pha_files)
    mag_files_new = mag_files
    pha_files_new = pha_files
    assert(len(mag_files) == len(pha_files))

    for i in range(len(mag_files)):

        # load data
        print(f"Loading mag={os.path.split(mag_files[i])[1]}...")
        mag_nii = nib.load(mag_files[i])
        print(f"Loading pha={os.path.split(pha_files[i])[1]}...")
        pha_nii = nib.load(pha_files[i])

        # check obliquity
        obliquity = np.rad2deg(nib.affines.obliquity(mag_nii.affine))
        obliquity_norm = np.linalg.norm(obliquity)
        if obliquity_norm < 10:
            print(f"Obliquity = {obliquity}; norm = {obliquity_norm} < 10; no resampling needed.")
            continue
        print(f"Obliquity = {obliquity}; norm = {obliquity_norm} >= 10; resampling will commence.")

        # scale phase if needed
        if pha_nii.header.get_data_dtype() == np.int16:
            print("Scaling phase to [-pi, +pi]...")
            scaled_pha_nii = scale_to_pi(pha_nii)

        # resample to axial
        print("Resampling to true axial...")
        mag_rot_nii, pha_rot_nii = resample_to_axial(mag_nii, scaled_pha_nii)

        # rescale phase if needed
        if pha_nii.header.get_data_dtype() == np.int16:
            print("Rescaling phase to [-4096, +4094]...")
            pha_rot_nii = scale_to_siemens(pha_rot_nii)

        # ensure magnitude uses int
        mag_rot = mag_rot_nii.get_fdata()
        mag_rot = np.array(np.round(mag_rot, 0), dtype=mag_nii.header.get_data_dtype())
        mag_rot_nii.header.set_data_dtype(mag_nii.header.get_data_dtype())
        mag_rot_nii = nib.Nifti1Image(mag_rot, affine=mag_rot_nii.affine, header=mag_rot_nii.header)
        
        # save results
        mag_fname = os.path.split(mag_files[i])[1].split('.')[0]
        pha_fname = os.path.split(pha_files[i])[1].split('.')[0]
        mag_extension = ".".join(mag_files[i].split('.')[1:])
        pha_extension = ".".join(pha_files[i].split('.')[1:])
        mag_resampled_fname = os.path.abspath(f"{mag_fname}_resampled.{mag_extension}")
        pha_resampled_fname = os.path.abspath(f"{pha_fname}_resampled.{pha_extension}")
        print(f"Saving mag={mag_resampled_fname}")
        nib.save(mag_rot_nii, mag_resampled_fname)
        print(f"Saving pha={pha_resampled_fname}")
        nib.save(pha_rot_nii, pha_resampled_fname)
        mag_files_new[i] = mag_resampled_fname
        pha_files_new[i] = pha_resampled_fname

    return mag_files_new, pha_files_new

def resample_nii_like(in_nii, in_like_nii):
    return nilearn.image.resample_img(in_nii, target_affine=in_like_nii.affine, target_shape=None, interpolation='continuous')

def rotate_nii_mag_phase(mag_nii, scaled_pha_nii):
    r = R.from_euler('x', 45).as_matrix()

    # compute real and imaginary components from magnitude and phase
    pha = scaled_pha_nii.get_fdata()
    mag = mag_nii.get_fdata()
    real = mag * np.cos(pha)
    imag = mag * np.sin(pha)
    cplx_header = mag_nii.header
    cplx_header.set_data_dtype(np.float32)
    real_nii = nib.Nifti1Image(real, affine=scaled_pha_nii.affine, header=cplx_header)
    imag_nii = nib.Nifti1Image(imag, affine=scaled_pha_nii.affine, header=cplx_header)

    # resample real and imaginary to base affine
    with warnings.catch_warnings():
        real_rot_nii = nilearn.image.resample_img(real_nii, target_affine=r, target_shape=None, interpolation='continuous')
        imag_rot_nii = nilearn.image.resample_img(imag_nii, target_affine=r, target_shape=None, interpolation='continuous')

    # convert real and imaginary to phase
    real_rot = real_rot_nii.get_fdata()
    imag_rot = imag_rot_nii.get_fdata()
    mag_rot = np.array(np.round(np.hypot(real_rot, imag_rot), 0), dtype=mag_nii.header.get_data_dtype())
    pha_rot = np.array(np.arctan2(imag_rot, real_rot), dtype=np.float16)

    # create nifti objects
    mag_rot_nii = nib.Nifti1Image(mag_rot, affine=real_rot_nii.affine, header=mag_nii.header)
    pha_rot_nii = nib.Nifti1Image(pha_rot, affine=real_rot_nii.affine, header=scaled_pha_nii.header)

    return mag_rot_nii, pha_rot_nii

def rotate_nii(in_nii):
    r = R.from_euler('x', -45).as_matrix()
    rot_nii = nilearn.image.resample_img(in_nii, target_affine=r, target_shape=None, interpolation='continuous')
    return rot_nii

def rotate_mag_phase_batch():
    mag_files = sorted(glob.glob("sub*mag*nii*"))
    pha_files = sorted(glob.glob("sub*pha*nii*"))

    for i in range(len(mag_files)):
        mag_nii = nib.load(mag_files[i])
        pha_nii = nib.load(pha_files[i])

        mag_rot_nii, pha_rot_nii = rotate_nii(mag_nii, pha_nii)

        mag_fname = os.path.split(mag_files[i])[1].split('.')[0]
        pha_fname = os.path.split(pha_files[i])[1].split('.')[0]
        mag_extension = ".".join(mag_files[i].split('.')[1:])
        pha_extension = ".".join(pha_files[i].split('.')[1:])
        mag_resampled_fname = os.path.abspath(f"{mag_fname}_resampled.{mag_extension}")
        pha_resampled_fname = os.path.abspath(f"{pha_fname}_resampled.{pha_extension}")
        print(f"Saving mag={mag_resampled_fname}")
        nib.save(mag_rot_nii, mag_resampled_fname)
        print(f"Saving pha={pha_resampled_fname}")
        nib.save(pha_rot_nii, pha_resampled_fname)

def set_axial_affine(nii):
    return nib.Nifti1Image(nii.get_fdata(), affine=get_base_affine(nii), header=nii.header)

if __name__ == "__main__":
    in_sim_norot_file = "/home/user/neurodesktop-storage/qsmxt/data/2022-orientation-abstract/tinaroo-results/qsm2/results/sub-sim_no-rot.nii"
    in_sim_noprocessing_file = "/home/user/neurodesktop-storage/qsmxt/data/2022-orientation-abstract/tinaroo-results/qsm2/results/sub-sim_rot-45_no-resampling_resampled.nii.gz"
    in_sim_processing_file = "/home/user/neurodesktop-storage/qsmxt/data/2022-orientation-abstract/tinaroo-results/qsm2/results/sub-sim_rot-45_resampled.nii.gz"
    in_sim_processing_diff_file = "/home/user/neurodesktop-storage/qsmxt/data/2022-orientation-abstract/tinaroo-results/qsm2/results/sub-sim_processing-diff.nii.gz"
    in_sim_noprocessing_diff_file = "/home/user/neurodesktop-storage/qsmxt/data/2022-orientation-abstract/tinaroo-results/qsm2/results/sub-sim_noprocessing-diff.nii.gz"
    in_chi_file = "/home/user/neurodesktop-storage/data/2022-orientation-abstract/tinaroo-results/qsm2/results/Chi_cropBrainExtracted.nii.gz"

    in_sim_norot_nii = nib.load(in_sim_norot_file)
    in_sim_noprocessing_nii = nib.load(in_sim_noprocessing_file)
    in_sim_processing_nii = nib.load(in_sim_processing_file)
    in_sim_processing_diff_nii = nib.load(in_sim_processing_diff_file)
    in_sim_noprocessing_diff_nii = nib.load(in_sim_noprocessing_diff_file)
    in_chi_nii = nib.load(in_chi_file)

    sim_norot = in_sim_norot_nii.get_fdata()
    sim_noprocessing = in_sim_noprocessing_nii.get_fdata()
    sim_processing = in_sim_processing_nii.get_fdata()
    sim_processing_diff = in_sim_processing_diff_nii.get_fdata()
    sim_noprocessing_diff = in_sim_noprocessing_diff_nii.get_fdata()
    chi = in_chi_nii.get_fdata()

    slc = 102

    in_sim_norot_slice = np.rot90(sim_norot[:,:,slc])
    in_sim_noprocessing_slice = np.rot90(sim_noprocessing[:,:,slc])
    in_sim_processing_slice = np.rot90(sim_processing[:,:,slc])
    in_sim_processing_diff_slice = np.rot90(sim_processing_diff[:,:,slc])
    in_sim_noprocessing_diff_slice = np.rot90(sim_noprocessing_diff[:,:,slc])
    in_chi_slice = np.rot90(chi[:,:,slc])
    in_chi_norot_diff_slice = np.rot90(chi[:,:,slc] - sim_norot[:,:,slc])
    in_chi_processing_diff_slice = np.rot90(chi[:,:,slc] - sim_processing[:,:,slc])
    in_chi_noprocessing_diff_slice = np.rot90(chi[:,:,slc] - sim_noprocessing[:,:,slc])

    fig = plt.figure()

    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(3,4) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0,1].imshow(in_sim_norot_slice, cmap='gray', vmin=-0.05, vmax=+0.05)
    axarr[0,2].imshow(in_sim_processing_slice, cmap='gray', vmin=-0.05, vmax=+0.05)
    axarr[0,3].imshow(in_sim_noprocessing_slice, cmap='gray', vmin=-0.05, vmax=+0.05)
    axarr[1,0].imshow(in_sim_norot_slice, cmap='gray', vmin=-0.05, vmax=+0.05)
    axarr[1,2].imshow(in_sim_processing_diff_slice, cmap='seismic', vmin=-0.05, vmax=+0.05)
    axarr[1,3].imshow(in_sim_noprocessing_diff_slice, cmap='seismic', vmin=-0.05, vmax=+0.05)
    axarr[2,0].imshow(in_chi_slice, cmap='gray', vmin=-0.1, vmax=+0.1)
    axarr[2,1].imshow(in_chi_norot_diff_slice, cmap='seismic', vmin=-0.1, vmax=+0.1)
    axarr[2,2].imshow(in_chi_processing_diff_slice, cmap='seismic', vmin=-0.1, vmax=+0.1)
    axarr[2,3].imshow(in_chi_noprocessing_diff_slice, cmap='seismic', vmin=-0.1, vmax=+0.1)

    axarr[0,0].axis('off')
    axarr[0,1].axis('off')
    axarr[0,2].axis('off')
    axarr[0,3].axis('off')
    axarr[1,0].axis('off')
    axarr[1,1].axis('off')
    axarr[1,2].axis('off')
    axarr[1,3].axis('off')
    axarr[2,0].axis('off')
    axarr[2,1].axis('off')
    axarr[2,2].axis('off')
    axarr[2,3].axis('off')
    

    #axarr[0,0].title.set_text("test")
    
    plt.tight_layout()
    plt.savefig("test.png", dpi=700)

