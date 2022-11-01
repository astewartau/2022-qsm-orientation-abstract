import os
import nibabel as nib
import numpy as np
import nilearn.image
import glob
import warnings

def resample_to_axial(mag_nii, scaled_pha_nii):
    # calculate base affine
    voxel_size = np.array(mag_nii.header.get_zooms())
    resolution = np.array(mag_nii.header.get_data_shape())
    base_affine = np.eye(4)
    np.fill_diagonal(base_affine, voxel_size * np.sign(np.diag(mag_nii.affine))[:3])
    base_affine[3,3] = 1
    base_affine[:3,3] = voxel_size * resolution/2 * -np.sign(np.diag(mag_nii.affine)[:3])

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

def resample_batch(mag_files, pha_files):

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

def resample_like(in_file, in_like_file):
    in_nii = nib.load(in_file)
    in_like_nii = nib.load(in_like_file)
    in_nii_resampled = nilearn.image.resample_img(in_nii, target_affine=in_like_nii.affine, target_shape=np.array(in_like_nii.header.get_data_shape()), interpolation='continuous')
    in_fname = os.path.split(in_file)[1].split('.')[0]
    in_extension = ".".join(in_file.split('.')[1:])
    in_resampled_fname = os.path.abspath(f"{in_fname}_resampled.{in_extension}")
    nib.save(in_nii_resampled, in_resampled_fname)
    return in_resampled_fname

if __name__ == "__main__":
    #mag_files = sorted(glob.glob("bids/sub-real/ses-original/anat/*mag*nii*gz"))
    #pha_files = sorted(glob.glob("bids/sub-real/ses-original/anat/*phase*nii*gz"))
    #print(resample_batch(mag_files, pha_files))

    print(resample_like("sub-real_ses-original_run-01_part-mag_T2starw_resampled.nii.gz", "qsm/qsm_final/sub-real_ses-original_run-01_part-phase_T2starw_scaled_qsm-filled_000_average.nii"))
    
