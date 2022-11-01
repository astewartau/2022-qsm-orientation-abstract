import os
import nibabel as nib
import numpy as np
import nilearn.image
import glob
import warnings
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import subprocess
import tempfile
import osfclient
import pandas as pd
from shutil import move
import seaborn as sns
from sklearn.metrics import mean_squared_error as mse

def sys_cmd(cmd, print_output=True, print_command=True):
    if print_command:
        print(cmd)
    
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
    result_stdout = result.stdout
    result_stderr = result.stderr
    
    result_stdout_str = result_stdout.decode('UTF-8')[:-1]
    
    if print_output:
        print(result_stdout_str, end="")
        
    if result_stderr:
        raise RuntimeError(f"Runtime error: {result_stderr}.")
    
    return result_stdout_str

def get_bids_data():
    tmp_dir = tempfile.gettempdir()
    if not os.path.exists(os.path.join(tmp_dir, 'bids-osf')):
        if not os.path.exists(os.path.join(tmp_dir, 'bids-osf.tar')):
            print("Downloading test data...")
            file_pointer = next(osfclient.OSF().project("9jc42").storage().files)
            file_handle = open(os.path.join(tmp_dir, 'bids-osf.tar'), 'wb')
            file_pointer.write_to(file_handle)
        print("Extracting test data...")
        sys_cmd(f"tar xf {os.path.join(tmp_dir, 'bids-osf.tar')} -C {tmp_dir}")
        sys_cmd(f"rm {os.path.join(tmp_dir, 'bids-osf.tar')}")
    return os.path.join(tmp_dir, 'bids-osf')

def get_extension(filename):
    return ".".join(filename.split(".")[1:])

def get_fname(filename, with_ext=True):
    if not with_ext: return os.path.split(filename)[1].split(".")[0]
    return os.path.split(filename)[1]

def get_dir(filename):
    return os.path.split(filename)[0]

def fname_append(filename, new_part, out_dir=None, fname_only=False):
    dir_part = get_dir(filename)
    extension = get_extension(filename)
    fname = get_fname(filename, with_ext=False)
    if fname_only: return f"{fname}{new_part}.{extension}"
    if not out_dir: return f"{dir_part}/{fname}{new_part}.{extension}"
    return f"{out_dir}/{fname}{new_part}.{extension}"

def nonzero_average(in_files, out_file):
    data = []
    for in_nii_file in in_files:
        in_nii = nib.load(in_nii_file)
        in_data = in_nii.get_fdata()
        data.append(in_data)
    try:
        data = np.array(data)
        mask = abs(data) >= 0.0001
    except ValueError:
        sizes = [x.shape for x in data]
        raise ValueError(f"Tried to average files of incompatible dimensions; {sizes}")
    final = np.divide(data.sum(0), mask.sum(0), out=np.zeros_like(data.sum(0)), where=mask.sum(0)!=0)
    nib.save(nib.nifti1.Nifti1Image(final, affine=in_nii.affine, header=in_nii.header), out_file)
    return out_file

def get_base_affine(nii):
    voxel_size = np.array(nii.header.get_zooms())
    resolution = np.array(nii.header.get_data_shape())
    base_affine = np.eye(4)
    np.fill_diagonal(base_affine, voxel_size * np.sign(np.diag(nii.affine))[:3])
    base_affine[3,3] = 1
    base_affine[:3,3] = voxel_size * resolution/2 * -np.sign(np.diag(nii.affine)[:3])
    return base_affine

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

def resample_to_axial_nii(mag_nii, scaled_pha_nii):
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

def resample_to_axial(mag_files, pha_files, out_dir=None):

    mag_files = sorted(mag_files)
    pha_files = sorted(pha_files)
    mag_files_new = mag_files.copy()
    pha_files_new = pha_files.copy()
    assert(len(mag_files) == len(pha_files))

    for i in range(len(mag_files)):
        
        mag_files_new[i] = fname_append(mag_files[i], "_resampled-axial", out_dir)
        pha_files_new[i] = fname_append(pha_files[i], "_resampled-axial", out_dir)
        
        if os.path.exists(mag_files_new[i]) and os.path.exists(pha_files_new[i]):
            continue

        # load data
        print(f"Loading mag={os.path.split(mag_files[i])[1]}...")
        mag_nii = nib.load(mag_files[i])
        print(f"Loading pha={os.path.split(pha_files[i])[1]}...")
        pha_nii = nib.load(pha_files[i])

        # check obliquity
        obliquity = np.rad2deg(nib.affines.obliquity(mag_nii.affine))
        obliquity_norm = np.linalg.norm(obliquity)

        # scale phase if needed
        if pha_nii.header.get_data_dtype() == np.int16:
            print("Scaling phase to [-pi, +pi]...")
            pha_nii = scale_to_pi(pha_nii)

        # resample to axial
        print("Resampling to true axial...")
        mag_rot_nii, pha_rot_nii = resample_to_axial_nii(mag_nii, pha_nii)

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
        print(f"Saving mag={mag_files_new[i]}")
        print(f"Saving pha={pha_files_new[i]}")
        nib.save(mag_rot_nii, mag_files_new[i])
        nib.save(pha_rot_nii, pha_files_new[i])

    return mag_files_new, pha_files_new

def resample_like(in_file, in_like_file, out_dir=None):
    out_file = fname_append(in_file, "_resample-like", out_dir if out_dir else get_dir(in_file))
    if os.path.exists(out_file): return out_file
    in_nii = nib.load(in_file)
    in_like_nii = nib.load(in_like_file)
    in_nii_resampled = nilearn.image.resample_img(in_nii, target_affine=in_like_nii.affine, target_shape=np.array(in_like_nii.header.get_data_shape()), interpolation='continuous')
    nib.save(in_nii_resampled, out_file)
    return out_file

def rotate_nii_mask(mask_nii, degrees):
    r = R.from_euler('x', degrees, degrees=True).as_matrix()
    with warnings.catch_warnings():
        mask_rot_nii = nilearn.image.resample_img(mask_nii, target_affine=r, target_shape=None, interpolation='nearest')
    return mask_rot_nii

def rotate_nii(mag_nii, scaled_pha_nii, degrees):
    r = R.from_euler('x', degrees, degrees=True).as_matrix()

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

def rotate_nii_batch():
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

def rotate_mag_phase(mag_files, pha_files, out_dir, degrees):
    os.makedirs(out_dir, exist_ok=True)
    rotated_mag_files = []
    rotated_pha_files = []
    for i in range(len(mag_files)):
        rotated_mag_name = fname_append(mag_files[i], f"_rot-{degrees}", out_dir)
        rotated_pha_name = fname_append(pha_files[i], f"_rot-{degrees}", out_dir)
        rotated_mag_files.append(rotated_mag_name)
        rotated_pha_files.append(rotated_pha_name)
        if os.path.exists(rotated_mag_name) and os.path.exists(rotated_pha_name): continue
        mag_nii = nib.load(mag_files[i])
        pha_nii = nib.load(pha_files[i])
        mag_rot_nii, pha_rot_nii = rotate_nii(mag_nii, pha_nii, degrees)
        nib.save(mag_rot_nii, rotated_mag_name)
        nib.save(pha_rot_nii, rotated_pha_name)
    return sorted(rotated_mag_files), sorted(rotated_pha_files)

def bet_masking(mag_file, fractional_intensity=0.5, out_dir=None, ignore_cache=False):
    bet_cmd = "bet {magnitude} {bet_output} -m {mask_output} -f {fractional_intensity}"
    if not out_dir: out_dir = get_dir(mag_file)
    bet_file = fname_append(mag_file, "_bet", out_dir=out_dir)
    mask_file = fname_append(mag_file, "_bet-mask", out_dir=out_dir)
    if os.path.exists(mask_file): return mask_file
    sys_cmd(bet_cmd.format(
        magnitude=mag_file,
        bet_output=bet_file,
        mask_output=mask_file,
        fractional_intensity=fractional_intensity
    ))
    return mask_file

def tgv_qsm_me(mask_file, pha_files, TEs, B0_str, eros, out_dir=None):
    os.environ["OMP_NUM_THREADS"] = "12"# str(os.cpu_count())
    tgv_qsm_cmd = "tgv_qsm -t {TE} --alpha 0.0015 0.0005  -i 1000 -f {B0_str} -e {eros} --ignore-orientation --no-resampling -m {mask} -o _qsm -p {phase}"
    qsm_files = []
    for i in range(len(pha_files)):
        out_file = fname_append(pha_files[i], "_qsm_000")
        qsm_files.append(out_file)
        if not out_dir and os.path.exists(out_file): continue
        if out_dir and os.path.exists(os.path.join(out_dir, get_fname(qsm_files[i]))): continue
        sys_cmd(tgv_qsm_cmd.format(
            TE=TEs[i],
            B0_str=B0_str,
            eros=eros,
            mask=mask_file,
            phase=pha_files[i]
        ))

    # move qsm files
    qsm_files.sort()
    if out_dir:
        for i in range(len(qsm_files)):
            new_qsm_fname = os.path.join(out_dir, get_fname(qsm_files[i]))
            qsm_files[i] = new_qsm_fname
            if not os.path.exists(new_qsm_fname):
                move(qsm_files[i], new_qsm_fname)

    # averaging
    qsm_average_fname = fname_append(qsm_files[0], "_average")
    if os.path.exists(qsm_average_fname): return qsm_average_fname
    return nonzero_average(qsm_files, qsm_average_fname)

def nextqsm_normalise_phase(pha_file, B0_str, TE, out_dir=None):
    pha_normalised_file = fname_append(pha_file, "_normalised", out_dir=out_dir)
    if os.path.exists(pha_normalised_file): return pha_normalised_file
    phase_nii = nib.load(pha_file)
    phase = phase_nii.get_fdata()
    centre_freq = 127736254 / 3 * B0_str
    normalised = phase / (2 * np.pi * TE * centre_freq) * 1e6
    nib.save(nib.nifti1.Nifti1Image(normalised, affine=phase_nii.affine, header=phase_nii.header), pha_normalised_file)
    return pha_normalised_file

def nextqsm_me(mask_file, pha_files, TEs, B0_str, eros, out_dir=None):
    # unwrap phase files
    pha_unwrapped_files = []
    unwrapping_cmd = "laplacian_unwrapping.jl {phase} {out_file}"
    for i in range(len(pha_files)):
        out_file = fname_append(pha_files[i], "_unwrapped", out_dir=out_dir)
        pha_unwrapped_files.append(out_file)
        if os.path.exists(out_file): continue
        sys_cmd(unwrapping_cmd.format(
            phase=pha_files[i],
            out_file=out_file
        ))
        if not os.path.exists(out_file):
            print(f"File {out_file} not found!")
            exit(1)
    
    # normalise unwrapped files
    pha_normalised_files = []
    for i in range(len(pha_unwrapped_files)):
        pha_normalised_files.append(nextqsm_normalise_phase(pha_unwrapped_files[i], B0_str, TEs[i], out_dir=out_dir))    
        if not os.path.exists(pha_normalised_files[i]):
            print(f"File {pha_normalised_files[i]} not found!")
            exit(1)

    # nextqsm on original    
    nextqsm_cmd = "predict_all.py --phase {phase} --mask {mask} --out_file {out_file}"
    qsm_files = []
    for i in range(len(pha_normalised_files)):
        out_file = fname_append(pha_normalised_files[i], "_nextqsm", out_dir=out_dir)
        qsm_files.append(out_file)
        if os.path.exists(out_file): continue
        sys_cmd(nextqsm_cmd.format(
            mask=mask_file,
            phase=pha_normalised_files[i],
            out_file=out_file
        ))
        if not os.path.exists(out_file):
            print(f"File {out_file} not found!")
            exit(1)

    # move qsm files
    qsm_files.sort()
    if out_dir:
        for i in range(len(qsm_files)):
            new_qsm_fname = os.path.join(out_dir, get_fname(qsm_files[i]))
            if not os.path.exists(new_qsm_fname):
                move(qsm_files[i], new_qsm_fname)
            qsm_files[i] = new_qsm_fname

    # averaging
    qsm_average_fname = fname_append(qsm_files[0], "_average")
    if os.path.exists(qsm_average_fname): return qsm_average_fname
    return nonzero_average(qsm_files, qsm_average_fname)

if __name__ == "__main__":
    
    # paths and files
    bids_dir = get_bids_data()
    session_path = os.path.join(bids_dir, "sub-1", "ses-1")
    anat_path = os.path.join(session_path, "anat")
    extra_data_path = os.path.join(session_path, "extra_data")
    chi_file = os.path.join(extra_data_path, "sub-1_ses-1_run-01_chi-cropped.nii.gz")
    segmentation_file = os.path.join(extra_data_path, "sub-1_ses-1_run-01_segmentation.nii.gz")
    mag_files = sorted(glob.glob(os.path.join(anat_path, "sub-1_ses-1_run-01_echo-*_part-mag_MEGRE.nii.gz")))
    pha_files = sorted(glob.glob(os.path.join(anat_path, "sub-1_ses-1_run-01_echo-*_part-phase_MEGRE.nii.gz")))
    assert(len(mag_files) == len(pha_files))
    tmp_dir = tempfile.gettempdir()
    
    # create augmented data
    degrees = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    for d in degrees:
        print(f"Creating rotated input images ({d} degrees)")
        out_dir = os.path.join(tmp_dir, f"inputs-rotated-{d}-degrees")
        print(f"Rotating data {d} degrees and resampling...")
        mag_rotated, pha_rotated = rotate_mag_phase(mag_files, pha_files, out_dir, d)
        print(f"Resampling rotated data to axial...")
        mag_rotated_axial, pha_rotated_axial = resample_to_axial(mag_rotated, pha_rotated, out_dir)

    # reconstruction parameters
    TEs = [0.004, 0.012, 0.020, 0.028]
    B0_str = 3
    eros = 3
    fractional_intensity = 0.5

    # original data
    print("BET masking for original data...")
    mask_file = bet_masking(mag_files[0], fractional_intensity, extra_data_path)
    print("QSM reconstruction for original data...")
    qsm_average_file = tgv_qsm_me(mask_file, pha_files, TEs, B0_str, eros, extra_data_path)
    #qsm_average_file = nextqsm_me(mask_file, pha_files, TEs, B0_str, eros, extra_data_path)

    # augmented data
    degrees = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    result_files = { }
    for d in degrees:
        out_dir = os.path.join(extra_data_path, f"inputs-rotated-{d}-degrees")
        mag_rotated, pha_rotated = rotate_mag_phase(mag_files, pha_files, out_dir, d)
        mag_rotated_axial, pha_rotated_axial = resample_to_axial(mag_rotated, pha_rotated, out_dir)

        print("BET masking rotated data...")
        mask_rotated = bet_masking(mag_rotated[0], fractional_intensity, out_dir)
        print("Reconstructing QSM for rotated data...")
        qsm_rotated_average = tgv_qsm_me(mask_rotated, pha_rotated, TEs, B0_str, eros, out_dir)
        #qsm_rotated_average = nextqsm_me(mask_rotated, pha_rotated, TEs, B0_str, eros, out_dir) #TODO REMOVE
        print("Resampling QSM to original space...")
        qsm_rotated_average_resampled_file = resample_like(qsm_rotated_average, mag_files[0])

        print("BET masking resampled-to-axial data...")
        mask_rotated_axial = bet_masking(mag_rotated_axial[0], fractional_intensity, out_dir)
        print("Reconstructing QSM for resampled-to-axial data...")
        qsm_rotated_axial_average = tgv_qsm_me(mask_rotated_axial, pha_rotated_axial, TEs, B0_str, eros, out_dir)
        #qsm_rotated_axial_average = nextqsm_me(mask_rotated_axial, pha_rotated_axial, TEs, B0_str, eros, out_dir)
        print("Resampling QSM to original space...")
        qsm_rotated_axial_average_resampled_file = resample_like(qsm_rotated_axial_average, mag_files[0])
        
        result_files[d] = [qsm_rotated_axial_average_resampled_file, qsm_rotated_average_resampled_file]
    
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
    
    # load ground truth data
    segmentation = np.rot90(nib.load(segmentation_file).get_fdata())
    chi = np.rot90(nib.load(chi_file).get_fdata())
    qsm_average = np.rot90(nib.load(qsm_average_file).get_fdata())
    chi = chi * (qsm_average != 0)
    mask = qsm_average != 0
    
    # Rotation Segmentation RMSE
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
    
    print("Creating line plot...")
    sns.set_style("darkgrid")
    sns.set_context("paper", rc={"font.size":8,"axes.titlesize":7,"axes.labelsize":5})
    fig, axes = plt.subplots(3,4)
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
        plot.tick_params(axis='both', which='major', labelsize=4)
        plot.tick_params(axis='both', which='minor', labelsize=4)
        plt.xlabel("")
        plot.get_legend().remove()
        index += 1
    handles, labels = plot.get_legend_handles_labels()
    plot.legend(handles, labels, loc=(1.2, 3.025))
    plt.savefig(f"lineplot.png", dpi=700, bbox_inches='tight')
    plt.close()
    
    print("Creating diff image figure...")
    plt.figure()
    sns.set_style("darkgrid")
    sns.set_context("paper", rc={"font.size":13,"axes.titlesize":13,"axes.labelsize":13})
    fig, axes = plt.subplots(4, 5, figsize=[10,10])
    fig.suptitle('QSM computed using TGV-QSM using simulated data rotated to various angles')
    slc = 102
    vmin = -0.1
    vmax = +0.1
    
    axes[0,0].set_ylabel("QSM at θ", rotation=0, horizontalalignment='right')
    axes[0,0].set_title("0°")
    axes[0,0].imshow(qsm_average[:,:,slc], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,1].set_title("5°")
    axes[0,1].imshow(np.rot90(nib.load(result_files[5][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,2].set_title("10°")
    axes[0,2].imshow(np.rot90(nib.load(result_files[10][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,3].set_title("15°")
    axes[0,3].imshow(np.rot90(nib.load(result_files[15][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[0,4].set_title("20°")
    axes[0,4].imshow(np.rot90(nib.load(result_files[20][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    
    axes[1,0].set_ylabel("Diff vs. 0°", rotation=0, horizontalalignment='right')
    axes[1,0].imshow(qsm_average[:,:,slc] - qsm_average[:,:,slc], cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1,1].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[5][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1,2].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[10][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1,3].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[15][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1,4].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[20][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    
    axes[2,0].set_title("25°")
    axes[2,0].set_ylabel("QSM at θ", rotation=0, horizontalalignment='right')
    axes[2,0].imshow(np.rot90(nib.load(result_files[25][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[2,1].set_title("30°")
    axes[2,1].imshow(np.rot90(nib.load(result_files[30][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[2,2].set_title("35°")
    axes[2,2].imshow(np.rot90(nib.load(result_files[35][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[2,3].set_title("40°")
    axes[2,3].imshow(np.rot90(nib.load(result_files[40][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    axes[2,4].set_title("45°")
    im1 = axes[2,4].imshow(np.rot90(nib.load(result_files[45][1]).get_fdata()[:,:,slc]), cmap='gray', vmin=vmin, vmax=vmax)
    
    axes[3,0].set_ylabel("Diff vs. 0°", rotation=0, horizontalalignment='right')
    axes[3,0].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[25][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[3,1].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[30][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[3,2].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[35][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    axes[3,3].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[40][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    im2 = axes[3,4].imshow(qsm_average[:,:,slc] - np.rot90(nib.load(result_files[45][1]).get_fdata()[:,:,slc]), cmap='seismic', vmin=vmin, vmax=vmax)
    
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
    
    plt.savefig(f"diff-images.png", dpi=700, bbox_inches='tight')
    
        
    
    