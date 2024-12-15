import numpy as np
import os
from mat_npy import load_data_from_npy
def align_shapes(*arrays):
    min_rows = min(array.shape[0] for array in arrays)
    min_cols = min(array.shape[1] for array in arrays)
    return tuple(array[:min_rows, :min_cols] for array in arrays)

def spectral_angle_mapper(img_data, target_spectrum):
    if np.all(target_spectrum == 0):  # Check for all zeros
        raise ValueError("Target spectrum is invalid (all zeros).")
    target_norm = np.linalg.norm(target_spectrum)
    img_norms = np.linalg.norm(img_data, axis=-1)
    dot_products = np.einsum("ijk,k->ij", img_data, target_spectrum)
    cos_angles = 1 - (dot_products / (img_norms * target_norm))
    return cos_angles

def mutual_info_band_selection(img_data, labels, n_bands=10):
    print("Using variance as a proxy for mutual information (NumPy only).")
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    mi_scores = np.var(reshaped_data, axis=0)  # Variance as proxy
    top_band_indices = np.argsort(mi_scores)[-n_bands:]
    return img_data[:, :, top_band_indices], top_band_indices

def remove_low_variance_bands(img_data, threshold=0.01):
    band_variances = np.var(img_data, axis=(0, 1))
    selected_bands = np.where(band_variances > threshold)[0]
    return img_data[:, :, selected_bands], selected_bands

def fft_denoise(img_data, threshold=0.1): # FFT-based denoising (NumPy only)
    denoised_bands = []
    for i in range(img_data.shape[2]):
        coeffs = np.fft.fft2(img_data[:, :, i])
        coeffs[np.abs(coeffs) < threshold] = 0
        denoised_bands.append(np.fft.ifft2(coeffs).real) #Take the real part
    return np.stack(denoised_bands, axis=-1)



def radiometric_correction(img_data, gain=1, offset=0):
    corrected_img = (img_data * gain) + offset
    print("Radiometric correction applied.")
    return corrected_img

def atmospheric_correction(img_data):
    dark_object_value = np.min(img_data, axis=(0, 1))
    corrected_img = img_data - dark_object_value
    corrected_img[corrected_img < 0] = 0
    print("Atmospheric correction applied using Dark Object Subtraction.")
    return corrected_img

def apply_incremental_pca(img_data, n_components=10, batch_size=1000):
    reshaped_data = img_data.reshape(-1, img_data.shape[2]).astype(np.float32)  # Ensure float32 for stability
    cov_matrix = np.cov(reshaped_data, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)  # Use eigh for symmetric matrices
    top_indices = np.argsort(eig_vals)[-n_components:][::-1]

    pca_data = np.dot(reshaped_data, eig_vecs[:, top_indices])
    pca_img = np.dot(pca_data, eig_vecs[:, top_indices].T).reshape(img_data.shape)

    return pca_img

def calculate_snr(img_data):
    mean_signal = np.mean(img_data, axis=(0, 1))
    noise = img_data - mean_signal
    snr = mean_signal / (np.std(noise, axis=(0, 1)) + 1e-10)  # Add small value to avoid division by zero
    print(f"Signal-to-Noise Ratio (SNR) calculated for each band.")
    return snr

def mnf_transform(img_data, n_components=10):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    cov = np.cov(reshaped_data, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)  # eigh is more suitable for covariance matrices
    indices = np.argsort(eig_vals)[-n_components:][::-1]
    mnf_img = np.dot(reshaped_data, eig_vecs[:, indices])
    mnf_img = mnf_img.reshape(img_data.shape[0], img_data.shape[1], n_components)
    print(f"MNF transformation applied with {n_components} components.")
    return mnf_img


def spectral_fusion(img1, img2):
    fused_img = (img1 + img2) / 2
    print("Spectral fusion applied.")
    return fused_img

def refine_band_selection(img_data, map_data, target_pixel=(50, 50), n_bands=10):

    target_spectrum = img_data[target_pixel[0], target_pixel[1], :]
    sam_result = spectral_angle_mapper(img_data, target_spectrum)

    anomaly_threshold = np.percentile(sam_result, 95)  # Using percentile is more robust
    labels = (sam_result > anomaly_threshold).astype(int).flatten()


    selected_img, selected_bands = mutual_info_band_selection(img_data, labels, n_bands=n_bands)

    spectral_ranges = [(0, 32), (150, 200)]  # Example ranges
    filtered_bands = []
    for start, end in spectral_ranges:
        filtered_bands.extend([band for band in selected_bands if start <= band <= end])

    print(f"Selected bands in specified ranges: {filtered_bands}")

    return selected_img, filtered_bands




def clean_and_process_data(file_path):
    img_data, map_data = load_data_from_npy()
    img_data, map_data = align_shapes(img_data, map_data)


    reduced_img, selected_bands = remove_low_variance_bands(img_data)
    radiometric_img = radiometric_correction(reduced_img)
    atmos_corrected_img = atmospheric_correction(radiometric_img)

    pca_img = apply_incremental_pca(atmos_corrected_img, n_components=30, batch_size=10000)
    print("Incremental PCA applied.")

    denoised_img = fft_denoise(pca_img, threshold=0.2)  # Using FFT denoising (NumPy only)
    print("FFT denoising applied.")

    snr = calculate_snr(denoised_img)
    print(f"SNR values: {snr}")


    mnf_img = mnf_transform(denoised_img, n_components=15)  # Reduced components to avoid potential errors
    print("MNF transformation applied.")


    selected_img, refined_bands = refine_band_selection(mnf_img, map_data)
    print("Refined band selection completed.")

    print("Data cleaning and processing completed successfully.")
    return selected_img, refined_bands, selected_bands



