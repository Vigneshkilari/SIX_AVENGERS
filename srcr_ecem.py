import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

# Load the .mat file
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['img'], data['map']

# Preprocessing: Remove bands with low variance
def remove_low_variance_bands(img_data, threshold=0.01):
    band_variances = np.var(img_data, axis=(0, 1))
    selected_bands = np.where(band_variances > threshold)[0]
    reduced_img = img_data[:, :, selected_bands]
    print(f"Selected {len(selected_bands)} bands")
    return reduced_img, selected_bands
# Perform RX Algorithm for anomaly detection
def rx_algorithm(img_data, alpha=0.05):
    # Flatten the hyperspectral image
    reshaped_data = img_data.reshape(-1, img_data.shape[2])

    # Fit a Gaussian distribution to the data using empirical covariance
    covariance = EmpiricalCovariance().fit(reshaped_data)

    # Calculate the Mahalanobis distance for each pixel
    mahalanobis_dist = covariance.mahalanobis(reshaped_data - covariance.location_)

    # Apply the threshold to classify anomalies
    threshold = np.percentile(mahalanobis_dist, 100 * (1 - alpha))
    anomaly_mask = mahalanobis_dist > threshold

    # Reshape the result back to image dimensions
    anomaly_img = anomaly_mask.reshape(img_data.shape[0], img_data.shape[1])
    return anomaly_img

# Apply Wavelet Denoising
def wavelet_denoise(img_data, wavelet='db1', level=1, threshold=0.1):
    denoised_img = np.zeros_like(img_data)
    for i in range(img_data.shape[2]):  # Process each band
        band_data = img_data[:, :, i]

        # Perform discrete wavelet transform (DWT)
        coeffs = pywt.wavedec2(band_data, wavelet, level=level)

        # Apply thresholding on the wavelet coefficients (both approximation and details)
        coeffs_thresholded = list(coeffs)  # Create a copy of the coefficients list
        for j in range(1, len(coeffs_thresholded)):  # Skip the approximation coefficients
            coeffs_thresholded[j] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs_thresholded[j])

        # Reconstruct the denoised band
        denoised_band = pywt.waverec2(coeffs_thresholded, wavelet)

        # Ensure the denoised band matches the original size (crop or pad if necessary)
        if denoised_band.shape != band_data.shape:
            min_shape = np.minimum(band_data.shape, denoised_band.shape)
            denoised_band = denoised_band[:min_shape[0], :min_shape[1]]

        denoised_img[:, :, i] = denoised_band

    return denoised_img

# Radiometric Correction
def radiometric_correction(img_data, gain=1, offset=0):
    corrected_img = (img_data * gain) + offset
    print("Radiometric correction applied.")
    return corrected_img

# Atmospheric Correction using Dark Object Subtraction (DOS)
def atmospheric_correction(img_data):
    dark_object_value = np.min(img_data, axis=(0, 1))  # Minimum value per band
    corrected_img = img_data - dark_object_value
    corrected_img[corrected_img < 0] = 0  # Ensure no negative values
    print("Atmospheric correction applied using Dark Object Subtraction.")
    return corrected_img

# SRCR-ECEM Anomaly Detection (Simplified Version)
def srcr_ecem_anomaly_detection(img_data, lambda_s=0.01, lambda_c=0.01):
    """
    Perform anomaly detection using SRCR-ECEM.

    Args:
        img_data (numpy.ndarray): Hyperspectral image data.
        lambda_s (float): Regularization parameter for spatial smoothness.
        lambda_c (float): Regularization parameter for endmember correlation.

    Returns:
        numpy.ndarray: Anomaly detection map.
    """
    reshaped_data = img_data.reshape(-1, img_data.shape[2])

    # Normalize data for better numerical stability
    normalized_data = reshaped_data / np.linalg.norm(reshaped_data, axis=1, keepdims=True)

    # Collaborative representation and spatial regularization
    endmembers = np.mean(normalized_data, axis=0)  # Simplified endmember extraction
    residuals = normalized_data - endmembers

    # Apply spatial regularization and endmember correlation enhancement
    anomaly_scores = np.linalg.norm(residuals, axis=1) + lambda_s * np.var(residuals, axis=1)

    # Reshape back to image dimensions
    anomaly_img = anomaly_scores.reshape(img_data.shape[0], img_data.shape[1])

    return anomaly_img

# Main function to clean and process data
def clean_and_process_data(file_path):
    # Load the data
    img_data, map_data = load_data(file_path)
    print("Data loaded successfully.")

    # 1. Remove low variance bands
    reduced_img, selected_bands = remove_low_variance_bands(img_data)

    # 2. Apply radiometric correction
    radiometric_img = radiometric_correction(reduced_img)

    # 3. Apply atmospheric correction
    atmos_corrected_img = atmospheric_correction(radiometric_img)

    # 4. Apply wavelet denoising
    denoised_img = wavelet_denoise(atmos_corrected_img, wavelet='db1', level=2, threshold=0.1)
    print("Wavelet denoising applied.")

    # 5. Apply SRCR-ECEM for anomaly detection
    anomaly_img_srcr = srcr_ecem_anomaly_detection(denoised_img)
    print(anomaly_img_srcr.shape)
    print(denoised_img.shape)
    plt.figure()  # Create a new figure for SRCR-ECEM
    plt.imshow(anomaly_img_srcr, cmap='jet')  # Show anomalies in a distinctive color map
    plt.title("Anomaly Detection (SRCR-ECEM)")
    plt.colorbar()
    plt.show()

    # 6. Perform RX Algorithm anomaly detection
    anomaly_img_rx = rx_algorithm(denoised_img, alpha=0.05)
    plt.figure()  # Create a new figure for RX anomaly detection
    plt.imshow(anomaly_img_rx, cmap='jet')
    plt.title("Anomaly Detection (RX Algorithm)")
    plt.colorbar()
    plt.show()

    # Return processed data
    return denoised_img, selected_bands, anomaly_img_srcr, anomaly_img_rx

# Call the main function with the path to your .mat file
file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper1.mat"  # Replace with your file path
cleaned_data = clean_and_process_data(file_path)
