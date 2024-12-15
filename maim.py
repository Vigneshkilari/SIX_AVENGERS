import numpy as np
import scipy.io
from sklearn.decomposition import IncrementalPCA
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import StandardScaler

def align_shapes(*arrays):
    min_rows = min(array.shape[0] for array in arrays)
    min_cols = min(array.shape[1] for array in arrays)
    aligned_arrays = tuple(array[:min_rows, :min_cols] for array in arrays)
    return aligned_arrays

def load_data(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        if 'img' not in data or 'map' not in data:
            raise KeyError("The .mat file must contain 'img' and 'map'.")
        return align_shapes(data['img'], data['map'])
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

# Band selection using Mutual Information
def mutual_info_band_selection(img_data, labels, n_bands=10):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    mi_scores = mutual_info_classif(reshaped_data, labels)
    top_band_indices = np.argsort(mi_scores)[-n_bands:]
    return img_data[:, :, top_band_indices], top_band_indices

# Remove low-variance bands
def remove_low_variance_bands(img_data, threshold=0.01):
    band_variances = np.var(img_data, axis=(0, 1))
    selected_bands = np.where(band_variances > threshold)[0]
    reduced_img = img_data[:, :, selected_bands]
    print(f"Selected {len(selected_bands)} bands after removing low-variance bands.")
    return reduced_img, selected_bands

# Denoising with wavelet
def wavelet_denoise_parallel(img_data, wavelet='db1', level=1, threshold=0.1):
    def denoise_band(band_data):
        coeffs = pywt.wavedec2(band_data, wavelet, level=level)
        coeffs_thresholded = [
            tuple(pywt.threshold(c, threshold, mode='soft') for c in details) if i > 0 else details
            for i, details in enumerate(coeffs)
        ]
        return pywt.waverec2(coeffs_thresholded, wavelet)

    denoised_bands = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(denoise_band, [img_data[:, :, i] for i in range(img_data.shape[2])])
        for denoised_band in results:
            aligned_band = denoised_band[:img_data.shape[0], :img_data.shape[1]]
            denoised_bands.append(aligned_band)

    denoised_img = np.stack(denoised_bands, axis=-1)
    return denoised_img

# Apply Incremental PCA
def apply_incremental_pca(img_data, n_components=10, batch_size=1000):
    rows, cols, bands = img_data.shape
    reshaped_data = img_data.reshape(-1, bands).astype(np.float32)

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca_data = ipca.fit_transform(reshaped_data)
    pca_img = pca_data.reshape(rows, cols, n_components)
    return pca_img

# Signal-to-Noise Ratio (SNR)
def calculate_snr(img_data):
    mean_signal = np.mean(img_data, axis=(0, 1))
    noise = img_data - mean_signal
    snr = mean_signal / (np.std(noise, axis=(0, 1)) + 1e-10)  # Added small constant to avoid division by zero
    return snr

# RX Anomaly Detection (using One-Class SVM)
def rx_anomaly_detection(img_data, nu=0.1, kernel='rbf', gamma='auto'): # Tunable parameters
    rows, cols, bands = img_data.shape
    reshaped_data = img_data.reshape(-1, bands)
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(reshaped_data)
    anomalies = ocsvm.predict(reshaped_data) == -1  # -1 indicates anomaly
    return anomalies.reshape(rows, cols)



# Main cleaning and processing function
def clean_and_process_data(file_path):
    img_data, map_data = load_data(file_path)
    reduced_img, _ = remove_low_variance_bands(img_data)
    pca_img = apply_incremental_pca(reduced_img)
    denoised_img = wavelet_denoise_parallel(pca_img)
    snr = calculate_snr(denoised_img)
    anomalies_rx = rx_anomaly_detection(denoised_img)


    return denoised_img, snr, anomalies_rx


file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper_cube(1).mat" # Replace with your file path

denoised_img, snr, anomalies_rx = clean_and_process_data(file_path)



plt.imshow(anomalies_rx, cmap='gray')
plt.title("RX Anomalies")
plt.show()