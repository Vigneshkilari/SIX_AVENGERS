import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from sklearn.feature_selection import mutual_info_classif
import pywt
from concurrent.futures import ThreadPoolExecutor

# Load and align data
def load_data(file_path):
    try:
        data = scipy.io.loadmat(file_path)
        if 'img' not in data or 'map' not in data:
            raise KeyError("The .mat file must contain 'img' and 'map'.")
        return data['img'], data['map']
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

# Remove low variance bands (increased threshold for stricter selection)
def remove_low_variance_bands(img_data, threshold=0.01):
    band_variances = np.var(img_data, axis=(0, 1))
    plt.plot(band_variances)
    plt.title("Band Variances")
    plt.xlabel("Band Index")
    plt.ylabel("Variance")
    plt.show()

    selected_bands = np.where(band_variances > threshold)[0]
    print(f"Selected {len(selected_bands)} bands after variance filtering.")
    reduced_img = img_data[:, :, selected_bands] if len(selected_bands) > 0 else img_data
    return reduced_img, selected_bands

# Apply Incremental PCA for dimensionality reduction
def apply_incremental_pca(img_data, n_components=10, batch_size=1000):
    rows, cols, bands = img_data.shape
    reshaped_data = img_data.reshape(-1, bands).astype(np.float32)

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    pca_data = ipca.fit_transform(reshaped_data)
    pca_img = pca_data.reshape(rows, cols, n_components)
    return pca_img

# Calculate Signal-to-Noise Ratio (SNR)
def calculate_snr(img_data):
    mean_signal = np.mean(img_data, axis=(0, 1))
    noise = img_data - mean_signal
    snr = mean_signal / (np.std(noise, axis=(0, 1)) + 1e-10)
    return snr

# Wavelet denoising using parallel processing
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

# Apply mutual information for band selection
def mutual_info_band_selection(img_data, labels, n_bands=10):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    mi_scores = mutual_info_classif(reshaped_data, labels)
    top_band_indices = np.argsort(mi_scores)[-n_bands:]
    return img_data[:, :, top_band_indices], top_band_indices

# Main function to clean and process data
def clean_and_process_data(file_path):
    try:
        img_data, map_data = load_data(file_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    try:
        reduced_img, selected_bands = remove_low_variance_bands(img_data, threshold=0.01)  # Increased threshold
    except Exception as e:
        print(f"Error during low variance band removal: {e}")
        return

    # Visualize reduced bands
    print(f"Remaining bands after low variance removal: {len(selected_bands)}")
    plt.imshow(reduced_img[:, :, 0], cmap='gray')  # Display the first band
    plt.title("First Band After Variance Filtering")
    plt.show()

    try:
        pca_img = apply_incremental_pca(reduced_img, n_components=10, batch_size=1000)
        print("Incremental PCA applied.")
    except Exception as e:
        print(f"Error during PCA: {e}")
        return

    try:
        denoised_img = wavelet_denoise_parallel(pca_img, wavelet='db2', level=2, threshold=0.2)
        print("Wavelet denoising applied.")
    except Exception as e:
        print(f"Error during wavelet denoising: {e}")
        return

    try:
        snr = calculate_snr(denoised_img)
        print(f"SNR values: {snr}")
    except Exception as e:
        print(f"Error during SNR calculation: {e}")
        return

    print("Data cleaning and processing completed successfully.")
    return denoised_img, selected_bands, snr

# Run the function with a sample file path (use your actual file path)
file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper_cube(1).mat"
denoised_img, selected_bands, snr = clean_and_process_data(file_path)

# Plot some results for visualization
if denoised_img is not None:
    plt.imshow(denoised_img[:, :, 0], cmap='gray')  # Display the first denoised band
    plt.title("First Denoised Band")
    plt.show()
    
    print(f"Selected Bands: {selected_bands[:5]}")  # Print only the first 5 selected bands
    print(f"SNR: {snr[:5]}")  # Display SNR for the first 5 bands for brevity