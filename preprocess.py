import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


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
            # Option 1: Crop the denoised band
            min_shape = np.minimum(band_data.shape, denoised_band.shape)
            denoised_band = denoised_band[:min_shape[0], :min_shape[1]]

        # Store the denoised band
        denoised_img[:, :, i] = denoised_band

    return denoised_img


# Radiometric Correction (example using a gain and offset method)
def radiometric_correction(img_data, gain=1, offset=0):
    corrected_img = (img_data * gain) + offset
    print("Radiometric correction applied.")
    return corrected_img


# Atmospheric Correction using Dark Object Subtraction (DOS)
def atmospheric_correction(img_data):
    # Dark Object Subtraction (DOS): assume the darkest value in each band is caused by atmospheric scattering
    dark_object_value = np.min(img_data, axis=(0, 1))  # Minimum value per band
    corrected_img = img_data - dark_object_value
    corrected_img[corrected_img < 0] = 0  # Ensure no negative values
    print("Atmospheric correction applied using Dark Object Subtraction.")
    return corrected_img


# Calculate and plot NDVI for a given hyperspectral image
def calculate_ndvi(img_data, red_band=50, nir_band=100):
    red = img_data[:, :, red_band]
    nir = img_data[:, :, nir_band]
    ndvi = (nir - red) / (nir + red)
    return ndvi


# Apply PCA for dimensionality reduction
def apply_pca(img_data, n_components=10):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(reshaped_data)
    print(f"PCA Components explained variance ratio: {pca.explained_variance_ratio_}")
    pca_img = pca.inverse_transform(pca_data).reshape(img_data.shape)
    return pca_img


# Perform KMeans clustering for vegetation or land use classification
def perform_kmeans(img_data, n_clusters=5):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(reshaped_data)
    clustered_img = labels.reshape(img_data.shape[0], img_data.shape[1])
    return clustered_img


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

    # 5. Calculate NDVI
    ndvi = calculate_ndvi(denoised_img)
    plt.figure()  # Create a new figure for NDVI
    plt.imshow(ndvi, cmap='hot')
    plt.title("NDVI Image")
    plt.colorbar()
    plt.show()

    # 6. Apply PCA
    pca_img = apply_pca(denoised_img)
    plt.figure()  # Create a new figure for PCA
    plt.imshow(pca_img[:, :, 0], cmap='gray')  # Show the first principal component
    plt.title("PCA First Component")
    plt.colorbar()
    plt.show()

    # 7. Perform KMeans clustering
    clustered_img = perform_kmeans(denoised_img)
    plt.figure()  # Create a new figure for clustering
    plt.imshow(clustered_img, cmap='tab20b')
    plt.title("KMeans Clustering Result")
    plt.colorbar()
    plt.show()

    # Return cleaned and processed data for further analysis
    return denoised_img, selected_bands, ndvi, pca_img, clustered_img


# Call the main function with the path to your .mat file
file_path = r"C:\Users\91637\Pictures\Saved Pictures\Hyper1.mat"  # Replace with your file path
cleaned_data = clean_and_process_data(file_path)