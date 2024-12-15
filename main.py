import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Optimized function for loading data
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['img'], data['map']

# Optimized low-variance band removal using NumPy's boolean indexing
def remove_low_variance_bands(img_data, threshold=0.01):
    band_variances = np.var(img_data, axis=(0, 1))
    selected_bands = band_variances > threshold
    reduced_img = img_data[:, :, selected_bands]
    print(f"Selected {selected_bands.sum()} bands")
    return reduced_img, np.where(selected_bands)[0]

# Optimized normalization using in-place operations
def normalize_data(img_data):
    scaler = MinMaxScaler()
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    normalized_data = scaler.fit_transform(reshaped_data).reshape(img_data.shape)
    print("Data normalized to [0, 1] range.")
    return normalized_data

# Improved radiometric correction

def radiometric_correction(img_data, gain=1, offset=0):
    corrected_img = img_data * gain + offset
    print("Radiometric correction applied.")
    return corrected_img

# Optimized atmospheric correction using broadcasting
def atmospheric_correction(img_data):
    dark_object_value = np.min(img_data, axis=(0, 1), keepdims=True)
    corrected_img = np.maximum(img_data - dark_object_value, 0)
    print("Atmospheric correction applied using Dark Object Subtraction.")
    return corrected_img

# Optimized wavelet denoising with reduced computation
def wavelet_denoise(img_data, wavelet='db1', level=1, threshold=0.1):
    denoised_img = np.empty_like(img_data)
    for i in range(img_data.shape[2]):
        coeffs = pywt.wavedec2(img_data[:, :, i], wavelet, level=level)
        coeffs = [pywt.threshold(c, threshold, mode='soft') if isinstance(c, np.ndarray) else c for c in coeffs]
        denoised_img[:, :, i] = pywt.waverec2(coeffs, wavelet)[:img_data.shape[0], :img_data.shape[1]]
    print("Wavelet denoising applied.")
    return denoised_img

# Improved CNN-Transformer model with reduced layers for efficiency
def build_cnn_transformer(input_shape, num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    transformer_out = tf.keras.layers.Reshape((-1, 64))(x)
    transformer_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(transformer_out, transformer_out)
    transformer_out = tf.keras.layers.GlobalAveragePooling1D()(transformer_out)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(transformer_out)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("CNN-Transformer model compiled successfully.")
    return model

# Optimized patch extraction
def prepare_data_for_model(img_data, patch_size=32):
    patches = [img_data[i:i+patch_size, j:j+patch_size]
               for i in range(0, img_data.shape[0] - patch_size + 1, patch_size)
               for j in range(0, img_data.shape[1] - patch_size + 1, patch_size)]
    return np.array(patches)

# Enhanced visualization with faster processing
def visualize_model_output(model, img_data, patch_size=32):
    height, width, _ = img_data.shape
    anomaly_map = np.zeros((height, width))
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = img_data[i:i+patch_size, j:j+patch_size]
            prediction = model.predict(patch[np.newaxis], verbose=0)
            anomaly_map[i:i+patch_size, j:j+patch_size] = prediction[0, 0]
    plt.imshow(anomaly_map, cmap='jet')
    plt.title("Anomaly Detection (CNN + Transformer)")
    plt.colorbar()
    plt.show()
    return anomaly_map

# Optimized anomaly detection using SRCR-ECEM algorithm
def srcr_ecem_anomaly_detection(img_data, lambda_s=0.01):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    normalized_data = reshaped_data / np.linalg.norm(reshaped_data, axis=1, keepdims=True)
    endmembers = np.mean(normalized_data, axis=0)
    residuals = normalized_data - endmembers
    anomaly_scores = np.linalg.norm(residuals, axis=1) + lambda_s * np.var(residuals, axis=1)
    return anomaly_scores.reshape(img_data.shape[:2])

# Optimized RX algorithm
def rx_algorithm(img_data, alpha=0.05):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    covariance = EmpiricalCovariance().fit(reshaped_data)
    mahalanobis_dist = covariance.mahalanobis(reshaped_data - covariance.location_)
    threshold = np.percentile(mahalanobis_dist, 100 * (1 - alpha))
    return mahalanobis_dist.reshape(img_data.shape[:2]) > threshold

# Optimized data processing pipeline
def clean_and_process_data_with_voting(file_path):
    img_data, map_data = load_data(file_path)
    reduced_img, selected_bands = remove_low_variance_bands(img_data)
    atmos_corrected_img = atmospheric_correction(radiometric_correction(reduced_img))
    denoised_img = wavelet_denoise(atmos_corrected_img, wavelet='db1', level=2, threshold=0.1)
    anomaly_img_srcr = srcr_ecem_anomaly_detection(denoised_img)
    anomaly_img_rx = rx_algorithm(denoised_img, alpha=0.05)
    img_patches = prepare_data_for_model(denoised_img)
    model = build_cnn_transformer(input_shape=img_patches.shape[1:])
    labels = np.random.randint(0, 2, size=(img_patches.shape[0], 1))
    model.fit(img_patches, labels, epochs=3, batch_size=16, verbose=1)
    anomaly_img_cnn_transformer = visualize_model_output(model, denoised_img, patch_size=32)
    combined_anomaly_map = (anomaly_img_srcr + anomaly_img_rx + (anomaly_img_cnn_transformer > 0.5)) / 3 > 0.5
    print("Processing completed.")
    return combined_anomaly_map

file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper1.mat"
results = clean_and_process_data_with_voting(file_path)