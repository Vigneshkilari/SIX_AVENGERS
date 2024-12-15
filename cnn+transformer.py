import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

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

# Normalize data to [0, 1] range
def normalize_data(img_data):
    scaler = MinMaxScaler()
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    normalized_data = scaler.fit_transform(reshaped_data).reshape(img_data.shape)
    print("Data normalized to [0, 1] range.")
    return normalized_data

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

# Apply Wavelet Denoising
def wavelet_denoise(img_data, wavelet='db1', level=1, threshold=0.1):
    denoised_img = np.zeros_like(img_data)
    for i in range(img_data.shape[2]):
        band_data = img_data[:, :, i]
        coeffs = pywt.wavedec2(band_data, wavelet, level=level)
        coeffs_thresholded = list(coeffs)
        for j in range(1, len(coeffs_thresholded)):
            coeffs_thresholded[j] = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeffs_thresholded[j])
        denoised_band = pywt.waverec2(coeffs_thresholded, wavelet)
        if denoised_band.shape != band_data.shape:
            min_shape = np.minimum(band_data.shape, denoised_band.shape)
            denoised_band = denoised_band[:min_shape[0], :min_shape[1]]
        denoised_img[:, :, i] = denoised_band
    print("Wavelet denoising applied.")
    return denoised_img

# CNN-Transformer Hybrid Model
def build_cnn_transformer(input_shape, num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # CNN feature extractor
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)

    # Transformer Encoder
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((-1, 128))(x)  # Reshape for multi-head attention
    transformer_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    transformer_out = layers.LayerNormalization(epsilon=1e-6)(transformer_out + x)
    transformer_out = layers.GlobalAveragePooling1D()(transformer_out)

    # Fully connected layers
    x = layers.Dense(64, activation='relu')(transformer_out)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print("CNN-Transformer model compiled successfully.")
    return model

# Prepare data for the CNN-Transformer model
def prepare_data_for_model(img_data, patch_size=32):
    height, width, bands = img_data.shape
    img_patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = img_data[i:i+patch_size, j:j+patch_size, :]
            img_patches.append(patch)
    img_patches = np.array(img_patches)
    return img_patches

# Visualize model predictions
def visualize_model_output(model, img_data, patch_size=32):
    height, width, bands = img_data.shape
    anomaly_map = np.zeros((height, width))
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = img_data[i:i+patch_size, j:j+patch_size, :]
            patch = np.expand_dims(patch, axis=0)
            prediction = model.predict(patch, verbose=0)
            anomaly_map[i:i+patch_size, j:j+patch_size] = prediction[0, 0]
    plt.figure(figsize=(10, 8))
    plt.imshow(anomaly_map, cmap='jet')
    plt.title("Anomaly Detection (CNN + Transformer)")
    plt.colorbar()
    plt.show()

# Main function
def clean_and_process_data(file_path):
    img_data, map_data = load_data(file_path)
    print("Data loaded successfully.")

    # Preprocessing
    reduced_img, selected_bands = remove_low_variance_bands(img_data)
    normalized_img = normalize_data(reduced_img)
    radiometric_img = radiometric_correction(normalized_img)
    atmos_corrected_img = atmospheric_correction(radiometric_img)
    denoised_img = wavelet_denoise(atmos_corrected_img, wavelet='db1', level=2, threshold=0.1)

    # Prepare patches for model
    img_patches = prepare_data_for_model(denoised_img)
    print(f"Extracted {img_patches.shape[0]} patches of size {img_patches.shape[1:]}")

    # Train the model
    model = build_cnn_transformer(input_shape=img_patches.shape[1:])
    labels = np.random.randint(0, 2, size=(img_patches.shape[0], 1))
    model.fit(img_patches, labels, epochs=5, batch_size=8, verbose=1)

    # Visualize results
    visualize_model_output(model, denoised_img)
    return denoised_img

# Run the pipeline
file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper1.mat"  # Replace with your file path
cleaned_data = clean_and_process_data(file_path)
