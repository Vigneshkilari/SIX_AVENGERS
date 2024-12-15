import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import pywt
from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score
from finalpreprocess import clean_and_process_data



def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['img'], data['map']

def build_cnn_transformer(input_shape, num_classes=1):
    inputs = tf.keras.layers.Input(shape=input_shape)

    
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Flatten()(x)

    # Transformer Encoder
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Reshape((-1, 128))(x)  # Reshape for multi-head attention
    transformer_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    transformer_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)(transformer_out + x)
    transformer_out = tf.keras.layers.GlobalAveragePooling1D()(transformer_out)

    # Fully connected layers
    x = tf.keras.layers.Dense(64, activation='relu')(transformer_out)
    outputs = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x)

    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    print("CNN-Transformer model compiled successfully.")
    return model


def prepare_data_for_model(img_data, patch_size=32):
    height, width, bands = img_data.shape
    img_patches = []
    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = img_data[i:i+patch_size, j:j+patch_size, :]
            img_patches.append(patch)
    img_patches = np.array(img_patches)
    return img_patches

def visualize_model_output(model, img_data, patch_size=32):
    height, width, bands = img_data.shape
    anomaly_map = np.zeros((height, width))

    for i in range(0, height - patch_size + 1, patch_size):
        for j in range(0, width - patch_size + 1, patch_size):
            patch = img_data[i:i + patch_size, j:j + patch_size, :]
            patch = np.expand_dims(patch, axis=0)
            prediction = model.predict(patch, verbose=0)
            anomaly_map[i:i + patch_size, j:j + patch_size] = prediction[0, 0]


    plt.figure(figsize=(10, 8))
    plt.imshow(anomaly_map, cmap='jet')
    plt.title("Anomaly Detection (CNN + Transformer)")
    plt.colorbar()
    plt.show()
    return anomaly_map  


def srcr_ecem_anomaly_detection(img_data, lambda_s=0.01, lambda_c=0.01):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])

    normalized_data = reshaped_data / np.linalg.norm(reshaped_data, axis=1, keepdims=True)

    endmembers = np.mean(normalized_data, axis=0) 
    residuals = normalized_data - endmembers

    anomaly_scores = np.linalg.norm(residuals, axis=1) + lambda_s * np.var(residuals, axis=1)


    anomaly_img = anomaly_scores.reshape(img_data.shape[0], img_data.shape[1])

    return anomaly_img


def rx_algorithm(img_data, alpha=0.05):
    reshaped_data = img_data.reshape(-1, img_data.shape[2])
    covariance = EmpiricalCovariance().fit(reshaped_data)
    mahalanobis_dist = covariance.mahalanobis(reshaped_data - covariance.location_)
    threshold = np.percentile(mahalanobis_dist, 100 * (1 - alpha))
    anomaly_mask = mahalanobis_dist > threshold
    anomaly_img = anomaly_mask.reshape(img_data.shape[0], img_data.shape[1])
    return anomaly_img




def clean_and_process_data_with_voting(file_path):
    img_data, map_data = load_data(file_path)
    selected_img, refined_bands, selected_bands = clean_and_process_data(file_path)
    denoised_img = selected_img
    anomaly_img_srcr = srcr_ecem_anomaly_detection(denoised_img)
    anomaly_img_rx = rx_algorithm(denoised_img, alpha=0.05)

    img_patches = prepare_data_for_model(denoised_img)


    cnn_transformer_model = build_cnn_transformer(input_shape=img_patches.shape[1:])


    labels = np.random.randint(0, 2, size=(img_patches.shape[0], 1))  
    cnn_transformer_model.fit(img_patches, labels, epochs=5, batch_size=8, verbose=1)

    
    anomaly_img_cnn_transformer = visualize_model_output(cnn_transformer_model, denoised_img, patch_size=32)

    combined_anomaly_map = (anomaly_img_srcr + anomaly_img_rx + (anomaly_img_cnn_transformer > 0.5)) / 3 > 0.5


    map_data_binary = (map_data > 0).astype(int)
    srcr_accuracy = accuracy_score(map_data_binary.flatten(), anomaly_img_srcr.flatten() > 0.5)
    rx_accuracy = accuracy_score(map_data_binary.flatten(), anomaly_img_rx.flatten())
    cnn_accuracy = accuracy_score(map_data_binary.flatten(), anomaly_img_cnn_transformer.flatten() > 0.5) 
    combined_accuracy = accuracy_score(map_data_binary.flatten(), combined_anomaly_map.flatten())

    print(f"SRCR-ECEM Accuracy: {srcr_accuracy:.2f}")
    print(f"RX Algorithm Accuracy: {rx_accuracy:.2f}")
    print(f"CNN-Transformer Accuracy: {cnn_accuracy:.2f}")
    print(f"Combined Voting Classifier Accuracy: {combined_accuracy:.2f}")

    # Visualize all results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(anomaly_img_srcr, cmap='jet')
    plt.title("SRCR-ECEM Anomaly Map")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(anomaly_img_rx, cmap='jet')
    plt.title("RX Algorithm Anomaly Map")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(anomaly_img_cnn_transformer, cmap='jet')
    plt.title("CNN-Transformer Anomaly Map")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.imshow(combined_anomaly_map, cmap='jet')
    plt.title("Combined Voting Classifier Anomaly Map")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return denoised_img, selected_bands, anomaly_img_srcr, anomaly_img_rx, anomaly_img_cnn_transformer, combined_anomaly_map



file_path = r"C:/_Vignesh_N/1_six_avengers_final/Hyper_cube(1).mat"
results = clean_and_process_data_with_voting(file_path)