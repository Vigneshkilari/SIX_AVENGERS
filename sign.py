import numpy as np
import time
import cv2
from pre_numpy import clean_and_process_data
from mat_npy import load_data_from_npy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

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
    covariance = np.cov(reshaped_data, rowvar=False)
    inv_covariance = np.linalg.inv(covariance)
    mahalanobis_dist = np.sqrt(np.sum(np.dot(reshaped_data - np.mean(reshaped_data, axis=0), inv_covariance) * (reshaped_data - np.mean(reshaped_data, axis=0)), axis=1))
    threshold = np.percentile(mahalanobis_dist, 100 * (1 - alpha))
    anomaly_mask = mahalanobis_dist > threshold
    anomaly_img = anomaly_mask.reshape(img_data.shape[0], img_data.shape[1])
    return anomaly_img

def build_cnn_classifier(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def classify_anomalies_with_cnn(anomaly_regions, model):
    predictions = []
    target_size = (32, 32)  # Define the target size for CNN input
    for region in anomaly_regions:
        # Resize each anomaly region to the target size
        resized_region = cv2.resize(region, target_size)
        resized_region = resized_region.reshape(1, *resized_region.shape, 1)  # Add batch and channel dimensions
        predictions.append(np.argmax(model.predict(resized_region), axis=1)[0])
    return predictions

def visualize_results(img, anomaly_map, classifications, display_rgb=True):
    plt.figure(figsize=(10, 5))

    if display_rgb:
        # Combine the first three bands for RGB visualization
        rgb_img = np.stack((img[:, :, 0], img[:, :, 1], img[:, :, 2]), axis=-1)  # Combine first 3 bands
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.title("Original Image (RGB)")
    else:
        # Choose a single band (e.g., band 0) for grayscale visualization
        grayscale_img = img[:, :, 0]  # You can choose a different band (e.g., 1, 2, etc.)
        plt.subplot(1, 2, 1)
        plt.imshow(grayscale_img, cmap='gray')
        plt.title("Original Image (Band 0)")

    plt.subplot(1, 2, 2)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.6)  # Show anomaly map with transparency
    
    for idx, classification in enumerate(classifications):
        y, x = np.where(anomaly_map == idx + 1)  # Mark the anomaly class regions
        plt.scatter(x, y, label=f"Class {classification}", alpha=0.7)
    
    plt.title("Target Classification")
    plt.legend()
    plt.show()

def clean_and_process_data_with_voting(file_path):
    start_time = time.time()  # Start timing
    
    img_data, map_data = load_data_from_npy()
    selected_img, refined_bands, selected_bands = clean_and_process_data(file_path)
    denoised_img = selected_img
    anomaly_img_srcr = srcr_ecem_anomaly_detection(denoised_img)
    anomaly_img_rx = rx_algorithm(denoised_img, alpha=0.05)
    
    # Combine the anomaly detection maps using simple averaging
    combined_anomaly_map = (anomaly_img_srcr + anomaly_img_rx) / 2 > 0.5  # Adjust threshold as needed
    combined_anomaly_map = combined_anomaly_map.astype(int)
    
    # Extract anomaly regions for CNN classification
    anomaly_regions = []
    for label in np.unique(combined_anomaly_map):
        if label == 0:
            continue  # Skip non-anomalous regions
        anomaly_region = denoised_img[combined_anomaly_map == label]
        anomaly_regions.append(anomaly_region)
    
    # Build and train the CNN
    cnn_model = build_cnn_classifier(input_shape=(32, 32, 1), num_classes=5)  # Adjust classes
    # Assume anomaly_regions have been split into training data
    classifications = classify_anomalies_with_cnn(anomaly_regions, cnn_model)
    
    # Visualize the results
    visualize_results(denoised_img, combined_anomaly_map, classifications, display_rgb=True)  # Set display_rgb=False for grayscale
    
    map_data_binary = (map_data > 0).astype(int)
    srcr_accuracy = np.mean(map_data_binary.flatten() == (anomaly_img_srcr.flatten() > 0.5))
    rx_accuracy = np.mean(map_data_binary.flatten() == anomaly_img_rx.flatten())
    combined_accuracy = np.mean(map_data_binary.flatten() == combined_anomaly_map.flatten())
    
    print(f"SRCR-ECEM Accuracy: {srcr_accuracy:.2f}")
    print(f"RX Algorithm Accuracy: {rx_accuracy:.2f}")
    print(f"Combined Voting Classifier Accuracy: {combined_accuracy:.2f}")
    
    end_time = time.time()
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
    
    return denoised_img, selected_bands, anomaly_img_srcr, anomaly_img_rx, combined_anomaly_map, classifications


# Example call to the function
file_path = "C:/_Vignesh_N/1_six_avengers_final/data_np"  # Update with your file path
results = clean_and_process_data_with_voting(file_path)
