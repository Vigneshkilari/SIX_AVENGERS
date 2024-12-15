import numpy as np
import time
import os
from pre_numpy import clean_and_process_data
from mat_npy import load_data_from_npy

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

def clean_and_process_data_with_voting(file_path):
    start_time = time.time()  # Start timing
    
    img_data, map_data = load_data_from_npy()
    selected_img, refined_bands, selected_bands = clean_and_process_data(file_path)
    denoised_img = selected_img
    anomaly_img_srcr = srcr_ecem_anomaly_detection(denoised_img)
    anomaly_img_rx = rx_algorithm(denoised_img, alpha=0.05)
    
    # Combining anomaly maps - Simple Averaging and Thresholding
    combined_anomaly_map = (anomaly_img_srcr + anomaly_img_rx) / 2 > 0.5  # Adjust threshold as needed

    map_data_binary = (map_data > 0).astype(int)
    srcr_accuracy = np.mean(map_data_binary.flatten() == (anomaly_img_srcr.flatten() > 0.5))
    rx_accuracy = np.mean(map_data_binary.flatten() == anomaly_img_rx.flatten())
    combined_accuracy = np.mean(map_data_binary.flatten() == combined_anomaly_map.flatten())
    
    print(f"SRCR-ECEM Accuracy: {srcr_accuracy:.2f}")
    print(f"RX Algorithm Accuracy: {rx_accuracy:.2f}")
    print(f"Combined Voting Classifier Accuracy: {combined_accuracy:.2f}")
    
    end_time = time.time()
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
    
    return denoised_img, selected_bands, anomaly_img_srcr, anomaly_img_rx, combined_anomaly_map


# Example call to the function
file_path = "C:/_Vignesh_N/1_six_avengers_final/data_np"  # Update with your file path
results = clean_and_process_data_with_voting(file_path)