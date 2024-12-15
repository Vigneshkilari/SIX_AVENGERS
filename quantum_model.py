import warnings
warnings.filterwarnings("ignore")

# Importing libraries
import numpy as np
import pennylane as qml
from pennylane.templates import AngleEmbedding
from pennylane.optimize import AdamOptimizer
import matplotlib.pyplot as plt
import scipy.io
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from finalpreprocess import clean_and_process_data

# Configuration
n_qubits = 12  # Number of qubits for QAE
latent_qubits = 6  # Latent space qubits
batch_size = 50  # Batch size for training
n_epochs = 1  # Number of epochs
learning_rate = 0.01  # Learning rate for optimizer

# Device setup
dev = qml.device("default.qubit", wires=n_qubits)

# Define the Quantum Autoencoder
@qml.qnode(dev)
def quantum_autoencoder(inputs, encoder_weights, decoder_weights):
    # Encoding
    AngleEmbedding(inputs, wires=range(n_qubits))
    for i in range(n_qubits):
        qml.RX(encoder_weights[i], wires=i)
        qml.RY(encoder_weights[i + n_qubits], wires=i)
    
    # Latent space compression
    for i in range(latent_qubits, n_qubits):
        qml.CNOT(wires=[i, i % latent_qubits])
        qml.Hadamard(wires=i)
    
    # Decoding
    for i in range(n_qubits):
        qml.RY(decoder_weights[i], wires=i)
        qml.RX(decoder_weights[i + n_qubits], wires=i)
    
    return qml.probs(wires=range(n_qubits))

# Reconstruction loss function
def reconstruction_loss(inputs, encoder_weights, decoder_weights):
    reconstructed = quantum_autoencoder(inputs, encoder_weights, decoder_weights)
    return np.sum((inputs - reconstructed[:len(inputs)]) ** 2)

# Load dataset
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['img']

# Anomaly detection threshold calculation
def calculate_threshold(errors):
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error + 3 * std_error  # 3-sigma rule

# Load and preprocess data
file_path = "C:/_Vignesh_N/1_six_avengers_final/Hyper1.mat"
selected_img, refined_bands, _ = clean_and_process_data(file_path)
X = selected_img
n_rows, n_cols, n_slices = X.shape

# Get the total number of elements in the original array
total_elements = np.prod(X.shape)

# Adjust n_cols dynamically to match the total number of elements
n_slices = 10  # Example: You want 10 slices
n_rows = X.shape[0]
n_cols = total_elements // (n_rows * n_slices)

# Reshape the data accordingly
X_scaled = X.reshape(n_rows, n_cols, n_slices)

# Initialize weights
encoder_weights = np.random.uniform(-0.1, 0.1, 2 * n_qubits)
decoder_weights = np.random.uniform(-0.1, 0.1, 2 * n_qubits)

# Placeholder for results
reconstructed_data = np.zeros_like(X_scaled)
reconstruction_errors_all = np.zeros((n_rows, n_slices))
anomalies_all = np.zeros((n_rows, n_slices), dtype=bool)

# Function to train and process each slice
def process_slice(slice_idx, encoder_weights, decoder_weights):
    X_slice = X_scaled[:, :, slice_idx]
    reconstructed_slice = np.zeros_like(X_slice)
    reconstruction_errors = np.zeros(n_rows)
    
    tqdm.write(f"Processing slice {slice_idx + 1}/{n_slices}")  # Thread-safe print
    
    # Training loop
    opt = AdamOptimizer(learning_rate)
    for epoch in range(n_epochs):
        total_loss = 0
        for start_idx in range(0, n_rows, batch_size):
            end_idx = min(start_idx + batch_size, n_rows)
            batch = X_slice[start_idx:end_idx, :n_qubits]
            
            for row_idx in range(len(batch)):
                inputs = batch[row_idx]
                
                def cost(weights):
                    enc_weights, dec_weights = weights[:len(encoder_weights)], weights[len(encoder_weights):]
                    return reconstruction_loss(inputs, enc_weights, dec_weights)
                
                params = np.concatenate([encoder_weights, decoder_weights])
                params = opt.step(cost, params)
                encoder_weights, decoder_weights = params[:len(encoder_weights)], params[len(encoder_weights):]
                total_loss += cost(params)
        
        tqdm.write(f"Epoch {epoch + 1}, Slice {slice_idx + 1}, Total Loss: {total_loss:.6f}")

    # Calculate reconstruction errors and detect anomalies
    for row_idx in range(n_rows):
        inputs = X_slice[row_idx, :n_qubits]
        reconstructed_probs = quantum_autoencoder(inputs, encoder_weights, decoder_weights)
        reconstructed_slice[row_idx, :n_qubits] = reconstructed_probs[:n_qubits]
        reconstruction_errors[row_idx] = np.sum((inputs - reconstructed_probs[:n_qubits]) ** 2)
    
    threshold = calculate_threshold(reconstruction_errors)
    anomalies = reconstruction_errors > threshold
    
    # Store results
    reconstructed_data[:, :, slice_idx] = reconstructed_slice
    reconstruction_errors_all[:, slice_idx] = reconstruction_errors
    anomalies_all[:, slice_idx] = anomalies

    tqdm.write(f"Slice {slice_idx + 1}: Threshold = {threshold:.6f}, Anomalies Detected = {np.sum(anomalies)}")
    return encoder_weights, decoder_weights

# Use ThreadPoolExecutor to process slices in parallel with tqdm
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust 'max_workers' as per CPU cores
    with tqdm(total=n_slices, desc="Processing slices") as pbar:
        for slice_idx in range(n_slices):
            executor.submit(process_slice, slice_idx, encoder_weights, decoder_weights)
            pbar.update(1)

# Visualization for the first slice (example)
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(X_scaled[:, :, 0], cmap="gray", aspect="auto")
plt.title("Original First Slice")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(reconstructed_data[:, :, 0], cmap="gray", aspect="auto")
plt.title("Reconstructed First Slice")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.plot(reconstruction_errors_all[:, 0], label="Reconstruction Error")
plt.axhline(y=calculate_threshold(reconstruction_errors_all[:, 0]), color="r", linestyle="--", label="Threshold")
plt.title("Reconstruction Errors & Anomalies (First Slice)")
plt.legend()

plt.tight_layout()
plt.show()

# Display overall metrics
total_anomalies = np.sum(anomalies_all)
print(f"Total Anomalies Detected Across All Slices: {total_anomalies}")
