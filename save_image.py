import numpy as np

def save_as_ppm(filename, image_data):
    """Save a NumPy array as a PPM image (plain text format)."""
    if len(image_data.shape) != 2:
        raise ValueError("This function supports only 2D grayscale images.")

    # Normalize data to 0-255 and convert to uint8
    image_data = (255 * (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))).astype(np.uint8)

    height, width = image_data.shape

    # Create PPM header
    header = f"P2\n{width} {height}\n255\n"

    # Write header and pixel data
    with open(filename, 'w') as f:
        f.write(header)
        for row in image_data:
            f.write(' '.join(map(str, row)) + '\n')

    print(f"Image saved as {filename}")

# Example usage
if __name__ == "__main__":
    # Example random grayscale image
    anomaly_img_rx = np.random.random((256, 256))  # Random data between 0 and 1
    save_as_ppm("C:/_Vignesh_N/1_six_avengers_final/data_npanomaly_image.ppm", anomaly_img_rx)
