
import numpy as np
import cv2
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


IMAGE_DATA_ROOT = os.getenv('DATA_ROOT', '/data/images')

OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'output')

# --- NEW: Paths to your metadata CSV files ---
METADATA_CSV_PATH_1 = os.getenv('METADATA_CSV_1', 'COCO_metadata.csv')
METADATA_CSV_PATH_2 = os.getenv('METADATA_CSV_2', 'SAFE_metadata.csv')


def get_srm_kernels():
    kernels = []

    # 1st order
    k1 = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    k2 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
    # 2nd order
    k3 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # 3rd order
    k4 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32)
    # 5x5
    k5 = np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], 
                   [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]], dtype=np.float32) / 12.0
    kernels.extend([k1, k2, k3, k4, k5])
    return kernels

# --- 3. Remote Feature Extractor ---

def extract_local_features(image_path, kernels, use_cuda):
    """
    Reads image from disk, decodes, applies SRM filters using GPU if available.
    """
    try:
        # Read image from disk
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logging.warning(f"Could not read or decode image: {image_path}")
            return None

        # Center Crop (Standard Forensics)
        h, w = img.shape
        crop_size = min(512, h, w)
        cy, cx = h // 2, w // 2
        img_crop = img[cy - crop_size // 2: cy + crop_size // 2,
                   cx - crop_size // 2: cx + crop_size // 2]

        img_float = img_crop.astype(np.float32)
        features = []

        if use_cuda:
            # Upload image to GPU once
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img_float)

        for k in kernels:
            if use_cuda:
                # Create a linear filter for GPU and apply it
                srm_filter = cv2.cuda.createLinearFilter(cv2.CV_32F, cv2.CV_32F, k)
                gpu_residual = srm_filter.apply(gpu_img)
                residual = gpu_residual.download()
            else:
                # Use CPU-based filtering
                residual = cv2.filter2D(img_float, -1, k)
            res_flat = residual.flatten()
            # Extract statistical moments
            features.extend([
                np.var(res_flat),
                skew(res_flat),
                kurtosis(res_flat)
            ])

        return np.array(features)
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None

def main():
    """Main function to run the clustering pipeline."""
    # --- 1. Setup and GPU Check ---
    logging.info("Starting feature extraction and clustering process.")
    use_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_cuda:
        logging.info("✅ CUDA is available. Using GPU for acceleration.")
    else:
        logging.warning("⚠️ CUDA not available. Falling back to CPU. Processing will be slower.")

    kernels = get_srm_kernels()

    # --- NEW: Load Model Metadata from CSVs ---
    try:
        logging.info(f"📖 Reading metadata from {METADATA_CSV_PATH_1} and {METADATA_CSV_PATH_2}...")
        df1 = pd.read_csv(METADATA_CSV_PATH_1)
        df2 = pd.read_csv(METADATA_CSV_PATH_2)
        metadata_df = pd.concat([df1, df2], ignore_index=True)

        metadata_df.columns = [col.lower().replace(' ', '_') for col in metadata_df.columns]

        if 'file_name' not in metadata_df.columns or 'model' not in metadata_df.columns:
            logging.error("❌ Error: CSVs must contain 'file_name' and 'model' columns.")
            return

        filename_to_model_map = pd.Series(metadata_df.model.values, index=metadata_df.file_name).to_dict()
        logging.info(f"✅ Successfully loaded metadata for {len(filename_to_model_map)} files.")

    except FileNotFoundError as e:
        logging.error(f"❌ Error reading metadata CSV file: {e}. Make sure paths are correct and mapped into the container if needed.")
        return
    except Exception as e:
        logging.error(f"❌ An error occurred during metadata loading: {e}")
        return

    # --- 2. Load Data Paths by scanning a directory ---
    image_paths = []
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'}
    logging.info(f"🔍 Scanning for images in '{IMAGE_DATA_ROOT}'...")
    try:
        if not os.path.isdir(IMAGE_DATA_ROOT):
            logging.error(f"❌ Error: The directory '{IMAGE_DATA_ROOT}' was not found. Please set this variable to your data directory.")
            return

        for root, dirnames, files in os.walk(IMAGE_DATA_ROOT):
            if 'originals' in dirnames:
                dirnames.remove('originals')
            for file in files:
                if os.path.splitext(file)[1].lower() in allowed_extensions:
                    image_paths.append(os.path.join(root, file))

        logging.info(f"✅ Found {len(image_paths)} images.")
        if not image_paths:
            logging.warning("⚠️ No images found in the specified directory. Exiting.")
            return
        logging.info("Sample paths:")
        for path in image_paths[:5]:
            logging.info(f"  - {path}")
    except Exception as e:
        logging.error(f"❌ An error occurred while scanning for images: {e}")
        return

    # --- 3. Feature Extraction ---
    data_features = []
    filenames = []
    image_models = []
    logging.info(f"⚙️ Processing {len(image_paths)} images...")
    for i, image_path in enumerate(image_paths):
        fname = os.path.basename(image_path)
        if (i + 1) % 200 == 0:
            logging.info(f"  Processed {i + 1}/{len(image_paths)}: {fname}")

        model_name = filename_to_model_map.get(fname)
        if model_name is None:
            logging.warning(f"⚠️ No model found for '{fname}' in metadata, skipping.")
            continue

        feats = extract_local_features(image_path, kernels, use_cuda)

        if feats is not None:
            data_features.append(feats)
            filenames.append(fname)
            image_models.append(model_name)

    if not data_features:
        logging.warning("No features were extracted. Cannot proceed with clustering.")
        return

    # --- 4. Clustering & Visualization ---
    logging.info(f"✅ Feature extraction complete. Extracted features for {len(data_features)} images.")
    logging.info("🔬 Starting dimensionality reduction and visualization by model...")

    # Ensure the output directory exists before saving plots.
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = np.array(data_features)

    # Normalize & Reduce
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Perplexity must be less than n_samples
    perplexity = min(30, len(X) - 1)
    if perplexity <= 0:
        logging.error("Not enough data points to perform t-SNE. Need at least 2.")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_embedded = tsne.fit_transform(X_pca)

    # Create a DataFrame for visualization
    df_viz = pd.DataFrame({
        'x': X_embedded[:, 0],
        'y': X_embedded[:, 1],
        'Model': image_models,
        'Filename': filenames
    })

    # Plotting
    plt.figure(figsize=(14, 10))
    sns.scatterplot(data=df_viz, x='x', y='y', hue='Model', palette='tab20', s=80, alpha=0.7)
    plt.title('t-SNE Visualization of Image Features by Model')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout() # Adjust plot to ensure everything fits

    # Save the plot
    output_filename = 'srm_features_by_model.png'
    output_filepath = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_filepath)
    logging.info(f"✅ Visualization plot saved to '{output_filepath}'")
    plt.close()

    # --- 5. Algorithmic Clustering (KMeans) ---
    logging.info("🔬 Now performing KMeans clustering for k=3 to 33...")
    clusters_to_test = range(3, 34)

    for cluster_count in clusters_to_test:
        # Cluster using KMeans. n_init is set to avoid a future warning.
        kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Add the new cluster labels to our visualization DataFrame
        df_viz['Cluster'] = labels

        # Plot
        # Plotting with colors by source directory and shapes by cluster
        plt.figure(figsize=(14, 10))
        sns.scatterplot(data=df_viz, x='x', y='y', hue='Model', style='Cluster', palette='tab20', s=80, alpha=0.7)
        plt.title(f'KMeans Clusters (k={cluster_count}) with Model Colors')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        output_filename = f'srm_clusters_k{cluster_count}.png'
        output_filepath = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_filepath)
        logging.info(f"✅ Clustering plot saved to '{output_filepath}'")
        plt.close()  # Free up memory by closing the figure

if __name__ == "__main__":
    main()