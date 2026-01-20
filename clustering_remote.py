
import paramiko
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
import getpass
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

#SSH Connection
hostname = '74.220.22.147'  
username = 'ade'             
remote_dir = '/mnt/data2/sample_curated/images'  
remote_csv_path_1 = '/mnt/data2/sample_curated/images/COCO_metadata.csv' 
remote_csv_path_2 = '/mnt/data2/sample_curated/images/SAFE_metadata.csv' 
PATH_COLUMN = "processed_path"

# Initialize SSH Client
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())


def read_remote_csv(sftp_client, remote_path):
    """Reads a remote CSV file into a pandas DataFrame."""
    logging.info(f"📖 Reading {remote_path}...")
    with sftp_client.open(remote_path) as f:
        # We open the remote file as a file-like object and pass it to pandas
        return pd.read_csv(f)


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

def extract_remote_features(remote_file_path, sftp_client, kernels):
    """
    Reads bytes from SFTP, decodes in memory, applies SRM filters.
    """
    try:
        # READ BYTES (The "Magic" Step)
        with sftp_client.open(remote_file_path) as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            
        # DECODE BYTES TO IMAGE
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return None
        
        # Center Crop (Standard Forensics)
        h, w = img.shape
        crop_size = min(512, h, w) 
        cy, cx = h // 2, w // 2
        img_crop = img[cy - crop_size//2 : cy + crop_size//2, 
                       cx - crop_size//2 : cx + crop_size//2]
        
        img_float = img_crop.astype(np.float32)
        features = []
        
        for k in kernels:
            residual = cv2.filter2D(img_float, -1, k)
            res_flat = residual.flatten()
            
            # Extract statistical moments
            features.append(np.var(res_flat))
            features.append(skew(res_flat))
            features.append(kurtosis(res_flat))
            
        return np.array(features)
        
    except Exception as e:
        logging.error(f"Error processing {remote_file_path}: {e}")
        return None

# --- 4. Main Execution Loop ---

kernels = get_srm_kernels()
data_features = []
filenames = []
ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}

image_paths = []
try:
    logging.info(f"🔌 Connecting to {hostname}...")
    ssh.connect(hostname, username=username)
    sftp = ssh.open_sftp()
    logging.info(f"✅ Connected to {hostname}")

    df1 = read_remote_csv(sftp, remote_csv_path_1)
    df2 = read_remote_csv(sftp, remote_csv_path_2)

    # --- Merge and Extract Paths ---
    full_df = pd.concat([df1, df2], ignore_index=True)
    logging.info(f"📊 Total rows found: {len(full_df)}")

    if PATH_COLUMN not in full_df.columns:
        logging.error(f"❌ Error: Column '{PATH_COLUMN}' not found. Available columns:")
        logging.error(full_df.columns.tolist())
    else:
        image_paths = full_df[PATH_COLUMN].dropna().unique().tolist()
        logging.info(f"✅ Successfully extracted {len(image_paths)} unique paths.")
        logging.info("\nSample paths:")
        logging.info(image_paths[:5])
        

except Exception as e:
    logging.error(f"❌ Error: {e}")
finally:
    # Close the connection after reading CSVs. The next part of the script will reconnect.
    if 'sftp' in locals() and sftp is not None:
        try:
            sftp.close()
        except Exception:
            pass
    if 'ssh' in locals():
        try:
            if ssh.get_transport() is not None:
                ssh.close()
                logging.info("🔌 SSH Connection for CSV reading closed.")
        except Exception:
            pass



try:
    # Reconnect to process the images from the CSV paths
    logging.info(f"\n🔌 Reconnecting to {hostname} to process images...")
    ssh.connect(hostname, username=username)
    sftp = ssh.open_sftp()
    logging.info(f"✅ Successfully reconnected to {hostname}")

    if not image_paths:
        logging.warning("⚠️ No image paths were extracted from CSVs. Skipping feature extraction.")
    else:
        logging.info(f"⚙️ Processing {len(image_paths)} images using paths from CSVs...")
        for i, full_remote_path in enumerate(image_paths):
            fname = full_remote_path.split('/')[-1] # Get filename for logging
            if i % 20 == 0:
                logging.info(f"  Processing {i}/{len(image_paths)}: {fname}")
            image_file_path = f"{remote_dir}/{full_remote_path.split('/')[-2]}/{full_remote_path.split('/')[-1]}"
            feats = extract_remote_features(image_file_path, sftp, kernels)

            if feats is not None:
                data_features.append(feats)
                filenames.append(fname)
except Exception as e:
    logging.error(f"❌ An error occurred during feature extraction: {e}")
finally:
    if 'sftp' in locals() and sftp is not None:
        try:
            sftp.close()
        except Exception:
            pass
    if 'ssh' in locals():
        try:
            if ssh.get_transport() is not None:
                ssh.close()
                logging.info("🔌 SSH Connection for feature extraction closed.")
        except Exception:
            pass

# --- 5. Clustering & Visualization (Same as local) ---

if len(data_features) > 0:
    X = np.array(data_features)
    
    # Normalize & Reduce
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=min(10, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_embedded = tsne.fit_transform(X_pca)
    
    # Cluster
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    # Plot
    df_viz = pd.DataFrame({'x': X_embedded[:,0], 'y': X_embedded[:,1], 'Cluster': labels, 'Filename': filenames})
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='x', y='y', hue='Cluster', palette='viridis', s=100, alpha=0.8)
    plt.title(f'SRM Noise Clusters (Remote Data Processing)')
    plt.show()
else:
    logging.warning("No features extracted.")