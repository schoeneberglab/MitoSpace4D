import os
import numpy as np

emb_dir_1 = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings"
emb_dir_2 = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings"
# emb_dir_3 = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_3"
# emb_dir_4 = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings_4"
emb_dir_combined = "/home/dhruvagarwal/projects/MitoSpace4D/runs/lightning_logs/resnetbilstm_encoded_normal/embeddings"

os.makedirs(emb_dir_combined, exist_ok=True)

emb_1 = np.load(os.path.join(emb_dir_1, "embeddings.npy"))
emb_2 = np.load(os.path.join(emb_dir_2, "embeddings_static_oligo.npy"))
# emb_3 = np.load(os.path.join(emb_dir_3, "embeddings.npy"))
# emb_4 = np.load(os.path.join(emb_dir_4, "embeddings.npy"))
labels_1 = np.load(os.path.join(emb_dir_1, "labels.npy"))
labels_2 = np.load(os.path.join(emb_dir_2, "labels_static_oligo.npy"))
# labels_3 = np.load(os.path.join(emb_dir_3, "labels.npy")) * 0 + 6
# labels_4 = np.load(os.path.join(emb_dir_4, "labels.npy")) * 0 + 14
label_names = np.load(os.path.join(emb_dir_1, "label_names.npy"))

emb_combined = np.concatenate([emb_1, emb_2])
labels_combined = np.concatenate([labels_1, labels_2])

np.save(os.path.join(emb_dir_combined, "embeddings_static_oligo.npy"), emb_combined)
np.save(os.path.join(emb_dir_combined, "labels_static_oligo.npy"), labels_combined)
np.save(os.path.join(emb_dir_combined, "label_names.npy"), label_names)
