import streamlit as st
import pickle
import clip
import torch
import random
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path

st.set_page_config(layout="wide")

CONFIG_PATH = Path('config.pkl')

def _save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def _load(file, default=None):
    if Path(file).exists():
        with open(file, 'rb') as f:
            return pickle.load(f)
    return default

def load_config():
    return _load(CONFIG_PATH, {})

def save_config(cfg):
    _save(CONFIG_PATH, cfg)
        
def load_files(cfg):
    data_folder = Path(cfg['data_folder'])
    data_files = list(data_folder.glob('*.jpg'))
    cfg['data_files'] = data_files
    
def calculate_vectors(cfg):
    vectors = {}
    model, preprocess = clip.load('ViT-B/32', 'cpu')
    st.sidebar.write('Calculate vectors:')
    progress = st.sidebar.progress(0)
    total = len(cfg['data_files'])
    vectors = {}
    
    with torch.no_grad():
        for idx, file in enumerate(cfg['data_files']):
            image = preprocess(Image.open(file)).unsqueeze(0)
            image_features = model.encode_image(image)[0].numpy()
            progress.progress(idx / total)
            vectors[str(file)] = image_features

    _save('vectors.pkl', vectors)
        
def load_vectors():
    return _load('vectors.pkl', {})
    
def load_labels():
    return _load('labels.pkl', {})

def add_label(label, vector):
    labels = load_labels()
    if label not in labels:
        labels[label] = []
    if vector is not None:
        labels[label].append(vector)
    _save('labels.pkl', labels)

def remove_label(label):
    labels = load_labels()
    if label in labels:
        del labels[label]
    _save('labels.pkl', labels)
    

def show_label_form():
    new_label_value = st.text_input('New label')
    do_add_label = st.button('Add label')
    
    if do_add_label and new_label_value:
        add_label(new_label_value, None)
    
    labels = load_labels()
    for label, vectors in labels.items():
        with st.container():
            cols = st.columns(3)
            cols[0].write(label)
            cols[1].write(f'Vectors: {len(vectors)}')
            cols[2].button('Delete', key=f'del_{label}', on_click=lambda: remove_label(label))
    
def show_export_labels():
    import subprocess
    import tempfile
    
    do_export = st.button('Export')
    
    if do_export:
        tempdir = tempfile.mkdtemp()
        target = Path(tempdir).joinpath('vectors')
        target_tar = Path(tempdir).joinpath('vectors.tar')
        target.mkdir(parents=True)
        
        labels = load_labels()
        for label, vectors in labels.items():
            with open(target.joinpath(f'{label}.txt'), 'wt') as f:
                np.savetxt(f, np.array(vectors))
        
        subprocess.call(['tar', '-cf', target_tar, '-C', tempdir, 'vectors'])
        with open(target_tar, "rb") as file:
            st.download_button(label='Download', data=file, file_name="vectors.tar")
    

@st.cache
def calculate_kmeans_centroid():
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import euclidean_distances
    kmeans = KMeans(25)

    vectors = load_vectors()
    image_paths = list(vectors.keys())
    image_vectors = list(vectors.values())
    kmeans.fit(image_vectors)

    result = []

    for cluster_id in range(kmeans.cluster_centers_.shape[0]):
        cc = kmeans.cluster_centers_[cluster_id]
        min_idx = np.argmin(euclidean_distances(cc[np.newaxis,:], image_vectors))
        result.append((str(image_paths[min_idx]), image_vectors[min_idx]))
    return result

@st.cache(suppress_st_warning=True)
def calculate_low_prob(top = 50):
    vectors = load_vectors()
    labels = load_labels()
    
    p = st.progress(0)
    
    max_score = {}
    
    for idx, (file_path, vector) in enumerate(vectors.items()):
        score = calculate_score(vector, labels)
        p.progress(idx / len(vectors))
        max_score[str(file_path)] = np.max(list(score.values()))
    max_score = list(max_score.items())
    max_score = sorted(max_score, key=lambda x: x[1])
    return {file_path: vectors[file_path] for file_path, score in max_score}

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@st.cache
def flatten_labels(labels):
    label_arr, vectors_arr = [], []
    for l, vectors in labels.items():
        for v in vectors:
            label_arr.append(l)
            vectors_arr.append(v)
    return np.array(label_arr), np.array(vectors_arr)

def euclidean_distances(vector, points):
    return np.array([np.linalg.norm(vector-point) for point in points])

def calculate_score(vector, labels):
    label_arr, vectors_arr = flatten_labels(labels)
    
    distances = -euclidean_distances(vector[np.newaxis,:], vectors_arr)
    prob = softmax(distances)
    score = {}
    for label in labels.keys():
        label_score = (prob * (label_arr == label)).sum()
        score[label] = label_score
    return score

def show_image(c, labels, image_path, vector):
    c.image(image_path)
    
    score = calculate_score(vector, labels)
    
    df = pd.DataFrame.from_dict(score, orient='index', columns=['prob'])
    df = df.sort_values('prob', ascending=False)
    df = df.style.format(precision=3)
    c.dataframe(df)
    
    labels_for_select = [''] + list(labels.keys())
    
    selected_label = c.selectbox('Label', key=f'label_{image_path}', options=labels_for_select)
    if selected_label:
        c.button('Set label', key=f'do_set_label_{image_path}', on_click=lambda: add_label(selected_label, vector))

    
def show_kmeans(no_columns = 4):
    columns = st.columns(no_columns)
    centroids = calculate_kmeans_centroid()
    labels = load_labels()
    
    for idx, (file_path, vector) in enumerate(centroids):
        c = columns[idx % no_columns].container()
        show_image(c, labels, file_path, vector)


def show_low_prob(no_columns = 4):
    columns = st.columns(no_columns)
    low_prob_score = calculate_low_prob()
    offset = 0
    columns = st.columns(no_columns)
    labels = load_labels()
    
    for idx, (file_path, vector) in enumerate(list(low_prob_score.items())[offset:offset+16]):
        c = columns[idx % no_columns].container()
        show_image(c, labels, file_path, vector)
        
cfg = load_config()


data_files = cfg.get('data_files', [])
labels = load_labels()

if len(data_files) > 0:
    with st.expander('KMeans'):
        show_kmeans()
    with st.expander('Low probability'):
        show_low_prob()
        
st.sidebar.write('Data folder')
cfg_data_folder = st.sidebar.text_input('Data folder', cfg.get('data_folder', ''))

st.sidebar.write(f'Files: {len(cfg.get("data_files", []))}')

do_update = st.sidebar.button('Update')
if do_update:
    cfg['data_folder'] = cfg_data_folder
    load_files(cfg)
    calculate_vectors(cfg)
    save_config(cfg)

with st.sidebar:
    show_label_form()
    show_export_labels()