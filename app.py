import cherrypy
import clip
import yaml
import torch
import copy
import tempfile
import requests
import numpy as np

from pathlib import Path

from PIL import Image

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def euclidean_distances(vector, points):
    return np.array([np.linalg.norm(vector-point) for point in points])

class ClipService:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(config['model'], self.device)
        self.init_categories(config)
        self.init_vectors(config)
    
    def init_categories(self, config):
        self.categories = []
        for category in config['categories']:
            category = copy.deepcopy(category)
            print(f'Process category: {category["name"]}')
            texts, labels = [], []
            for value in category['values']:
                if isinstance(value['text'], list):
                    for text in value['text']:
                        texts.append(text)
                        labels.append(value['label'])
                else:
                    texts.append(value['text'])
                    labels.append(value['label'])
            category['_texts'] = texts
            category['_labels'] = labels
            category['_tensors'] = clip.tokenize(texts).to(self.device)
            self.categories.append(category)
    
    def init_vectors(self, config):
        vectors_path = config['vectors']['path']
        vectors_folder = list(Path(vectors_path).glob('*.txt'))
        
        
        self.labels_arr = []
        self.vectors_arr = []
        self.labels_map = {}
        
        for idx, file_path in enumerate(vectors_folder):
            self.labels_map[idx] = file_path.stem
            vectors = np.loadtxt(str(file_path))
            print(vectors.shape)
            if len(vectors.shape) == 1:
                vectors = [vectors]
                
            for vector in vectors:
                self.labels_arr.append(idx)
                self.vectors_arr.append(vector)
                
        self.labels_arr = np.array(self.labels_arr)

    def cluster(self, image):
        img = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(img)
        vector = image_features.cpu().detach().numpy()[0]
        
        distances = -euclidean_distances(vector, self.vectors_arr)
        prob = softmax(distances)

        result = {}
        for label_candidate in set(self.labels_arr):
            label_prob = (prob * (self.labels_arr == label_candidate)).sum()
            result[self.labels_map[label_candidate]] =  label_prob
        
        return result
            
    def predict(self, image):
        pimage = self.preprocess(image).unsqueeze(0).to(self.device)
        results = []
        condition_tracking = {}
        
        with torch.no_grad():
            for category in self.categories:
                #check conditions
                skip = False
                if 'conditions' in category:
                    for cond in category['conditions']:
                        if cond['type'] == 'must':
                            expected_value = condition_tracking.get(cond['category'], None)
                            if expected_value != cond['value']:
                                skip = True
                                break
                        else:
                            print('Unsupported condition type')
                if skip:
                    results.append({
                        'name': category['name'],
                        'condition': 0
                    })
                    continue
                
                logits_per_image, logits_per_text = self.model(pimage, category['_tensors']) 
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                result = {'raw': []}
                
                idxmax = probs.argmax()
                result['label'] = category['_labels'][idxmax]
                result['text'] = category['_texts'][idxmax]
                result['prob'] = float(probs[0][idxmax])
                result['name'] = category['name']
                
                for text, label, prob in zip(category['_texts'], category['_labels'], probs[0]):
                    result['raw'].append({
                        'text': text,
                        'label': label,
                        'prob': float(prob)
                    })
                results.append(result)
                
                condition_tracking[category['name']] = category['_labels'][idxmax]
                
        pimage.detach()
        return results

class ClipServer:
    
    def __init__(self, service, config):
        self.service = service
        self.config = config

    """ Sample request handler class. """
    @cherrypy.expose
    def index(self):
        return 'Hello'
    
    def _opByUrl(self, url, op):
        is_ok = True
        with tempfile.NamedTemporaryFile() as f:
            res = requests.get(url, stream=True)
            res.raise_for_status()
            chunks = 0
            for chunk in res.iter_content(chunk_size=8192):
                f.write(chunk)
                chunks += 1
                if self.config['max_file_size'] / 8192 < chunks:
                    is_ok = False
                    break
            f.flush()
            if is_ok:
                img = Image.open(f.name)
                return op(img)
        return {'Error': True}
    
    def _opByUpload(self, ufile, op):
        with tempfile.NamedTemporaryFile() as f:
            f.write(ufile.file.read())
            f.flush()
            img = Image.open(f.name)
            return op(img)
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predictByUrl(self, url):
        return self._opByUrl(url, self.service.predict)
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predictByUpload(self, ufile):
        return self._opByUpload(ufile, self.service.predict)
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def clusterByUrl(self, url):
        return self._opByUrl(url, self.service.cluster)
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def clusterByUpload(self, ufile):
        return self._opByUpload(ufile, self.service.cluster)

if __name__ == '__main__':
    service = ClipService(CONFIG)
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'} )
    cherrypy.quickstart(ClipServer(service, CONFIG))
    
