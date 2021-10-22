import cherrypy
import clip
import yaml
import torch
import copy
import tempfile
import requests

from PIL import Image

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()
print(CONFIG)
class ClipService:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(config['model'], self.device)
        self.init_categories(config)
    
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
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predictByUrl(self, url):
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
                return self.service.predict(img)
        return {'Error': True}
    
    @cherrypy.expose
    @cherrypy.tools.json_out()
    def predictByUpload(self, ufile):
        with tempfile.NamedTemporaryFile() as f:
            f.write(ufile.file.read())
            f.flush()
            img = Image.open(f.name)
            return self.service.predict(img)

if __name__ == '__main__':
    service = ClipService(CONFIG)
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'} )
    cherrypy.quickstart(ClipServer(service, CONFIG))
    
