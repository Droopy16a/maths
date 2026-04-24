import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import numpy as np
import os

class InkMLDataset(Dataset):
    def __init__(self, folder_path, tokenizer, transform=None):
        self.folder_path = folder_path
        self.tokenizer = tokenizer
        self.transform = transform
        # On liste les fichiers une seule fois au démarrage
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.inkml')]

    def __len__(self):
        return len(self.files)

    def parse_inkml(self, file_path):
        """Extrait les traces et la formule LaTeX du XML."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        
        # Truth (LaTeX)
        truth = ""
        for anno in root.findall('inkml:annotation', ns):
            if anno.attrib.get('type') == 'truth':
                truth = anno.text.strip()
                break
        
        # Traces (Coordonnées)
        traces = []
        for trace_tag in root.findall('inkml:trace', ns):
            points = [p.strip().split()[:2] for p in trace_tag.text.strip().split(',')]
            traces.append([[float(p[0]), float(p[1])] for p in points])
        
        return traces, truth

    def render_to_tensor(self, traces):
        """Dessine l'image en RAM et retourne une image PIL."""
        all_pts = np.vstack(traces)
        min_p = np.min(all_pts, axis=0)
        max_p = np.max(all_pts, axis=0)
        
        # Normalisation et dessin
        w, h = max_p - min_p + 1
        img = Image.new('RGB', (int(w)+40, int(h)+40), 'white')
        draw = ImageDraw.Draw(img)
        
        for trace in traces:
            pts = [(p[0] - min_p[0] + 20, p[1] - min_p[1] + 20) for p in trace]
            draw.line(pts, fill='black', width=3)
            
        return img

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        try:
            traces, formula = self.parse_inkml(file_path)
            
            # On crée l'image en mémoire
            image = self.render_to_tensor(traces)
            
            if self.transform:
                image = self.transform(image)
                
            # Tokenization de la formule
            tokens = self.tokenizer.pattern.findall(formula)
            token_ids = [self.tokenizer.vocab['[BOS]']] + \
                        [self.tokenizer.vocab.get(t, self.tokenizer.vocab['[UNK]']) for t in tokens] + \
                        [self.tokenizer.vocab['[EOS]']]
            
            return image, torch.tensor(token_ids, dtype=torch.long)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.files))


def collate_fn(batch, pad_idx):
    images, formulas = zip(*batch)
    images = torch.stack(images)
    formulas_padded = torch.nn.utils.rnn.pad_sequence(
        formulas, batch_first=True, padding_value=pad_idx
    )
    return images, formulas_padded