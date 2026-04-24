import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
from model import HMERModel
from tokenizer import LaTeXTokenizer

def load_prediction_model(model_path, vocab_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(vocab_path, 'r') as f:
        vocab_list = json.load(f)
    tokenizer = LaTeXTokenizer(vocab_list=vocab_list)
    
    model = HMERModel(vocab_size=tokenizer.get_vocab_size()).to(device)
    
    # Charger les poids (on gère le dictionnaire complet ou juste les poids)
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model.eval()
    return model, tokenizer, device

def predict(image_path, model, tokenizer, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    max_len = 100
    with torch.no_grad():
        memory = model.forward_encoder(img_tensor)
        # On commence par le token de début [BOS]
        tokens = [tokenizer.vocab['[BOS]']]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor([tokens]).to(device)
            # Embedding + Position
            tgt_embed = model.embedding(tgt_tensor) + model.pos_encoder[:, :tgt_tensor.size(1), :]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            
            output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(output)
            
            # On prend le dernier symbole prédit
            next_token = logits[0, -1, :].argmax().item()
            tokens.append(next_token)
            
            if next_token == tokenizer.vocab['[EOS]']:
                break
                
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    # Paramètres
    MODEL_WEIGHTS = "INTERRUPTED_model.pth" # Ou le dernier model_weights_epoch_X.pth
    VOCAB_FILE = "vocab.json"
    TEST_IMAGE = "testX.png" # Place une image ici pour tester
    
    print("Chargement du modèle...")
    model, tokenizer, device = load_prediction_model(MODEL_WEIGHTS, VOCAB_FILE)
    
    print("Analyse de l'image...")
    result = predict(TEST_IMAGE, model, tokenizer, device)
    print(f"\nRésultat LaTeX : {result}")