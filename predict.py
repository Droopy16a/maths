import torch
from torchvision import transforms
from PIL import Image
from model import HMERModel
from tokenizer import LaTeXTokenizer

def predict_latex(image_path, model_path="model_weights.pth"):
    # 1. Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ATTENTION: Il faut charger la vraie liste de vocabulaire générée pendant l'entraînement
    # Ici on simule une liste factice pour que le code tourne
    vocab_simule = ['[PAD]', '[BOS]', '[EOS]', '[UNK]', 'x', '^', '2', '+', 'y', '=', 'z', '\\alpha']
    tokenizer = LaTeXTokenizer(vocab_list=vocab_simule)
    
    model = HMERModel(vocab_size=tokenizer.get_vocab_size()).to(device)
    
    # 2. Chargement des poids (Décommente ceci une fois que tu as entraîné le modèle !)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Préparation de l'image (identique à l'entraînement de l'article)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # DenseNet attend du RGB
        transforms.Resize((224, 224)), # Taille standard ImageNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # Ajout dimension Batch

    # 4. Inférence (Génération autorégressive)
    max_length = 150 # Longueur maximale de l'équation
    
    with torch.no_grad():
        # L'image passe dans DenseNet
        memory = model.forward_encoder(image_tensor)
        
        # On commence la phrase avec le token [BOS]
        target_indices = [tokenizer.vocab['[BOS]']]
        
        for _ in range(max_length):
            tgt_tensor = torch.LongTensor([target_indices]).to(device)
            tgt_embed = model.embedding(tgt_tensor) + model.pos_encoder[:, :tgt_tensor.size(1), :]
            
            # Mask pour empêcher le modèle de tricher (regarder dans le futur)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)
            
            # Prédiction du prochain symbole
            output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(output)
            next_token_id = logits[0, -1, :].argmax().item()
            
            target_indices.append(next_token_id)
            
            if next_token_id == tokenizer.vocab['[EOS]']:
                break
                
    # 5. Décodage final
    latex_result = tokenizer.decode(target_indices[1:]) # On ignore le [BOS]
    return latex_result

if __name__ == "__main__":
    # Test avec une image
    chemin_image = "equation_test.png" 
    try:
        result = predict_latex(chemin_image)
        print("\n--- Équation Détectée ---")
        print(f"$$ {result} $$")
    except FileNotFoundError:
        print(f"Crée une petite image nommée '{chemin_image}' pour tester le script !")