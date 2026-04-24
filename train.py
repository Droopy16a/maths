import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json, os
from tqdm import tqdm

from model import HMERModel
from tokenizer import LaTeXTokenizer
from dataset import InkMLDataset, collate_fn

def train_model():
    # CONFIGURATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    SAVE_PATH = "maths_model.pth"
    DATA_DIR = r'crohme2019/crohme2019/train' # À ajuster

    # CHARGEMENT VOCAB ET MODEL
    with open('vocab.json', 'r') as f:
        vocab_list = json.load(f)
    tokenizer = LaTeXTokenizer(vocab_list=vocab_list)
    
    model = HMERModel(vocab_size=tokenizer.get_vocab_size()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])

    # DATASET
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = InkMLDataset(DATA_DIR, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=lambda b: collate_fn(b, tokenizer.vocab['[PAD]']))

    print(f"Entraînement lancé sur {device}. Ctrl+C pour stopper.")

    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for imgs, seqs in pbar:
                imgs, seqs = imgs.to(device), seqs.to(device)
                tgt_in, tgt_out = seqs[:, :-1], seqs[:, 1:]
                
                optimizer.zero_grad()
                memory = model.forward_encoder(imgs)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
                
                # Embedding + Position
                tgt_embed = model.embedding(tgt_in) + model.pos_encoder[:, :tgt_in.size(1), :]
                output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(output)
                
                loss = criterion(logits.reshape(-1, tokenizer.get_vocab_size()), tgt_out.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            torch.save(model.state_dict(), f"epoch_{epoch+1}.pth")

    except KeyboardInterrupt:
        print("\nArrêt manuel. Sauvegarde de sécurité...")
        torch.save(model.state_dict(), "INTERRUPTED_model.pth")

if __name__ == "__main__":
    train_model()