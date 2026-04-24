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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 30
    LR = 1e-4
    CHECKPOINT_PATH = "INTERRUPTED_model.pth"
    DATA_DIR = '/content/crohme2019/crohme2019/train' # Chemin Colab

    # CHARGEMENT VOCAB
    with open('vocab.json', 'r') as f:
        vocab_list = json.load(f)
    tokenizer = LaTeXTokenizer(vocab_list=vocab_list)
    vocab_size = tokenizer.get_vocab_size()
    
    # INITIALISATION
    model = HMERModel(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['[PAD]'])
    
    start_epoch = 0

    # REPRISE SI UN MODÈLE EXISTE
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Chargement du checkpoint : {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Reprise de l'entraînement à l'époque {start_epoch + 1}")

    # DATASET
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = InkMLDataset(DATA_DIR, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            collate_fn=lambda b: collate_fn(b, tokenizer.vocab['[PAD]']))

    def save_checkpoint(current_epoch):
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        # Sauvegarde aussi un backup spécifique par époque
        torch.save(model.state_dict(), f"model_weights_epoch_{current_epoch+1}.pth")

    try:
        for epoch in range(start_epoch, EPOCHS):
            model.train()
            epoch_loss = 0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            
            for imgs, seqs in pbar:
                imgs, seqs = imgs.to(device), seqs.to(device)
                tgt_in, tgt_out = seqs[:, :-1], seqs[:, 1:]
                
                optimizer.zero_grad()
                memory = model.forward_encoder(imgs)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
                
                tgt_embed = model.embedding(tgt_in) + model.pos_encoder[:, :tgt_in.size(1), :]
                output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(output)
                
                loss = criterion(logits.reshape(-1, vocab_size), tgt_out.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            save_checkpoint(epoch)
            print(f"Époque {epoch+1} terminée. Sauvegarde effectuée.")

    except KeyboardInterrupt:
        print("\nInterruption détectée. Sauvegarde forcée...")
        save_checkpoint(epoch)

if __name__ == "__main__":
    train_model()