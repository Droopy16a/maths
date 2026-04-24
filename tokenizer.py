import re

class LaTeXTokenizer:
    def __init__(self, vocab_list=None):
        self.pad_token = '[PAD]'
        self.bos_token = '[BOS]'
        self.eos_token = '[EOS]'
        self.unk_token = '[UNK]'
        
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        
        # On commence toujours par les tokens spéciaux
        self.vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Si une liste est fournie, on ajoute les symboles à la suite
        if vocab_list:
            for token in vocab_list:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)
            
        self.inverse_vocab = {i: token for token, i in self.vocab.items()}
        self.pattern = re.compile(r"\\[a-zA-Z]+|\. |[a-zA-Z0-9]|\S")

    def decode(self, token_ids):
        """Transforme les IDs en chaîne LaTeX, gère les int et les Tenseurs."""
        tokens = []
        for t in token_ids:
            # Correction : On vérifie si 't' a la méthode .item() (cas du Tenseur)
            # Sinon, on utilise 't' directement (cas de l'entier)
            val = t.item() if hasattr(t, 'item') else t
            
            token = self.inverse_vocab.get(val, self.unk_token)
            
            if token == self.eos_token:
                break
            if token not in self.special_tokens:
                tokens.append(token)
        return " ".join(tokens)

    def get_vocab_size(self):
        return len(self.vocab)