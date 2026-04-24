import os, json, re
import xml.etree.ElementTree as ET

def build():
    # Chemin vers tes données sur Colab
    path = '/content/crohme2019/crohme2019/train'
    tokens = set()
    # Pattern pour capturer les commandes LaTeX et les caractères individuels
    pattern = re.compile(r"\\[a-zA-Z]+|[a-zA-Z0-9]|\S")
    
    if not os.path.exists(path):
        print(f"Erreur : Le dossier {path} est introuvable.")
        return

    files = [f for f in os.listdir(path) if f.endswith('.inkml')]
    print(f"Extraction du vocabulaire sur {len(files)} fichiers...")
    
    files_processed = 0
    errors = 0

    for f in files:
        try:
            tree = ET.parse(os.path.join(path, f))
            # Utilisation de findall avec ou sans namespace selon le fichier
            for anno in tree.getroot().findall('.//{http://www.w3.org/2003/InkML}annotation'):
                if anno.attrib.get('type') == 'truth':
                    content = anno.text.strip() if anno.text else ""
                    tokens.update(pattern.findall(content))
            files_processed += 1
        except ET.ParseError:
            # Si le fichier est mal formé, on le passe
            errors += 1
            continue
    
    # Nettoyage : on enlève d'éventuels tokens vides ou None
    vocab_list = sorted(list(filter(None, tokens)))
    
    with open('vocab.json', 'w') as f:
        json.dump(vocab_list, f)
    
    print(f"Terminé !")
    print(f"- Fichiers analysés avec succès : {files_processed}")
    print(f"- Fichiers ignorés (erreurs XML) : {errors}")
    print(f"- Symboles uniques trouvés : {len(vocab_list)}")

if __name__ == "__main__":
    build()