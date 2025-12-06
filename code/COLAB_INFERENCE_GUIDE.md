# Guide complet : Tester l'inférence LLaMA + LoRA dans Google Colab

Ce guide vous explique étape par étape comment tester votre modèle LLaMA fine-tuné avec LoRA dans Google Colab.

---

## 📋 Prérequis

- Un compte Google avec accès à [Google Colab](https://colab.research.google.com/)
- Un token Hugging Face avec accès au modèle LLaMA (si le modèle est gated)
- Votre modèle LoRA fine-tuné (fichiers LoRA sauvegardés)

---

## 🚀 ÉTAPE 1 : Préparer Colab

### 1.1 Créer un nouveau notebook Colab

1. Allez sur [Google Colab](https://colab.research.google.com/)
2. Cliquez sur **"New notebook"**
3. Renommez le notebook (par exemple : "Test Inference LLaMA LoRA")

### 1.2 Activer le GPU

1. Dans le menu, cliquez sur **Runtime** → **Change runtime type**
2. Sélectionnez **T4 GPU** (gratuit) ou **A100** (payant, plus rapide)
3. Cliquez sur **Save**

### 1.3 Vérifier que le GPU est actif

Exécutez cette cellule pour vérifier :

```python
import torch
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

---

## 📦 ÉTAPE 2 : Installer les dépendances

Exécutez cette cellule pour installer tous les packages nécessaires :

```python
!pip install -q torch transformers accelerate bitsandbytes peft pdfplumber pyffx scikit-learn natsort fpdf2 lxml nltk
```

⚠️ **Note** : L'installation prend quelques minutes. Attendez que ce soit terminé avant de continuer.

---

## 🔐 ÉTAPE 3 : Authentification Hugging Face

### 3.1 Obtenir votre token Hugging Face

1. Allez sur [huggingface.co](https://huggingface.co/)
2. Connectez-vous ou créez un compte
3. Allez dans **Settings** → **Access Tokens**
4. Créez un nouveau token (ou utilisez un token existant)
5. Copiez le token (commence par `hf_...`)

### 3.2 Se connecter dans Colab

```python
from huggingface_hub import login

# Remplacez VOTRE_TOKEN par votre vrai token Hugging Face
login("VOTRE_TOKEN_HF")
```

Exécutez cette cellule. Un lien apparaîtra - cliquez dessus pour autoriser l'accès, puis revenez à Colab.

---

## 📁 ÉTAPE 4 : Télécharger votre code et vos fichiers

### 4.1 Option A : Depuis GitHub (si votre code est sur GitHub)

```python
!git clone https://github.com/VOTRE_REPO/amalytics-ml.git
```

### 4.2 Option B : Upload manuel dans Colab

1. Créez un dossier pour votre code :
```python
!mkdir -p /content/amalytics-ml/code
```

2. Cliquez sur l'icône 📁 (Files) dans la barre latérale de Colab
3. Uploadez votre code dans `/content/amalytics-ml/code/`

### 4.3 Ajouter le code au Python path

```python
import sys
from pathlib import Path

# Ajouter le dossier src au path Python
code_dir = Path("/content/amalytics-ml/code")
src_dir = code_dir / "src"
sys.path.insert(0, str(src_dir))

print(f"✅ Code ajouté au path: {src_dir}")
```

---

## 📥 ÉTAPE 5 : Télécharger votre modèle LoRA

Vous devez avoir vos fichiers LoRA quelque part (Google Drive, Hugging Face Hub, etc.).

### 5.1 Option A : Depuis Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Copier depuis Google Drive vers Colab
!cp -r /content/drive/MyDrive/path/to/your/lora-output /content/lora-output
```

### 5.2 Option B : Depuis Hugging Face Hub

Si votre LoRA est sur Hugging Face :

```python
from huggingface_hub import snapshot_download

# Remplacez par votre repo Hugging Face
snapshot_download(
    repo_id="VOTRE_USERNAME/VOTRE_LORA_REPO",
    local_dir="/content/lora-output",
    token="VOTRE_TOKEN_HF"  # Optionnel si déjà connecté
)
```

### 5.3 Option C : Upload direct dans Colab

```python
!mkdir -p /content/lora-output
```

Puis uploadez vos fichiers LoRA via l'interface Files de Colab.

### 5.4 Vérifier que les fichiers LoRA sont présents

```python
import os
lora_path = "/content/lora-output"
if os.path.exists(lora_path):
    files = os.listdir(lora_path)
    print(f"✅ Fichiers LoRA trouvés ({len(files)} fichiers):")
    for f in files[:10]:  # Afficher les 10 premiers
        print(f"  - {f}")
    if len(files) > 10:
        print(f"  ... et {len(files) - 10} autres fichiers")
else:
    print("❌ Dossier LoRA non trouvé!")
```

---

## 📄 ÉTAPE 6 : Préparer les fichiers de test

### 6.1 Télécharger ou uploader un PDF de test

```python
!mkdir -p /content/test_data
```

Puis uploadez un PDF de test dans `/content/test_data/` via l'interface Files.

### 6.2 Télécharger ou créer un template JSON vide

```python
!mkdir -p /content/test_data/templates
```

Uploadez votre template JSON vide dans `/content/test_data/templates/`.

### 6.3 Vérifier les fichiers

```python
import os
test_dir = "/content/test_data"
if os.path.exists(test_dir):
    print("📁 Fichiers de test:")
    for root, dirs, files in os.walk(test_dir):
        level = root.replace(test_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limiter l'affichage
            print(f'{subindent}{file}')
```

---

## ⚙️ ÉTAPE 7 : Créer la configuration d'inférence

Créez un fichier de configuration JSON. Vous pouvez le créer directement dans Colab :

```python
import json

config = {
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lora_path": "/content/lora-output",
    "template_path": "/content/test_data/templates/sample_1_template_empty.json",
    "max_new_tokens": 3000,
    "do_sample": False,
    "return_scores": True,
    "load_in_4bit": True,
    "device_map": "auto",
    "use_batch_inference": False,
    "max_measurements_per_batch": 2,
    "dedup_consecutive_keys": True,
    "apply_anonymization": False,
    "extra_generation_kwargs": {}
}

# Sauvegarder la configuration
with open("/content/infer_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Configuration créée: /content/infer_config.json")
print(json.dumps(config, indent=2))
```

---

## 🔍 ÉTAPE 8 : Tester les imports

Vérifiez que tous les modules peuvent être importés :

```python
try:
    from amalytics_ml.config import InferenceConfig
    from amalytics_ml.models.inference import run_inference
    print("✅ Imports réussis!")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    import traceback
    traceback.print_exc()
```

---

## 🎯 ÉTAPE 9 : Exécuter l'inférence

### 9.1 Charger la configuration

```python
import json
from pathlib import Path
from amalytics_ml.config import InferenceConfig

# Charger la config
with open("/content/infer_config.json", "r") as f:
    cfg_dict = json.load(f)

# Créer l'objet InferenceConfig
cfg = InferenceConfig(**cfg_dict)

# Charger le template si nécessaire
if cfg.template is None and cfg.template_path:
    template_path = Path(cfg.template_path)
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as f:
            cfg.template = json.load(f)
        print(f"✅ Template chargé: {cfg.template_path}")
    else:
        print(f"❌ Template non trouvé: {cfg.template_path}")
```

### 9.2 Vérifier les chemins

```python
print("📁 Vérification des chemins:")
print(f"  Model: {cfg.model_path}")
print(f"  LoRA: {cfg.lora_path}")
print(f"  Template: {cfg.template_path if cfg.template_path else 'Inclus dans config'}")

# Vérifier l'existence des fichiers locaux
from pathlib import Path

if cfg.lora_path:
    lora_path = Path(cfg.lora_path)
    if lora_path.exists():
        print(f"  ✅ LoRA trouvé: {len(list(lora_path.glob('*')))} fichiers")
    else:
        print(f"  ⚠️  LoRA non trouvé (sera téléchargé si HuggingFace ID)")

if cfg.template:
    print(f"  ✅ Template chargé en mémoire")
```

### 9.3 Exécuter l'inférence

⚠️ **ATTENTION** : Cette étape va charger le modèle complet en mémoire GPU. Cela peut prendre 5-15 minutes.

```python
from amalytics_ml.models.inference import run_inference
from pathlib import Path
import json

# Chemin vers votre PDF de test
pdf_path = "/content/test_data/sample_1.pdf"  # Remplacez par votre PDF

print("🚀 Démarrage de l'inférence...")
print("⏳ Chargement du modèle (cela peut prendre plusieurs minutes)...")

try:
    # Exécuter l'inférence
    result = run_inference(
        model_path=cfg.model_path,
        lora_path=cfg.lora_path if cfg.lora_path else "",
        input_text=Path(pdf_path),  # Le code détectera automatiquement que c'est un PDF
        config=cfg,
    )
    
    print("✅ Inférence terminée!")
    
    # Afficher le résultat
    print("\n" + "="*60)
    print("RÉSULTAT JSON")
    print("="*60)
    print(json.dumps(result.parsed_json, indent=2, ensure_ascii=False))
    
    # Afficher les scores de confiance si disponibles
    if result.confidence_scores:
        print("\n" + "="*60)
        print("SCORES DE CONFIANCE")
        print("="*60)
        print(json.dumps(result.confidence_scores, indent=2, ensure_ascii=False))
    
    # Sauvegarder les résultats
    output_dir = Path("/content/outputs")
    output_dir.mkdir(exist_ok=True)
    
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result.parsed_json, f, indent=2, ensure_ascii=False)
    
    if result.confidence_scores:
        with (output_dir / "confidence.json").open("w", encoding="utf-8") as f:
            json.dump(result.confidence_scores, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Résultats sauvegardés dans {output_dir}/")
    
except Exception as e:
    print(f"❌ Erreur pendant l'inférence: {e}")
    import traceback
    traceback.print_exc()
```

---

## 📊 ÉTAPE 10 : Analyser les résultats

### 10.1 Comparer avec la ground truth (si disponible)

```python
# Si vous avez une ground truth
gt_path = "/content/test_data/ground_truth/sample_1.json"
if Path(gt_path).exists():
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)
    
    print("Comparaison avec la ground truth:")
    # Ici vous pouvez ajouter votre logique de comparaison
```

### 10.2 Visualiser les scores de confiance

```python
if result.confidence_scores:
    import matplotlib.pyplot as plt
    
    scores = list(result.confidence_scores.values())
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black')
    plt.xlabel('Score de confiance')
    plt.ylabel('Nombre de champs')
    plt.title('Distribution des scores de confiance')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Score moyen: {sum(scores) / len(scores):.4f}")
    print(f"Score min: {min(scores):.4f}")
    print(f"Score max: {max(scores):.4f}")
```

---

## 🔧 ÉTAPE 11 : Dépannage

### Problème : Out of Memory (OOM)

**Solution 1** : Réduire `max_new_tokens`
```python
config["max_new_tokens"] = 1500  # Au lieu de 3000
```

**Solution 2** : S'assurer que 4-bit est activé
```python
config["load_in_4bit"] = True
```

**Solution 3** : Utiliser un GPU plus puissant (A100 au lieu de T4)

### Problème : Erreur lors du chargement du modèle

```python
# Vérifier la version de transformers
!pip show transformers

# Mettre à jour si nécessaire
!pip install -U transformers
```

### Problème : LoRA non trouvé

```python
# Vérifier le contenu du dossier LoRA
import os
lora_path = "/content/lora-output"
print("Contenu du dossier LoRA:")
for f in os.listdir(lora_path):
    print(f"  - {f}")
```

### Problème : Template non chargé

```python
# Vérifier que le template est bien chargé
if cfg.template:
    print(f"✅ Template chargé: {type(cfg.template)}")
    print(f"Clés principales: {list(cfg.template.keys())[:5]}")
else:
    print("❌ Template non chargé")
```

---

## ✅ Checklist finale

Avant de lancer l'inférence, vérifiez :

- [ ] GPU activé dans Colab (Runtime → Change runtime type)
- [ ] Toutes les dépendances installées
- [ ] Authentification Hugging Face réussie
- [ ] Code téléchargé/uploadé dans Colab
- [ ] Fichiers LoRA téléchargés/uploadés
- [ ] PDF de test disponible
- [ ] Template JSON disponible
- [ ] Configuration JSON créée

---

## 🎉 Exemple complet (tout-en-un)

Voici un script complet que vous pouvez exécuter dans une seule cellule (après avoir téléchargé vos fichiers) :

```python
# ============================================
# INFÉRENCE COMPLÈTE LLaMA + LoRA dans Colab
# ============================================

import json
import sys
from pathlib import Path

# 1. Setup
print("📦 Installation des dépendances...")
!pip install -q torch transformers accelerate bitsandbytes peft pdfplumber pyffx scikit-learn natsort fpdf2 lxml nltk

# 2. Authentification
print("🔐 Authentification Hugging Face...")
from huggingface_hub import login
login("VOTRE_TOKEN_HF")  # ⚠️ REMPLACER PAR VOTRE TOKEN

# 3. Ajouter le code au path
code_dir = Path("/content/amalytics-ml/code")
src_dir = code_dir / "src"
sys.path.insert(0, str(src_dir))
print(f"✅ Code ajouté au path: {src_dir}")

# 4. Configuration
print("⚙️ Configuration...")
config = {
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lora_path": "/content/lora-output",
    "template_path": "/content/test_data/templates/sample_1_template_empty.json",
    "max_new_tokens": 3000,
    "do_sample": False,
    "return_scores": True,
    "load_in_4bit": True,
    "device_map": "auto",
}

cfg = InferenceConfig(**config)

# Charger le template
with open(cfg.template_path, "r") as f:
    cfg.template = json.load(f)
print("✅ Configuration prête")

# 5. Inférence
print("🚀 Lancement de l'inférence...")
from amalytics_ml.models.inference import run_inference

result = run_inference(
    model_path=cfg.model_path,
    lora_path=cfg.lora_path,
    input_text="/content/test_data/sample_1.pdf",
    config=cfg,
)

# 6. Résultats
print("\n" + "="*60)
print("RÉSULTAT")
print("="*60)
print(json.dumps(result.parsed_json, indent=2, ensure_ascii=False))

print("\n✅ Inférence terminée avec succès!")
```

---

## 📝 Notes importantes

1. **Temps d'exécution** : Le chargement du modèle peut prendre 5-15 minutes la première fois
2. **Mémoire GPU** : Utilisez `load_in_4bit: true` pour économiser la mémoire
3. **Token Hugging Face** : Nécessaire si le modèle est "gated" (accès restreint)
4. **Sauvegarde** : Les fichiers dans Colab sont temporaires - téléchargez les résultats avant de fermer !

---

## 🆘 Besoin d'aide ?

Si vous rencontrez des problèmes :

1. Vérifiez les messages d'erreur dans Colab
2. Consultez la section "Dépannage" ci-dessus
3. Vérifiez que tous les chemins de fichiers sont corrects
4. Assurez-vous que le GPU est bien activé

---

**Bon test ! 🚀**

