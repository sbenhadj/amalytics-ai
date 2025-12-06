# 🚀 Guide rapide : Inférence LLaMA + LoRA dans Colab

## 📝 Checklist rapide

### Préparation (5 minutes)
- [ ] Créer un notebook Colab
- [ ] Activer le GPU (Runtime → Change runtime type → T4 GPU)
- [ ] Obtenir votre token Hugging Face

### Installation (5-10 minutes)
- [ ] Installer les dépendances
- [ ] S'authentifier avec Hugging Face
- [ ] Télécharger/Uploadez votre code
- [ ] Télécharger/Uploadez votre modèle LoRA
- [ ] Préparer un PDF de test + template JSON

### Exécution (15-30 minutes)
- [ ] Créer la configuration
- [ ] Lancer l'inférence
- [ ] Voir les résultats

---

## 🎯 Script complet (copier-coller)

```python
# ============================================
# ÉTAPE 1 : Setup GPU et dépendances
# ============================================

import torch
print(f"GPU: {torch.cuda.is_available()}")

!pip install -q torch transformers accelerate bitsandbytes peft pdfplumber pyffx scikit-learn natsort fpdf2 lxml nltk

# ============================================
# ÉTAPE 2 : Authentification Hugging Face
# ============================================

from huggingface_hub import login
login("VOTRE_TOKEN_HF")  # ⚠️ REMPLACER

# ============================================
# ÉTAPE 3 : Ajouter le code au path
# ============================================

import sys
from pathlib import Path

# Option A : Upload manuel (créer le dossier puis uploader via Files)
!mkdir -p /content/amalytics-ml/code

# Option B : Depuis GitHub
# !git clone https://github.com/VOTRE_REPO/amalytics-ml.git

sys.path.insert(0, "/content/amalytics-ml/code/src")

# ============================================
# ÉTAPE 4 : Télécharger le modèle LoRA
# ============================================

# Option A : Depuis Google Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/path/to/lora-output /content/lora-output

# Option B : Depuis Hugging Face
# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="USER/REPO", local_dir="/content/lora-output", token="VOTRE_TOKEN")

# Option C : Upload direct (créer le dossier puis uploader)
# !mkdir -p /content/lora-output

# ============================================
# ÉTAPE 5 : Préparer les fichiers de test
# ============================================

!mkdir -p /content/test_data/templates
# Uploadez via l'interface Files:
# - PDF dans /content/test_data/
# - Template JSON dans /content/test_data/templates/

# ============================================
# ÉTAPE 6 : Configuration et inférence
# ============================================

import json
from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import run_inference
from pathlib import Path

# Configuration
config = {
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lora_path": "/content/lora-output",
    "template_path": "/content/test_data/templates/sample_1_template_empty.json",  # ⚠️ MODIFIER
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

# ============================================
# ÉTAPE 7 : Lancer l'inférence
# ============================================

pdf_path = "/content/test_data/sample_1.pdf"  # ⚠️ MODIFIER

print("🚀 Lancement de l'inférence (5-15 minutes)...")

result = run_inference(
    model_path=cfg.model_path,
    lora_path=cfg.lora_path,
    input_text=Path(pdf_path),
    config=cfg,
)

# ============================================
# ÉTAPE 8 : Afficher les résultats
# ============================================

print("\n" + "="*60)
print("RÉSULTAT JSON")
print("="*60)
print(json.dumps(result.parsed_json, indent=2, ensure_ascii=False))

if result.confidence_scores:
    print("\n" + "="*60)
    print("SCORES DE CONFIANCE")
    print("="*60)
    print(json.dumps(result.confidence_scores, indent=2, ensure_ascii=False))

# Sauvegarder
output_dir = Path("/content/outputs")
output_dir.mkdir(exist_ok=True)

with (output_dir / "result.json").open("w", encoding="utf-8") as f:
    json.dump(result.parsed_json, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultat sauvegardé dans {output_dir}/")
print("✅ Terminé!")
```

---

## 🔧 Chemins à modifier

Dans le script ci-dessus, remplacez :

1. `"VOTRE_TOKEN_HF"` → Votre token Hugging Face
2. `"/content/drive/MyDrive/path/to/lora-output"` → Chemin vers votre LoRA sur Drive
3. `"/content/test_data/templates/sample_1_template_empty.json"` → Chemin vers votre template
4. `"/content/test_data/sample_1.pdf"` → Chemin vers votre PDF de test

---

## 📚 Documentation complète

Pour plus de détails, consultez : **`COLAB_INFERENCE_GUIDE.md`**

---

## ⚠️ Problèmes courants

### Out of Memory
```python
config["load_in_4bit"] = True  # Déjà activé
config["max_new_tokens"] = 1500  # Réduire si nécessaire
```

### LoRA non trouvé
```python
import os
print(os.listdir("/content/lora-output"))  # Vérifier les fichiers
```

### Template non chargé
```python
from pathlib import Path
template_path = Path("/content/test_data/templates/sample_1_template_empty.json")
print(f"Existe: {template_path.exists()}")
```

---

**Bon test ! 🎉**

