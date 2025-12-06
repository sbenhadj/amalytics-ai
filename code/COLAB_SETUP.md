# Guide de configuration pour Google Colab

Ce guide explique comment utiliser le code d'inférence avec LLaMA + LoRA sur Google Colab.

## ✅ Tests de validation passés

Tous les tests de structure ont été validés :
- ✅ Imports des modules
- ✅ Configuration
- ✅ Split de template
- ✅ Anonymisation
- ✅ Détection PDF
- ✅ Chargement de template

## 📋 Configuration recommandée pour Colab

### 1. Configuration JSON pour l'inférence

Créez un fichier `infer_colab.json` :

```json
{
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lora_path": "/content/lora-output",
    "template_path": "/content/data/templates/empty/sample_1_template_empty.json",
    "max_new_tokens": 3000,
    "do_sample": false,
    "return_scores": true,
    "load_in_4bit": true,
    "device_map": "auto",
    "use_batch_inference": true,
    "max_measurements_per_batch": 2,
    "dedup_consecutive_keys": true,
    "apply_anonymization": false,
    "anonymization_use_ner": false,
    "extra_generation_kwargs": {}
}
```

### 2. Configuration avec anonymisation

Si vous voulez activer l'anonymisation :

```json
{
    "model_path": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "lora_path": "/content/lora-output",
    "template_path": "/content/data/templates/empty/sample_1_template_empty.json",
    "max_new_tokens": 3000,
    "do_sample": false,
    "return_scores": true,
    "load_in_4bit": true,
    "device_map": "auto",
    "use_batch_inference": true,
    "max_measurements_per_batch": 2,
    "apply_anonymization": true,
    "anonymization_secret_key": "sbh86",
    "anonymization_use_ner": true,
    "extra_generation_kwargs": {}
}
```

## 🚀 Code pour Colab

### Setup initial

```python
# Installer les dépendances
!pip install -q torch transformers accelerate bitsandbytes peft pdfplumber pyffx scikit-learn natsort fpdf2 lxml nltk

# Authentification Hugging Face
from huggingface_hub import login
login("VOTRE_TOKEN_HF")  # Remplacez par votre token

# Importer les modules
import sys
from pathlib import Path

# Ajouter le code au path
sys.path.insert(0, '/content/amalytics-ml/code/src')

from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import run_inference
```

### Exemple d'inférence simple

```python
# Charger la configuration
with open('/content/infer_colab.json', 'r') as f:
    cfg_dict = json.load(f)

cfg = InferenceConfig(**cfg_dict)

# Charger le template
with open(cfg.template_path, 'r') as f:
    cfg.template = json.load(f)

# Exécuter l'inférence
result = run_inference(
    model_path=cfg.model_path,
    lora_path=cfg.lora_path,
    input_text="/content/report.pdf",  # Ou texte directement
    config=cfg,
)

# Afficher les résultats
print("Résultat JSON:")
print(json.dumps(result.parsed_json, indent=2, ensure_ascii=False))

if result.confidence_scores:
    print("\nScores de confiance:")
    print(json.dumps(result.confidence_scores, indent=2, ensure_ascii=False))
```

### Exemple avec anonymisation

```python
# Configuration avec anonymisation activée
cfg.apply_anonymization = True

# L'inférence va automatiquement:
# 1. Extraire le texte du PDF
# 2. Anonymiser le texte
# 3. Faire l'inférence

result = run_inference(
    model_path=cfg.model_path,
    lora_path=cfg.lora_path,
    input_text="/content/report.pdf",
    config=cfg,
)
```

## 📝 Notes importantes

1. **4-bit quantization** : Utilisez `load_in_4bit: true` pour économiser la mémoire GPU
2. **Batch inference** : Activez `use_batch_inference: true` pour traiter plus rapidement les grands templates
3. **Anonymisation** : L'anonymisation avec NER peut être lente - désactivez `anonymization_use_ner` si nécessaire
4. **Token Hugging Face** : N'oubliez pas de vous authentifier avec votre token HF

## 🔧 Dépannage

### Erreur de mémoire GPU
- Réduisez `max_new_tokens`
- Activez `load_in_4bit: true`
- Utilisez un modèle plus petit

### Erreur avec LoRA
- Vérifiez que le chemin `lora_path` est correct
- Assurez-vous que les fichiers LoRA sont téléchargés dans Colab

### Inférence trop lente
- Activez `use_batch_inference: true`
- Réduisez `max_measurements_per_batch`
- Utilisez un GPU plus puissant (T4 -> A100)

## ✅ Checklist avant exécution

- [ ] Token Hugging Face configuré
- [ ] Modèle LoRA téléchargé
- [ ] Template JSON disponible
- [ ] PDF de test disponible
- [ ] Configuration JSON créée
- [ ] GPU activé dans Colab (Runtime -> Change runtime type -> GPU)

