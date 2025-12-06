# Guide de validation pour Google Colab

## ⚠️ Points critiques à vérifier

### 1. **Compatibilité avec `bitsandbytes`**

Le code utilise `load_in_4bit` qui est **déprécié** dans les nouvelles versions de `transformers`. 
Pour Colab, vous devrez peut-être utiliser `BitsAndBytesConfig`.

**Solution si erreur :**

Si vous obtenez un warning ou une erreur sur `load_in_4bit`, modifiez `_load_model_and_tokenizer` dans `inference.py` :

```python
from transformers import BitsAndBytesConfig

if cfg.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=cfg.bnb_compute_dtype,
    )
    model_kwargs["quantization_config"] = quantization_config
```

### 2. **Problèmes potentiels identifiés**

#### a) Détection de PDF
- Le code vérifie si un chemin est un PDF en regardant l'extension
- **Vérifier** : Si vous passez un texte qui contient ".pdf" dans le nom, cela peut être mal interprété

#### b) Chargement LoRA
- Si `lora_path` est vide, le code ne charge pas de LoRA (OK)
- Si `lora_path` existe mais est invalide, cela échouera

#### c) Template loading
- Le template doit être chargé AVANT `run_inference`
- Le script `infer.py` le fait déjà, mais dans Colab vous devrez le faire manuellement

### 3. **Checklist de validation pour Colab**

Avant de lancer l'inférence, vérifiez :

- [ ] **Dépendances installées**
  ```python
  !pip install torch transformers accelerate bitsandbytes peft pdfplumber pyffx scikit-learn natsort fpdf2 lxml nltk
  ```

- [ ] **Token Hugging Face configuré**
  ```python
  from huggingface_hub import login
  login("VOTRE_TOKEN")
  ```

- [ ] **GPU activé** (Runtime → Change runtime type → GPU)

- [ ] **Modèle LoRA téléchargé et accessible**
  - Vérifier le chemin vers `lora-output`
  - Vérifier que `adapter_config.json` existe

- [ ] **Template JSON chargé**
  - Vérifier que le template existe
  - Vérifier que c'est un JSON valide

- [ ] **PDF accessible** (si vous utilisez un PDF)
  - Vérifier que le chemin est correct
  - Vérifier les permissions

### 4. **Test progressif dans Colab**

#### Étape 1 : Test d'import
```python
import sys
sys.path.insert(0, '/content/amalytics-ml/code/src')

from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import InferenceResult
print("✅ Imports OK")
```

#### Étape 2 : Test de configuration
```python
config = InferenceConfig(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    lora_path="/content/lora-output",
    apply_anonymization=False,
    use_batch_inference=False,
)
print("✅ Configuration OK")
```

#### Étape 3 : Test de chargement de template
```python
import json
with open('/content/template.json', 'r') as f:
    template = json.load(f)
config.template = template
print("✅ Template chargé")
```

#### Étape 4 : Test d'extraction PDF (sans modèle)
```python
from amalytics_ml.data.anonymization import extract_text_from_pdf
text = extract_text_from_pdf("/content/report.pdf")
print(f"✅ PDF extrait: {len(text)} caractères")
```

#### Étape 5 : Test d'anonymisation (sans modèle)
```python
from amalytics_ml.data.anonymization import anonymize_text, AnonymizationConfig
anon_config = AnonymizationConfig(use_ner=False)
anonymized = anonymize_text(text, anon_config)
print("✅ Anonymisation OK")
```

#### Étape 6 : Test complet avec modèle
```python
result = run_inference(
    model_path=config.model_path,
    lora_path=config.lora_path,
    input_text=text,  # ou chemin PDF
    config=config,
)
print("✅ Inférence réussie!")
```

### 5. **Erreurs courantes et solutions**

| Erreur | Cause probable | Solution |
|--------|---------------|----------|
| `load_in_4bit is deprecated` | Version récente de transformers | Utiliser `BitsAndBytesConfig` |
| `CUDA out of memory` | GPU trop petit | Réduire `max_new_tokens` ou activer 4-bit |
| `FileNotFoundError: LoRA path` | Chemin incorrect | Vérifier le chemin vers lora-output |
| `Template not found` | Template non chargé | Charger le template avant run_inference |
| `JSON decode error` | Template invalide | Valider le JSON avec json.load() |

### 6. **Code de test complet pour Colab**

```python
# ===== SETUP =====
import sys
from pathlib import Path

# Ajouter le code au path
sys.path.insert(0, '/content/amalytics-ml/code/src')

# ===== IMPORTS =====
import json
from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import run_inference
from amalytics_ml.data.anonymization import extract_text_from_pdf

# ===== CONFIGURATION =====
config = InferenceConfig(
    model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
    lora_path="/content/lora-output",  # Ajustez selon votre chemin
    template_path="/content/template.json",  # Ajustez selon votre chemin
    max_new_tokens=3000,
    return_scores=True,
    load_in_4bit=True,
    device_map="auto",
    use_batch_inference=True,
    max_measurements_per_batch=2,
    apply_anonymization=False,  # Activez si nécessaire
)

# ===== CHARGER TEMPLATE =====
with open(config.template_path, 'r') as f:
    config.template = json.load(f)

# ===== INFÉRENCE =====
result = run_inference(
    model_path=config.model_path,
    lora_path=config.lora_path,
    input_text="/content/report.pdf",  # Ou texte directement
    config=config,
)

# ===== RÉSULTATS =====
print("Résultat JSON:")
print(json.dumps(result.parsed_json, indent=2, ensure_ascii=False))

if result.confidence_scores:
    print("\nScores de confiance:")
    print(json.dumps(result.confidence_scores, indent=2, ensure_ascii=False))
```

### 7. **Modifications recommandées pour Colab**

Si vous rencontrez des problèmes, voici les modifications à apporter :

#### Modification 1 : Utiliser BitsAndBytesConfig (si erreur de dépréciation)

Dans `inference.py`, remplacer :
```python
if cfg.load_in_4bit:
    model_kwargs["load_in_4bit"] = True
    model_kwargs["bnb_4bit_compute_dtype"] = cfg.bnb_compute_dtype
```

Par :
```python
if cfg.load_in_4bit:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=cfg.bnb_compute_dtype,
    )
    model_kwargs["quantization_config"] = quantization_config
```

#### Modification 2 : Gestion des erreurs de mémoire

Ajouter une option pour réduire automatiquement `max_new_tokens` si erreur de mémoire.

## ✅ Conclusion

**Le code est structurellement correct**, mais vous devrez tester dans Colab pour valider :
- La compatibilité avec la version de transformers/bitsandbytes
- Le chargement du modèle LoRA
- Les chemins de fichiers dans l'environnement Colab

**Recommandation** : Testez progressivement (étapes 1-6) avant de lancer l'inférence complète.

