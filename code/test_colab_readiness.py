"""
Test de validation pour vérifier la préparation au déploiement sur Colab.

Ce test vérifie :
1. La structure du code
2. Les dépendances nécessaires
3. Les problèmes potentiels connus
4. La compatibilité avec Colab

⚠️ IMPORTANT: Ce test NE charge PAS de modèles.
Pour valider complètement, vous devrez tester dans Colab avec un vrai modèle.
"""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def check_imports():
    """Vérifie que tous les imports fonctionnent."""
    print("=" * 70)
    print("VÉRIFICATION 1: Imports des modules")
    print("=" * 70)
    
    issues = []
    
    try:
        from amalytics_ml.config import InferenceConfig
        print("✅ InferenceConfig importé")
    except Exception as e:
        print(f"❌ Erreur import InferenceConfig: {e}")
        issues.append("InferenceConfig import")
    
    try:
        from amalytics_ml.models.inference import InferenceResult, run_inference
        print("✅ Fonctions d'inférence importées")
    except Exception as e:
        print(f"❌ Erreur import inference: {e}")
        issues.append("Inference functions import")
    
    try:
        from amalytics_ml.data.anonymization import (
            extract_text_from_pdf,
            anonymize_text,
            AnonymizationConfig,
        )
        print("✅ Fonctions d'anonymisation importées")
    except Exception as e:
        print(f"❌ Erreur import anonymization: {e}")
        issues.append("Anonymization import")
    
    try:
        from amalytics_ml.utils.template_split import (
            split_template_by_measurements,
            deep_merge,
        )
        print("✅ Fonctions de split importées")
    except Exception as e:
        print(f"❌ Erreur import template_split: {e}")
        issues.append("Template split import")
    
    # Vérifier les dépendances critiques
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'pdfplumber': 'PDF extraction',
        'pyffx': 'FPE encryption',
    }
    
    print("\nVérification des dépendances:")
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ⚠️  {name} non installé")
            issues.append(f"Dépendance manquante: {module}")
    
    return len(issues) == 0, issues


def check_config_structure():
    """Vérifie la structure de configuration."""
    print("\n" + "=" * 70)
    print("VÉRIFICATION 2: Structure de configuration")
    print("=" * 70)
    
    issues = []
    
    try:
        from amalytics_ml.config import InferenceConfig
        
        # Test configuration minimale
        config = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
        )
        
        # Vérifier que les champs d'anonymisation existent
        required_fields = [
            'apply_anonymization',
            'anonymization_secret_key',
            'anonymization_use_ner',
            'use_batch_inference',
            'max_measurements_per_batch',
        ]
        
        for field in required_fields:
            if not hasattr(config, field):
                issues.append(f"Champ manquant: {field}")
                print(f"❌ Champ manquant: {field}")
            else:
                print(f"  ✅ {field}: {getattr(config, field)}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False, [str(e)]


def check_pdf_detection_logic():
    """Vérifie la logique de détection PDF."""
    print("\n" + "=" * 70)
    print("VÉRIFICATION 3: Logique de détection PDF")
    print("=" * 70)
    
    issues = []
    
    try:
        from amalytics_ml.models.inference import run_inference
        from amalytics_ml.config import InferenceConfig
        
        # Test avec différents types d'input
        config = InferenceConfig(
            model_path="test/model",
            lora_path="",
            template={},
        )
        
        # Simuler la détection (sans vraiment appeler run_inference)
        test_cases = [
            ("/path/to/file.pdf", True, "Chemin PDF"),
            ("file.pdf", True, "Fichier PDF simple"),
            ("This is text", False, "Texte simple"),
            ("Report with .pdf in text", False, "Texte contenant .pdf"),
        ]
        
        print("Test de détection (simulation):")
        for input_val, should_be_pdf, description in test_cases:
            from pathlib import Path
            path_obj = Path(input_val)
            is_pdf = path_obj.suffix.lower() == '.pdf' and (
                '/' in input_val or '\\' in input_val or len(input_val) < 260
            )
            
            if is_pdf == should_be_pdf:
                print(f"  ✅ {description}: {is_pdf}")
            else:
                print(f"  ⚠️  {description}: détecté comme {is_pdf}, attendu {should_be_pdf}")
                issues.append(f"Détection PDF incorrecte: {description}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False, [str(e)]


def check_potential_colab_issues():
    """Vérifie les problèmes potentiels spécifiques à Colab."""
    print("\n" + "=" * 70)
    print("VÉRIFICATION 4: Problèmes potentiels Colab")
    print("=" * 70)
    
    issues = []
    warnings = []
    
    # Vérifier la configuration 4-bit
    try:
        from amalytics_ml.models.inference import _load_model_and_tokenizer
        import inspect
        
        source = inspect.getsource(_load_model_and_tokenizer)
        
        # Vérifier si BitsAndBytesConfig est utilisé
        if 'BitsAndBytesConfig' not in source and 'load_in_4bit' in source:
            warnings.append(
                "⚠️  Le code utilise 'load_in_4bit' directement. "
                "Dans les nouvelles versions de transformers, cela est déprécié. "
                "Vous devrez peut-être utiliser BitsAndBytesConfig dans Colab."
            )
            print(warnings[-1])
        
        print("  ✅ Fonction de chargement de modèle trouvée")
        
    except Exception as e:
        issues.append(f"Impossible de vérifier _load_model_and_tokenizer: {e}")
    
    # Vérifier la gestion des chemins
    print("\nVérifications de chemins:")
    
    # Test avec chemin Colab
    colab_paths = [
        "/content/lora-output",
        "/content/drive/MyDrive/lora-output",
    ]
    
    for path_str in colab_paths:
        path = Path(path_str)
        if path.parts[0] == 'content' or 'drive' in path.parts:
            print(f"  ✅ Format de chemin Colab reconnu: {path_str}")
    
    return len(issues) == 0, issues, warnings


def print_colab_checklist():
    """Affiche le checklist pour Colab."""
    print("\n" + "=" * 70)
    print("CHECKLIST POUR COLAB")
    print("=" * 70)
    
    checklist = [
        ("Installer les dépendances", "!pip install torch transformers accelerate bitsandbytes peft pdfplumber pyffx"),
        ("Authentifier Hugging Face", "from huggingface_hub import login; login('TOKEN')"),
        ("Activer le GPU", "Runtime → Change runtime type → GPU"),
        ("Télécharger le modèle LoRA", "Vérifier que lora-output/ existe avec adapter_config.json"),
        ("Préparer le template JSON", "Vérifier que le template est valide"),
        ("Tester les imports", "from amalytics_ml.config import InferenceConfig"),
        ("Tester l'extraction PDF", "from amalytics_ml.data.anonymization import extract_text_from_pdf"),
        ("Lancer l'inférence", "Utiliser run_inference() avec la config"),
    ]
    
    for i, (task, command) in enumerate(checklist, 1):
        print(f"{i}. {task}")
        if command:
            print(f"   → {command}")
    
    print("\n⚠️  PROBLÈMES POTENTIELS À SURVEILLER:")
    print("   1. Warning 'load_in_4bit is deprecated' → Utiliser BitsAndBytesConfig")
    print("   2. CUDA out of memory → Réduire max_new_tokens ou activer 4-bit")
    print("   3. LoRA path not found → Vérifier le chemin dans Colab")
    print("   4. Template not loaded → Charger le template avant run_inference")


def main():
    """Exécute toutes les vérifications."""
    print("\n" + "🔍" * 35)
    print("VALIDATION POUR COLAB")
    print("🔍" * 35)
    print("\nCe script vérifie la préparation du code pour Colab.\n")
    
    results = {}
    all_issues = []
    all_warnings = []
    
    # Vérifications
    ok1, issues1 = check_imports()
    results["Imports"] = ok1
    all_issues.extend(issues1)
    
    ok2, issues2 = check_config_structure()
    results["Configuration"] = ok2
    all_issues.extend(issues2)
    
    ok3, issues3 = check_pdf_detection_logic()
    results["Détection PDF"] = ok3
    all_issues.extend(issues3)
    
    ok4, issues4, warnings = check_potential_colab_issues()
    results["Problèmes Colab"] = ok4
    all_issues.extend(issues4)
    all_warnings.extend(warnings)
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    
    for name, ok in results.items():
        status = "✅ OK" if ok else "❌ PROBLÈME"
        print(f"{name:.<50} {status}")
    
    if all_warnings:
        print("\n⚠️  AVERTISSEMENTS:")
        for warning in all_warnings:
            print(f"   {warning}")
    
    if all_issues:
        print("\n❌ PROBLÈMES DÉTECTÉS:")
        for issue in all_issues:
            print(f"   - {issue}")
    
    print_colab_checklist()
    
    print("\n" + "=" * 70)
    if not all_issues:
        print("✅ STRUCTURE DU CODE VALIDÉE")
        print("\nLe code est structurellement prêt pour Colab.")
        print("⚠️  MAIS vous devez tester avec un vrai modèle dans Colab pour valider:")
        print("   - Le chargement du modèle et LoRA")
        print("   - La compatibilité avec bitsandbytes")
        print("   - Les chemins de fichiers dans Colab")
        print("\nConsultez COLAB_VALIDATION.md pour le guide complet.")
        sys.exit(0)
    else:
        print("❌ PROBLÈMES DÉTECTÉS - Corrigez avant Colab")
        sys.exit(1)


if __name__ == "__main__":
    main()

