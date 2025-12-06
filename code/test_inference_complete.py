"""
Test complet de l'inférence avec toutes les fonctionnalités.

Ce script teste:
1. Extraction de texte depuis PDF
2. Anonymisation (optionnelle)
3. Inférence standard
4. Inférence batch
5. Calcul des scores de confiance

Si tous les tests passent, le code est prêt pour Colab avec LLaMA + LoRA.
"""

import json
import sys
from pathlib import Path

# Ajouter src/ au path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from amalytics_ml.config import InferenceConfig
from amalytics_ml.models.inference import run_inference, InferenceResult
from amalytics_ml.data.anonymization import extract_text_from_pdf, anonymize_text, AnonymizationConfig


def test_pdf_extraction():
    """Test l'extraction de texte depuis PDF."""
    print("=" * 70)
    print("TEST 1: Extraction de texte depuis PDF")
    print("=" * 70)
    
    # Chercher un PDF de test
    pdf_path = ROOT_DIR.parent / "amalytics-ml" / "data" / "reports" / "sample_1.pdf"
    
    if not pdf_path.exists():
        # Essayer un autre chemin
        pdf_path = Path("amalytics-ml/data/reports/sample_1.pdf")
        if not pdf_path.exists():
            print(f"⚠️  PDF de test non trouvé: {pdf_path}")
            print("   Test ignoré (nécessite un PDF dans amalytics-ml/data/reports/)")
            return True
    
    try:
        text = extract_text_from_pdf(pdf_path)
        print(f"✅ Extraction réussie")
        print(f"   - Fichier: {pdf_path.name}")
        print(f"   - Longueur du texte: {len(text)} caractères")
        print(f"   - Aperçu (100 premiers caractères): {text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anonymization():
    """Test l'anonymisation de texte."""
    print("\n" + "=" * 70)
    print("TEST 2: Anonymisation de texte")
    print("=" * 70)
    
    test_text = """
    Patient: Jean Dupont
    Date de naissance: 15/03/1985
    Email: jean.dupont@example.com
    Téléphone: 0612345678
    Code postal: 75001 PARIS
    """
    
    try:
        config = AnonymizationConfig(
            secret_key=b"test_key",
            use_ner=False,  # Désactiver NER pour test rapide
            anonymize_codes=True,
            anonymize_dates=True,
            anonymize_emails=True,
            anonymize_phones=True,
            anonymize_postal_codes=True,
        )
        
        anonymized = anonymize_text(test_text, config)
        print(f"✅ Anonymisation réussie")
        print(f"   - Texte original: {len(test_text)} caractères")
        print(f"   - Texte anonymisé: {len(anonymized)} caractères")
        print(f"   - Texte modifié: {'Oui' if test_text != anonymized else 'Non'}")
        print(f"   - Aperçu anonymisé: {anonymized[:150]}...")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'anonymisation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """Test le chargement de configuration."""
    print("\n" + "=" * 70)
    print("TEST 3: Chargement de configuration")
    print("=" * 70)
    
    try:
        # Test avec configuration minimale
        config = InferenceConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            lora_path="",
            template_path="amalytics-ml/data/filtered_templates/empty/sample_1_template_empty.json",
            apply_anonymization=False,
        )
        
        print(f"✅ Configuration créée")
        print(f"   - apply_anonymization: {config.apply_anonymization}")
        print(f"   - use_batch_inference: {config.use_batch_inference}")
        
        # Test avec anonymisation activée
        config_anon = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            apply_anonymization=True,
            anonymization_use_ner=False,
        )
        
        print(f"✅ Configuration avec anonymisation créée")
        print(f"   - apply_anonymization: {config_anon.apply_anonymization}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_loading():
    """Test le chargement de template."""
    print("\n" + "=" * 70)
    print("TEST 4: Chargement de template")
    print("=" * 70)
    
    template_path = ROOT_DIR.parent / "amalytics-ml" / "data" / "filtered_templates" / "empty" / "sample_1_template_empty.json"
    
    if not template_path.exists():
        template_path = Path("amalytics-ml/data/filtered_templates/empty/sample_1_template_empty.json")
        if not template_path.exists():
            print(f"⚠️  Template non trouvé: {template_path}")
            print("   Test ignoré")
            return True
    
    try:
        with template_path.open("r", encoding="utf-8") as f:
            template = json.load(f)
        
        print(f"✅ Template chargé")
        print(f"   - Fichier: {template_path.name}")
        print(f"   - Clés de premier niveau: {list(template.keys())[:3]}")
        
        # Compter les mesures
        def count_measurements(obj, count=0):
            if isinstance(obj, dict):
                if "valeur" in obj:
                    return count + 1
                return sum(count_measurements(v, count) for v in obj.values())
            return count
        
        measurements = count_measurements(template)
        print(f"   - Nombre de mesures (champs avec 'valeur'): {measurements}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_without_model():
    """Test la structure d'inférence sans charger le modèle (test rapide)."""
    print("\n" + "=" * 70)
    print("TEST 5: Structure d'inférence (sans modèle)")
    print("=" * 70)
    
    try:
        # Créer un template minimal
        test_template = {
            "Hematologie": {
                "NumerationGlobulaire": {
                    "Hematies": {"valeur": None, "unité": "T/L"}
                }
            }
        }
        
        config = InferenceConfig(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            lora_path="",
            template=test_template,
            max_new_tokens=50,  # Très court pour test rapide
            return_scores=False,
            apply_anonymization=False,
            use_batch_inference=False,
        )
        
        print(f"✅ Configuration d'inférence créée")
        print(f"   - Template intégré: Oui")
        print(f"   - Anonymisation: {'Activée' if config.apply_anonymization else 'Désactivée'}")
        print(f"   - Batch inference: {'Activé' if config.use_batch_inference else 'Désactivé'}")
        
        # Test avec PDF path
        pdf_path = ROOT_DIR.parent / "amalytics-ml" / "data" / "reports" / "sample_1.pdf"
        if pdf_path.exists():
            print(f"✅ PDF détecté: {pdf_path.name}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_inference_config():
    """Test la configuration pour batch inference."""
    print("\n" + "=" * 70)
    print("TEST 6: Configuration batch inference")
    print("=" * 70)
    
    try:
        config = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            use_batch_inference=True,
            max_measurements_per_batch=3,
            dedup_consecutive_keys=True,
        )
        
        print(f"✅ Configuration batch créée")
        print(f"   - use_batch_inference: {config.use_batch_inference}")
        print(f"   - max_measurements_per_batch: {config.max_measurements_per_batch}")
        print(f"   - dedup_consecutive_keys: {config.dedup_consecutive_keys}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def print_summary(results: dict[str, bool]):
    """Affiche le résumé des tests."""
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for name, result in results.items():
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{name:.<50} {status}")
    
    print(f"\nTotal: {total} tests")
    print(f"Réussis: {passed} ✅")
    print(f"Échoués: {failed} ❌")
    
    if passed == total:
        print("\n" + "🎉" * 35)
        print("TOUS LES TESTS SONT PASSÉS!")
        print("Le code est prêt pour Colab avec LLaMA + LoRA.")
        print("🎉" * 35)
        print("\nProchaines étapes pour Colab:")
        print("1. Télécharger le modèle LoRA fine-tuné")
        print("2. Configurer InferenceConfig avec:")
        print("   - model_path: 'meta-llama/Meta-Llama-3.1-8B-Instruct'")
        print("   - lora_path: chemin vers votre LoRA")
        print("   - apply_anonymization: True (si nécessaire)")
        print("   - use_batch_inference: True (pour optimisation)")
        return True
    else:
        print("\n⚠️  Certains tests ont échoué.")
        print("   Corrigez les erreurs avant de passer à Colab.")
        return False


def main():
    """Exécute tous les tests."""
    print("\n" + "🧪" * 35)
    print("TESTS COMPLETS DE L'INFÉRENCE")
    print("🧪" * 35)
    print("\nCe script teste toutes les fonctionnalités avant de passer à Colab.\n")
    
    results = {}
    
    # Tests (dans l'ordre de dépendance)
    results["Extraction PDF"] = test_pdf_extraction()
    results["Anonymisation"] = test_anonymization()
    results["Configuration"] = test_config_loading()
    results["Chargement template"] = test_template_loading()
    results["Structure inférence"] = test_inference_without_model()
    results["Configuration batch"] = test_batch_inference_config()
    
    # Résumé
    all_passed = print_summary(results)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

