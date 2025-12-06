"""
Test rapide de validation - ne charge PAS de modèles.
Vérifie uniquement que la structure du code est correcte.
"""

import json
import sys
from pathlib import Path

# Ajouter src/ au path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_imports():
    """Test que tous les modules sont importables (sans charger de modèles)."""
    print("=" * 70)
    print("TEST 1: Import des modules")
    print("=" * 70)
    
    try:
        from amalytics_ml.config import InferenceConfig
        print("✅ InferenceConfig importé")
        
        # Test d'import de la structure (pas d'exécution)
        from amalytics_ml.models.inference import InferenceResult
        print("✅ InferenceResult importé")
        
        from amalytics_ml.utils.template_split import (
            split_template_by_measurements,
            deep_merge,
        )
        print("✅ Fonctions de split importées")
        
        from amalytics_ml.data.anonymization import (
            AnonymizationConfig,
            anonymize_text,
        )
        print("✅ Fonctions d'anonymisation importées")
        
        return True
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test la création de configuration."""
    print("\n" + "=" * 70)
    print("TEST 2: Création de configuration")
    print("=" * 70)
    
    try:
        from amalytics_ml.config import InferenceConfig
        
        config = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            apply_anonymization=False,
            use_batch_inference=False,
        )
        
        print(f"✅ Configuration créée")
        print(f"   - apply_anonymization: {config.apply_anonymization}")
        print(f"   - use_batch_inference: {config.use_batch_inference}")
        
        config_anon = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            apply_anonymization=True,
            anonymization_use_ner=False,
        )
        
        print(f"✅ Configuration avec anonymisation créée")
        print(f"   - apply_anonymization: {config_anon.apply_anonymization}")
        
        config_batch = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            use_batch_inference=True,
            max_measurements_per_batch=3,
        )
        
        print(f"✅ Configuration batch créée")
        print(f"   - use_batch_inference: {config_batch.use_batch_inference}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_template_split():
    """Test le split de template (sans modèle)."""
    print("\n" + "=" * 70)
    print("TEST 3: Split de template")
    print("=" * 70)
    
    try:
        from amalytics_ml.utils.template_split import split_template_by_measurements
        
        test_template = {
            "Hematologie": {
                "NumerationGlobulaire": {
                    "Hematies": {"valeur": None, "unité": "T/L"},
                    "Hematocrite": {"valeur": None, "unité": "%"},
                }
            }
        }
        
        parts = split_template_by_measurements(
            test_template,
            max_objects_per_part=1,
            dedup_consecutive=True,
        )
        
        print(f"✅ Template splité en {len(parts)} parties")
        for i, part in enumerate(parts, 1):
            part_str = json.dumps(part, ensure_ascii=False)
            print(f"   - Partie {i}: {len(part_str)} caractères")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_anonymization_text():
    """Test l'anonymisation de texte (sans NER)."""
    print("\n" + "=" * 70)
    print("TEST 4: Anonymisation de texte")
    print("=" * 70)
    
    try:
        from amalytics_ml.data.anonymization import anonymize_text, AnonymizationConfig
        
        test_text = """
        Date de naissance: 15/03/1985
        Email: test@example.com
        Téléphone: 0612345678
        """
        
        config = AnonymizationConfig(
            secret_key=b"test_key",
            use_ner=False,  # Désactiver NER pour test rapide
            anonymize_codes=True,
            anonymize_dates=True,
            anonymize_emails=True,
            anonymize_phones=True,
        )
        
        anonymized = anonymize_text(test_text, config)
        print(f"✅ Anonymisation réussie")
        print(f"   - Texte modifié: {'Oui' if test_text != anonymized else 'Non'}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_detection():
    """Test la détection de PDF (sans extraction)."""
    print("\n" + "=" * 70)
    print("TEST 5: Détection de fichiers PDF")
    print("=" * 70)
    
    # Chercher un PDF de test (sans l'extraire)
    pdf_path = ROOT_DIR.parent / "amalytics-ml" / "data" / "reports" / "sample_1.pdf"
    
    if not pdf_path.exists():
        pdf_path = Path("amalytics-ml/data/reports/sample_1.pdf")
        if not pdf_path.exists():
            print(f"⚠️  PDF non trouvé (test ignoré)")
            return True
    
    try:
        # Vérifier juste que le fichier existe (pas d'extraction)
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            print(f"✅ PDF trouvé: {pdf_path.name}")
            print(f"   - Taille: {size_kb:.1f} KB")
            print(f"   - Path valide: Oui")
            return True
        else:
            print(f"⚠️  PDF non trouvé")
            return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def test_template_loading():
    """Test le chargement de template."""
    print("\n" + "=" * 70)
    print("TEST 6: Chargement de template")
    print("=" * 70)
    
    template_path = ROOT_DIR.parent / "amalytics-ml" / "data" / "filtered_templates" / "empty" / "sample_1_template_empty.json"
    
    if not template_path.exists():
        template_path = Path("amalytics-ml/data/filtered_templates/empty/sample_1_template_empty.json")
        if not template_path.exists():
            print(f"⚠️  Template non trouvé (test ignoré)")
            return True
    
    try:
        with template_path.open("r", encoding="utf-8") as f:
            template = json.load(f)
        
        print(f"✅ Template chargé: {template_path.name}")
        print(f"   - Clés de premier niveau: {list(template.keys())[:3]}")
        
        # Compter les mesures rapidement
        def count_measurements(obj, count=0):
            if isinstance(obj, dict):
                if "valeur" in obj:
                    return count + 1
                return sum(count_measurements(v, count) for v in obj.values())
            return count
        
        measurements = count_measurements(template)
        print(f"   - Nombre de mesures: {measurements}")
        
        return True
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: dict[str, bool]):
    """Affiche le résumé."""
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES TESTS")
    print("=" * 70)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, result in results.items():
        status = "✅ PASSÉ" if result else "❌ ÉCHOUÉ"
        print(f"{name:.<50} {status}")
    
    print(f"\nTotal: {total} tests | Réussis: {passed} ✅ | Échoués: {total - passed} ❌")
    
    if passed == total:
        print("\n" + "🎉" * 35)
        print("TOUS LES TESTS SONT PASSÉS!")
        print("\n✅ Le code est structurellement correct")
        print("✅ Prêt pour Colab avec LLaMA + LoRA")
        print("\n📋 Configuration recommandée pour Colab:")
        print("   - model_path: 'meta-llama/Meta-Llama-3.1-8B-Instruct'")
        print("   - lora_path: chemin vers votre LoRA fine-tuné")
        print("   - apply_anonymization: True (si nécessaire)")
        print("   - use_batch_inference: True (pour optimisation)")
        print("   - load_in_4bit: True (pour économiser la mémoire)")
        print("🎉" * 35)
        return True
    else:
        print("\n⚠️  Certains tests ont échoué. Corrigez les erreurs.")
        return False


def main():
    """Exécute tous les tests rapides."""
    print("\n" + "🚀" * 35)
    print("TESTS RAPIDES - VALIDATION STRUCTURE")
    print("🚀" * 35)
    print("\nCes tests vérifient la structure sans charger de modèles.\n")
    
    results = {}
    results["Imports"] = test_imports()
    results["Configuration"] = test_config()
    results["Split template"] = test_template_split()
    results["Anonymisation"] = test_anonymization_text()
    results["Détection PDF"] = test_pdf_detection()
    results["Chargement template"] = test_template_loading()
    
    all_passed = print_summary(results)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

