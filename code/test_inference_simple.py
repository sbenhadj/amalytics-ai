"""
Test simple pour vérifier que les fonctions d'inférence sont importables et valides.
"""

import sys
from pathlib import Path

# Ajouter src/ au path
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def test_imports():
    """Test que tous les modules sont importables."""
    print("=" * 60)
    print("TEST: Import des modules")
    print("=" * 60)
    
    try:
        from amalytics_ml.config import InferenceConfig
        print("✅ InferenceConfig importé")
        
        from amalytics_ml.models.inference import (
            run_inference,
            InferenceResult,
            _calculate_confidence,
            _run_batch_inference,
        )
        print("✅ Fonctions d'inférence importées")
        
        from amalytics_ml.utils.template_split import (
            split_template_by_measurements,
            deep_merge,
        )
        print("✅ Fonctions de split importées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur d'import: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test la création d'une configuration."""
    print("\n" + "=" * 60)
    print("TEST: Création de configuration")
    print("=" * 60)
    
    try:
        from amalytics_ml.config import InferenceConfig
        
        config = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            template_path="test/template.json",
            use_batch_inference=False,
            max_measurements_per_batch=2,
        )
        
        print(f"✅ Configuration créée")
        print(f"   - use_batch_inference: {config.use_batch_inference}")
        print(f"   - max_measurements_per_batch: {config.max_measurements_per_batch}")
        
        # Test avec batch
        config_batch = InferenceConfig(
            model_path="test/model",
            lora_path="test/lora",
            template_path="test/template.json",
            use_batch_inference=True,
            max_measurements_per_batch=2,
        )
        
        print(f"✅ Configuration batch créée")
        print(f"   - use_batch_inference: {config_batch.use_batch_inference}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_template_split():
    """Test le split de template."""
    print("\n" + "=" * 60)
    print("TEST: Split de template")
    print("=" * 60)
    
    try:
        from amalytics_ml.utils.template_split import split_template_by_measurements
        
        # Template de test simple
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
            print(f"   - Partie {i}: {len(str(part))} caractères")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de split: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Tests de validation du code d'inférence...\n")
    
    success1 = test_imports()
    success2 = test_config()
    success3 = test_template_split()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Imports:        {'✅ PASSÉ' if success1 else '❌ ÉCHOUÉ'}")
    print(f"Configuration:  {'✅ PASSÉ' if success2 else '❌ ÉCHOUÉ'}")
    print(f"Split template: {'✅ PASSÉ' if success3 else '❌ ÉCHOUÉ'}")
    
    if success1 and success2 and success3:
        print("\n🎉 Tous les tests de validation sont passés!")
        print("   Le code est structurellement correct.")
        sys.exit(0)
    else:
        print("\n⚠️ Certains tests ont échoué.")
        sys.exit(1)

