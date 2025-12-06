"""
Script de test pour vérifier le bon fonctionnement de l'inférence.
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
from amalytics_ml.models.inference import run_inference

def test_standard_inference():
    """Test l'inférence standard (non-batch)."""
    print("=" * 60)
    print("TEST 1: Inférence standard (non-batch)")
    print("=" * 60)
    
    # Configuration simple
    config = InferenceConfig(
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_path="",
        template_path="amalytics-ml/data/filtered_templates/empty/sample_1_template_empty.json",
        max_new_tokens=256,  # Réduit pour test rapide
        do_sample=False,
        return_scores=True,
        load_in_4bit=False,
        device_map="cpu",
        use_batch_inference=False,
    )
    
    # Charger le template
    template_path = ROOT_DIR.parent / config.template_path
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as f:
            config.template = json.load(f)
    else:
        print(f"❌ Template non trouvé: {template_path}")
        return False
    
    # Texte de test simple
    input_text = """
    HEMATOLOGIE
    Hématies: 4.92 T/L
    Hématocrite: 37.8%
    Leucocytes: 8.4 G/L
    Plaquettes: 213 G/L
    
    BIOCHIMIE
    Sodium: 142.8 mmol/L
    Urée: 4.8 mmol/L
    """
    
    try:
        result = run_inference(
            model_path=config.model_path,
            lora_path=config.lora_path,
            input_text=input_text,
            config=config,
        )
        
        print(f"✅ Inférence réussie!")
        print(f"   - JSON parsé: {len(str(result.parsed_json))} caractères")
        print(f"   - Scores de confiance: {'Oui' if result.confidence_scores else 'Non'}")
        
        if result.parsed_json:
            print(f"   - Structure JSON valide: Oui")
            # Afficher les premières clés
            keys = list(result.parsed_json.keys())[:3]
            print(f"   - Premières clés: {keys}")
        
        if result.confidence_scores:
            conf_keys = list(result.confidence_scores.keys())[:3]
            print(f"   - Exemples de scores: {conf_keys}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'inférence: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_inference():
    """Test l'inférence en mode batch."""
    print("\n" + "=" * 60)
    print("TEST 2: Inférence en mode batch")
    print("=" * 60)
    
    # Configuration batch
    config = InferenceConfig(
        model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_path="",
        template_path="amalytics-ml/data/filtered_templates/empty/sample_1_template_empty.json",
        max_new_tokens=256,
        do_sample=False,
        return_scores=True,
        load_in_4bit=False,
        device_map="cpu",
        use_batch_inference=True,
        max_measurements_per_batch=2,
        dedup_consecutive_keys=True,
    )
    
    # Charger le template
    template_path = ROOT_DIR.parent / config.template_path
    if template_path.exists():
        with template_path.open("r", encoding="utf-8") as f:
            config.template = json.load(f)
    else:
        print(f"❌ Template non trouvé: {template_path}")
        return False
    
    # Texte de test
    input_text = """
    HEMATOLOGIE
    Hématies: 4.92 T/L
    Hématocrite: 37.8%
    Leucocytes: 8.4 G/L
    """
    
    try:
        result = run_inference(
            model_path=config.model_path,
            lora_path=config.lora_path,
            input_text=input_text,
            config=config,
        )
        
        print(f"✅ Inférence batch réussie!")
        print(f"   - JSON parsé: {len(str(result.parsed_json))} caractères")
        print(f"   - Scores de confiance: {'Oui' if result.confidence_scores else 'Non'}")
        print(f"   - Texte brut: {'Oui' if result.raw_text else 'Non'}")
        
        if result.parsed_json:
            print(f"   - Structure JSON valide: Oui")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'inférence batch: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🧪 Démarrage des tests d'inférence...\n")
    
    success1 = test_standard_inference()
    success2 = test_batch_inference()
    
    print("\n" + "=" * 60)
    print("RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Test standard: {'✅ PASSÉ' if success1 else '❌ ÉCHOUÉ'}")
    print(f"Test batch:    {'✅ PASSÉ' if success2 else '❌ ÉCHOUÉ'}")
    
    if success1 and success2:
        print("\n🎉 Tous les tests sont passés!")
        sys.exit(0)
    else:
        print("\n⚠️ Certains tests ont échoué.")
        sys.exit(1)

