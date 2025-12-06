# 📤 Guide : Uploader le code sur GitHub et le cloner dans Colab

Ce guide explique comment uploader votre code sur GitHub et le cloner dans Google Colab.

---

## 🚀 Méthode 1 : Via l'interface GitHub (pour débutants)

### Étape 1 : Créer un nouveau repository GitHub

1. Allez sur [github.com](https://github.com)
2. Connectez-vous à votre compte
3. Cliquez sur le bouton **"+"** en haut à droite → **"New repository"**
4. Remplissez les informations :
   - **Repository name** : `amalytics-ml` (ou le nom que vous voulez)
   - **Description** : (optionnel) "LLaMA fine-tuning for medical report extraction"
   - **Visibility** : Choisissez **Public** ou **Private**
   - ⚠️ **NE PAS cocher** "Initialize this repository with a README" (on va uploader le code)
5. Cliquez sur **"Create repository"**

### Étape 2 : Uploader vos fichiers

1. Sur la page de votre nouveau repository, vous verrez une section "quick setup"
2. Choisissez **"uploading an existing file"**
3. Cliquez sur **"uploading an existing file"**
4. **Glissez-déposez** votre dossier `code/` entier ou sélectionnez les fichiers
5. Ajoutez un message de commit : "Initial commit: Amalytics ML codebase"
6. Cliquez sur **"Commit changes"**

✅ **Votre code est maintenant sur GitHub !**

---

## 💻 Méthode 2 : Via Git en ligne de commande (recommandé)

### Étape 1 : Initialiser Git dans votre projet

Ouvrez un terminal dans le dossier de votre projet :

```bash
# Naviguer vers le dossier code
cd "C:\Saima Work\AI4Cure\Code\amalytics-ml\code"

# Initialiser Git
git init

# Créer un fichier .gitignore (important pour ne pas uploader des fichiers inutiles)
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo "*.egg-info/" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "*.log" >> .gitignore
echo ".DS_Store" >> .gitignore
echo "models/" >> .gitignore  # Exclure les modèles (trop gros)
echo "data/" >> .gitignore  # Exclure les données si nécessaire

# Ajouter tous les fichiers
git add .

# Créer le premier commit
git commit -m "Initial commit: Amalytics ML codebase"
```

### Étape 2 : Créer le repository sur GitHub

1. Allez sur [github.com](https://github.com)
2. Créez un nouveau repository (comme dans Méthode 1, Étape 1)
3. **NE PAS** initialiser avec README
4. Copiez l'URL du repository (ex: `https://github.com/VOTRE_USERNAME/amalytics-ml.git`)

### Étape 3 : Connecter et pousser le code

```bash
# Dans le même terminal
# Remplacer VOTRE_USERNAME et amalytics-ml par vos valeurs
git remote add origin https://github.com/VOTRE_USERNAME/amalytics-ml.git

# Renommer la branche principale (si nécessaire)
git branch -M main

# Pousser le code vers GitHub
git push -u origin main
```

Si vous êtes demandé de vous authentifier :
- **Username** : Votre nom d'utilisateur GitHub
- **Password** : Utilisez un **Personal Access Token** (pas votre mot de passe)
  - Créez-en un : GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Donnez-lui les permissions `repo`

✅ **Votre code est maintenant sur GitHub !**

---

## 📥 Cloner dans Google Colab

### Option A : Cloner directement dans une cellule Colab

Une fois votre code sur GitHub, dans votre notebook Colab :

```python
# Dans une cellule du notebook Colab
!git clone https://github.com/VOTRE_USERNAME/amalytics-ml.git /content/amalytics-ml

# Vérifier que le code est bien téléchargé
import os
print("✅ Code cloné!")
print(f"📁 Contenu: {os.listdir('/content/amalytics-ml')}")
```

### Option B : Cloner depuis le notebook créé

Dans le notebook `COLAB_INFERENCE_NOTEBOOK.ipynb`, modifiez la cellule "ÉTAPE 4" :

```python
# Option A : Depuis GitHub (décommentez et modifiez)
!git clone https://github.com/VOTRE_USERNAME/amalytics-ml.git /content/amalytics-ml

# Ajouter le code au path Python
import sys
from pathlib import Path

code_dir = Path("/content/amalytics-ml/code")
src_dir = code_dir / "src"
sys.path.insert(0, str(src_dir))

print(f"✅ Code cloné depuis GitHub")
print(f"✅ Code ajouté au path: {src_dir}")
```

---

## 🔒 Si votre repository est Private

Si vous avez créé un repository **private**, vous devrez vous authentifier :

### Méthode 1 : Personal Access Token dans l'URL

```python
# Remplacer VOTRE_TOKEN par votre Personal Access Token
!git clone https://VOTRE_TOKEN@github.com/VOTRE_USERNAME/amalytics-ml.git /content/amalytics-ml
```

### Méthode 2 : Configuration Git dans Colab

```python
!git config --global user.name "Votre Nom"
!git config --global user.email "votre@email.com"

# Cloner avec authentification
import os
os.environ['GIT_ASKPASS'] = 'echo'
os.environ['GIT_USERNAME'] = 'VOTRE_USERNAME'
os.environ['GIT_PASSWORD'] = 'VOTRE_TOKEN'

!git clone https://github.com/VOTRE_USERNAME/amalytics-ml.git /content/amalytics-ml
```

---

## 📝 Créer un fichier .gitignore recommandé

Créez un fichier `.gitignore` dans votre dossier `code/` pour éviter d'uploader des fichiers inutiles :

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Jupyter Notebook
.ipynb_checkpoints/
*.ipynb

# Environnements virtuels
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log

# Modèles et données (optionnel - commentez si vous voulez les inclure)
models/
*.pt
*.pth
*.bin
*.safetensors

# Données (optionnel - commentez si vous voulez les inclure)
data/
*.pdf

# OS
.DS_Store
Thumbs.db

# Colab
/content/
```

---

## ✅ Checklist finale

- [ ] Repository GitHub créé
- [ ] Code uploadé sur GitHub
- [ ] `.gitignore` créé (recommandé)
- [ ] README.md créé (optionnel mais recommandé)
- [ ] Repository testé (vous pouvez voir vos fichiers sur GitHub)
- [ ] Prêt à cloner dans Colab

---

## 🎯 Exemple complet dans Colab

Voici un exemple complet pour cloner et utiliser votre code dans Colab :

```python
# ÉTAPE 1 : Cloner depuis GitHub
!git clone https://github.com/VOTRE_USERNAME/amalytics-ml.git /content/amalytics-ml

# ÉTAPE 2 : Vérifier la structure
import os
from pathlib import Path

repo_path = Path("/content/amalytics-ml/code")
print("📁 Structure du repository:")
for item in sorted(repo_path.rglob("*"))[:20]:  # Afficher les 20 premiers
    if item.is_file():
        print(f"  📄 {item.relative_to(repo_path)}")

# ÉTAPE 3 : Ajouter au path Python
import sys
src_dir = repo_path / "src"
sys.path.insert(0, str(src_dir))
print(f"\n✅ Code ajouté au path: {src_dir}")

# ÉTAPE 4 : Vérifier les imports
try:
    from amalytics_ml.config import InferenceConfig
    print("✅ Import réussi!")
except ImportError as e:
    print(f"❌ Erreur: {e}")
```

---

## 🆘 Problèmes courants

### Erreur : "repository not found"
- Vérifiez que l'URL est correcte
- Si le repo est private, utilisez un token d'authentification
- Vérifiez que le repository existe bien sur GitHub

### Erreur : "authentication failed"
- Utilisez un Personal Access Token au lieu du mot de passe
- Créez un token : GitHub → Settings → Developer settings → Personal access tokens

### Fichiers trop gros
- GitHub a une limite de 100 MB par fichier
- Utilisez `.gitignore` pour exclure les modèles et données
- Utilisez Git LFS pour les gros fichiers si nécessaire

---

## 📚 Ressources utiles

- [GitHub Docs](https://docs.github.com/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [Creating a Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

---

**Maintenant vous pouvez cloner votre code dans Colab facilement ! 🚀**

