 🎭 Génération de Visages Haute Résolution avec StyleGAN2-ADA
 
Implémentation de **StyleGAN2-ADA (NVIDIA)** pour générer des visages humains synthétiques haute résolution et produire une interpolation fluide entre deux identités.

---

 Fonctionnalités

- **Chargement du modèle préentraîné** FFHQ (Faces with High Quality) via `ffhq.pkl`
-  **Génération de visages** synthétiques à partir de vecteurs latents aléatoires
-  **Interpolation fluide** entre deux visages dans l'espace latent W
-  **Comparaison multi-résolution** : 1024×1024 · 128×128 · 16×16
-  **Export vidéo** `.avi` affichant les 3 résolutions côte à côte

---

##  Aperçu des résultats

| 16×16 | 128×128 | 1024×1024 |
|-------|---------|-----------|
| Très basse résolution — forme abstraite | Résolution moyenne — détails flous | Haute résolution — texture réaliste |

> *(Ajoute ici des captures d'écran ou un GIF de l'interpolation)*

---

## Lancer le projet

### 1. Ouvrir dans Google Colab
Ce projet tourne sur **Google Colab** avec GPU activé.  
`Exécution → Modifier le type d'exécution → GPU`

### 2. Cloner StyleGAN2-ADA
```bash
!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
%cd stylegan2-ada-pytorch
```

### 3. Télécharger le modèle préentraîné
```bash
!mkdir -p pretrained
!wget -O pretrained/ffhq.pkl https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

### 4. Lancer le notebook
Ouvre et exécute **`generation_des_visages.ipynb`** cellule par cellule.

---

## 📂 Structure du projet

```
stylegan2-face-generation/
├── generation_des_visages.ipynb   # Notebook principal
└── README.md
```

---

##  Pipeline technique

```
Vecteur latent z  →  Mapping Network  →  Espace W
       →  Synthesis Network  →  Image 1024×1024
```

L'interpolation se fait en combinant progressivement deux vecteurs `w[0]` et `w[1]` :
```python
w_interp = w[0] * (1 - t) + w[1] * t   # t ∈ [0, 1]
img = G.synthesis(w_interp, noise_mode='const')
```

---

## 🛠️ Stack technique

| Outil | Rôle |
|-------|------|
| Python 3.x | Langage principal |
| PyTorch | Deep learning & GPU |
| StyleGAN2-ADA (NVIDIA) | Modèle génératif |
| NumPy | Manipulation des vecteurs latents |
| Pillow (PIL) | Traitement des images |
| OpenCV (cv2) | Création de la vidéo d'interpolation |
| Google Colab | Environnement d'exécution GPU |

---

## Résultats

- ✅ Interpolation fluide sur **300 frames** entre deux visages synthétiques
- ✅ Images haute résolution **1024×1024** avec détails réalistes (peau, yeux, cheveux)
- ✅ Vidéo `.avi` combinée affichant les 3 résolutions en parallèle
- ✅ Mise en évidence de la perte d'information à basse résolution (16×16)

---



##  Références

- [StyleGAN2-ADA — NVlabs (GitHub officiel)](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [NVIDIA Research — StyleGAN2-ADA Paper](https://arxiv.org/abs/2006.06676)
- [FFHQ Dataset](https://github.com/NVlabs/ffhq-dataset)
