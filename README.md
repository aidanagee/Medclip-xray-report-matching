# Medical Image–Report Matching with CLIP
### Multimodal Deep Learning — Contrastive Vision-Language Learning

A CLIP-style dual encoder model that learns a joint embedding space between chest X-ray images and radiology reports. Enables zero-shot disease classification, cross-modal retrieval, and report-guided image search — without requiring manually labeled image data.

Part of a medical AI project series.

---

## Overview

Radiologists translate visual findings from X-rays into natural language reports every day. This project teaches a neural network to learn that same cross-modal mapping using **contrastive learning** — pulling matching image-report pairs together in a shared embedding space while pushing non-matching pairs apart.

### Real-World Applications
- **Zero-shot disease classification** — classify X-rays using text prompts with no labeled image training data
- **Report retrieval** — given a new X-ray, find the most similar historical report in a database
- **Image retrieval** — given a clinical description, find matching X-rays
- **Weakly supervised detection** — localize findings using only report-level supervision

---

## How It Works

Given a batch of N chest X-ray + radiology report pairs, we compute an N×N cosine similarity matrix:

```
              Report_1   Report_2   Report_3
X-ray_1   [   HIGH,      low,       low    ]   ← correct pair
X-ray_2   [   low,       HIGH,      low    ]   ← correct pair
X-ray_3   [   low,       low,       HIGH   ]   ← correct pair
```

The **InfoNCE loss** maximizes diagonal similarity (correct pairs) while minimizing off-diagonal similarity (incorrect pairs). Both encoders are trained end-to-end simultaneously.

---

## Architecture

```
Chest X-ray  ──► [ResNet-50 Backbone] ──► [Projection MLP] ──► image_embedding (256-d, L2-norm)
                                                                          │
                                                               cosine similarity matrix
                                                                          │
Radiology Report ──► [DistilBERT] ──► [CLS token] ──► [Projection MLP] ──► text_embedding (256-d, L2-norm)
```

| Component | Model | Output Dim |
|---|---|---|
| Image Encoder | ResNet-50 (ImageNet pretrained) + MLP projection | 256-d |
| Text Encoder | DistilBERT + MLP projection | 256-d |
| Temperature | Learnable scalar (initialized: ln(1/0.07)) | — |

### Key Design Decisions

| Choice | Reason |
|---|---|
| ResNet-50 backbone | Strong ImageNet pretraining; efficient for medical imaging |
| DistilBERT text encoder | 40% fewer params than BERT, ~97% performance |
| L2 normalization | Cosine similarity = dot product of unit vectors |
| Learnable temperature | Model learns optimal confidence scaling |
| Differential LRs | Pretrained layers (1e-5) vs random projection heads (1e-4) |
| Symmetric InfoNCE | Loss computed image→text AND text→image, averaged |

---

## Dataset

**MIMIC-CXR** — 227,827 chest X-ray studies from Beth Israel Deaconess Medical Center  
Each study includes chest X-ray images + free-text radiology report (Findings + Impression)

Access requires PhysioNet credentialing: https://physionet.org/content/mimic-cxr/2.0.0/

The notebook runs fully on **synthetic data** that mirrors real MIMIC-CXR structure — no download required to explore the full pipeline.

### Pathology Classes

| Finding | Description |
|---|---|
| Normal | No acute cardiopulmonary process |
| Pneumonia | Focal consolidation / airspace opacity |
| Pleural Effusion | Fluid in pleural space |
| Cardiomegaly | Enlarged cardiac silhouette |
| Pneumothorax | Air in pleural space |

---

## Results

| Metric | Score |
|---|---|
| R@1 (Image→Text) | ~0.65 |
| R@5 (Image→Text) | ~0.90 |
| Zero-shot accuracy | ~0.60 |

*Results on synthetic demonstration data. Real MIMIC-CXR training expected to achieve R@1 ~0.45-0.55 (competitive with published baselines on this task).*

---

## Zero-Shot Classification

One of the most powerful capabilities — classify X-rays using text prompts with **no labeled image training data**:

```python
prompts = {
    'Normal': "No acute cardiopulmonary process. Clear lungs bilaterally.",
    'Pneumonia': "Focal consolidation consistent with pneumonia.",
    'Pleural Effusion': "Pleural effusion with blunting of costophrenic angle.",
    'Cardiomegaly': "Enlarged cardiac silhouette. Increased cardiothoracic ratio.",
    'Pneumothorax': "Pneumothorax with visible lung edge. Lung collapse."
}

predicted_class, probs = zero_shot_classify(model, tokenizer, xray_image, prompts, device)
# predicted_class: 'Pneumonia'
# probs: {'Normal': 0.08, 'Pneumonia': 0.71, 'Effusion': 0.09, ...}
```

This mirrors the approach of **CheXzero** (Tiu et al., Nature Biomedical Engineering 2022), which achieved radiologist-level performance using only report supervision.

---

## Project Structure

```
medclip-xray-report-matching/
├── medclip_xray_report_matching.ipynb   # Full notebook
├── README.md
├── synthetic_xrays.png                  # Synthetic X-ray visualization (generated)
├── training_curves.png                  # Loss and retrieval metric curves (generated)
└── similarity_matrix.png               # Embedding space similarity visualization (generated)
```

---

## How to Run

```bash
# Install dependencies
pip install torch torchvision transformers Pillow matplotlib

# Run the notebook
jupyter notebook medclip_xray_report_matching.ipynb
```

**To use real MIMIC-CXR data:**
1. Apply for PhysioNet access at https://physionet.org/content/mimic-cxr/2.0.0/
2. Update `MimicCXRDataset` to load from your downloaded directory
3. Parse reports using the Findings + Impression sections
4. No other changes needed — the rest of the pipeline is identical

---

## Medical AI Project Series

| Project | Modality | Task | Architecture |
|---|---|---|---|
| [Intracranial Aneurysm Detection](https://github.com/aidanagee/aneurysm-detection-rsna) | 3D CT | Segmentation | 3D U-Net |
| [Lung Nodule Detection](https://github.com/aidanagee/lung-nodule-detection-3d-unet) | 3D CT | Segmentation | 3D U-Net |
| [ADR Detection](https://github.com/aidanagee/adr-detection-distilbert) | Text | Classification | DistilBERT |
| **Image-Report Matching (this project)** | Image + Text | Multimodal Retrieval | CLIP-style |

---

## Related Work

| Paper | Contribution |
|---|---|
| CLIP (Radford et al., 2021) | Original contrastive image-text pretraining |
| CheXzero (Tiu et al., 2022) | Zero-shot chest X-ray diagnosis via report supervision |
| MedCLIP (Wang et al., 2022) | Decoupled contrastive learning for medical vision-language |
| BioViL (Bannur et al., 2023) | Phrase grounding for biomedical vision-language models |

---

## References

1. Radford, A., et al. (2021). Learning Transferable Visual Models from Natural Language Supervision. *ICML 2021.*
2. Tiu, E., et al. (2022). Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. *Nature Biomedical Engineering.*
3. Wang, Z., et al. (2022). MedCLIP: Contrastive Learning from Unpaired Medical Images and Text. *EMNLP 2022.*
4. Bannur, S., et al. (2023). Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing. *CVPR 2023.*
5. Johnson, A., et al. (2019). MIMIC-CXR: A large publicly available database of labeled chest radiographs. *PhysioNet.*

---

## Author

**Aidan Agee** — [GitHub](https://github.com/aidanagee) | [LinkedIn](https://linkedin.com/in/adaadaada)
