<h1 align="center">Engaging preference optimization alignment in large language model for continual radiology report generation: A hybrid approach</h1>

<p align="center">
  <strong>
    <a href="https://scholar.google.com/citations?user=FeMCtswAAAAJ&hl=en">Amaan Izhar</a>, 
    <a href="https://scholar.google.com.my/citations?user=IgUMlGcAAAAJ&hl=en">Norisma Idris</a>, 
    <a href="https://scholar.google.com/citations?user=TyH59tkAAAAJ&hl=en">Nurul Japar</a>
  </strong>
  <br/><br/>
  <a href="https://link.springer.com/article/10.1007/s12559-025-10404-6">
    <img src="https://img.shields.io/badge/Read%20Paper-Springer-brightgreen?style=for-the-badge" alt="Read Paper Badge">
  </a>
</p>

## 📄 Abstract
<p align="justify">Large language models (LLMs) remain relatively underutilized in medical imaging, particularly in radiology, which is essential for disease diagnosis and management. Nonetheless, radiology report generation (RRG) is a time-consuming task that can result in delays and inconsistencies. To address these challenges, we present a novel hybrid approach that integrates multi-modal radiology information and preference optimization alignment in LLM for continual RRG. Our method integrates a pre-trained small multi-modal model to analyze radiology images and generate an initial report, which is subsequently refined and aligned by an LLM using odds ratio preference optimization (ORPO) and with historical patient data and assessments to mimic radiologist-like responses, bypassing reinforcement learning from human feedback-based (RLHF) alignment. This two-stage fusion—supervised fine-tuning followed by preference optimization—ensures high accuracy while minimizing hallucinations and errors. We also propose a data field curation strategy extendable to various other RRG modality datasets, focusing on selecting relevant responses for preference alignment. We evaluate our approach on two public datasets, achieving state-of-the-art performance with average Bleu scores of 0.375 and 0.647, Meteor scores of 0.495 and 0.714, Rouge-L scores of 0.483 and 0.732, and average F1-RadGraph scores of 0.488 and 0.487, for chest X-rays and lung CT scan datasets, respectively. We further provide in-depth qualitative analyses and ablation studies to explain the workings of our model and grasp the clinical relevance for RRG. This work presents the first application of preference optimization in continual RRG, representing a significant advancement in automating clinically reliable report generation. By reducing cognitive burdens on radiologists through AI-powered reasoning and alignment in LLMs, the proposed model improves decision-making, perception, and diagnostic precision, streamlining workflows and enhancing patient care.</p>

---

## 🏗️ Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Model Architecture"/>
</p>

---

## 📚 Citation

If you find this work useful, please consider citing our paper and giving this repository a ⭐:

```bibtex
@article{izhar2025r2gpoallm,
  title={Engaging preference optimization alignment in large language model for continual radiology report generation: A hybrid approach},
  author={Izhar, Amaan and Idris, Norisma and Japar, Nurul},
  journal={Cognitive Computation},
  volume={17},
  number={1},
  pages={53},
  year={2025},
  publisher={Springer},
  doi={10.1007/s12559-025-10404-6}
}
```

---

## 🛠️ Reproducibility

### ✅ System Requirements

| Component        | Specification                      |
|------------------|------------------------------------|
| OS               | Ubuntu 22.04                       |
| GPU              | ≥ 24 GB VRAM                       |
| RAM              | ≥ 16 GB                            |
| Disk Space       | ≥ 200 GB                           |
| Env Modules      | Miniconda, Python 3.10             |
| Dependencies     | CUDA ≥ 12.1, Huggingface token     |

---

### 📦 Environment Setup
**Run**:
```bash
# Clone the repo
git clone https://github.com/AI-14/r2gpoallm.git
cd r2gpoallm

# Create and activate conda environment
conda create -n env python=3.10 --yes
conda activate env
conda install pip
pip install -r requirements.txt

# Update specific package
pip install -U datasets
```

---

### 📁 Dataset Setup

1. Follow instructions from the [PKATransNet](https://github.com/AI-14/pkatransnet) repository.
2. After running experiments there, rename `beam-search-predictions.csv` to `test_p.csv`.
3. Move `test_p.csv` into the correct `datasets/<sub-directory>/` folder in this repository.
4. Copy the entire `datasets` folder from PKATransNet into this repository.

---

### 🔬 Running Experiments

#### IUXRAY

Update your Huggingface token in the following files:
- `scripts/iuxray/base_sft_like.sh`
- `scripts/iuxray/base_po_like.sh`
- `scripts/iuxray/sft.sh`
- `scripts/iuxray/po.sh`

**Run**:
```bash
source scripts/iuxray/pp.sh                  
source scripts/iuxray/base_sft_like.sh       
source scripts/iuxray/base_po_like.sh        
source scripts/iuxray/sft.sh                 
source scripts/iuxray/po.sh                  
```

---

#### COVCTR

Update your Huggingface token in the following files:
- `scripts/covctr/base_sft_like.sh`
- `scripts/covctr/base_po_like.sh`
- `scripts/covctr/sft.sh`
- `scripts/covctr/po.sh`

**Run**:
```bash
source scripts/covctr/pp.sh                  
source scripts/covctr/base_sft_like.sh       
source scripts/covctr/base_po_like.sh        
source scripts/covctr/sft.sh                 
source scripts/covctr/po.sh
```

---

## 🧹 Clean Up

**Run**:
```bash
cd ..
conda deactivate
conda remove --name env --all
rm -r r2gpoallm
```

---

> ⚠️ **Note:** Due to the small dataset sizes and sensitivity to random seed initialization, results may vary slightly across runs. For rigorous experiments, consider multiple seed evaluations.
