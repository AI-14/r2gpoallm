<h1 align="center">Engaging Preference Optimization Alignment in Large Language Model for Continual Radiology Report Generation: A Hybrid Approach</h1>

<p align="center">
  <strong>
    <a href="https://scholar.google.com/citations?user=FeMCtswAAAAJ&hl=en">Amaan Izhar<sup>1</sup></a>, 
    <a href="https://scholar.google.com.my/citations?user=IgUMlGcAAAAJ&hl=en">Norisma Idris<sup>1</sup></a>, 
    <a href="https://scholar.google.com/citations?user=TyH59tkAAAAJ&hl=en">Nurul Japar<sup>1*</sup></a>
  </strong>
  <br/>
  <strong><sup>1</sup>Faculty of Computer Science and Information Technology, Universiti Malaya, Kuala Lumpur, 50603, Malaysia</strong>
  <br/>
  <em>*Corresponding Author</em>
  <br/><br/>
  <a href="https://link.springer.com/article/10.1007/s12559-025-10404-6">
    <img src="https://img.shields.io/badge/Read%20Paper-Springer-brightgreen?style=for-the-badge" alt="Read Paper Badge">
  </a>
  <br/><br/>
</p>

___

## :bulb: Abstract
<p align="justify">Large language models (LLMs) remain relatively underutilized in medical imaging, particularly in radiology, which is essential for disease diagnosis and management. Nonetheless, radiology report generation (RRG) is a time-consuming task that can result in delays and inconsistencies. To address these challenges, we present a novel hybrid approach that integrates multi-modal radiology information and preference optimization alignment in LLM for continual RRG. Our method integrates a pre-trained small multi-modal model to analyze radiology images and generate an initial report, which is subsequently refined and aligned by an LLM using odds ratio preference optimization (ORPO) and with historical patient data and assessments to mimic radiologist-like responses, bypassing reinforcement learning from human feedback-based (RLHF) alignment. This two-stage fusion—supervised fine-tuning followed by preference optimization—ensures high accuracy while minimizing hallucinations and errors. We also propose a data field curation strategy extendable to various other RRG modality datasets, focusing on selecting relevant responses for preference alignment. We evaluate our approach on two public datasets, achieving state-of-the-art performance with average Bleu scores of 0.375 and 0.647, Meteor scores of 0.495 and 0.714, Rouge-L scores of 0.483 and 0.732, and average F1-RadGraph scores of 0.488 and 0.487, for chest X-rays and lung CT scan datasets, respectively. We further provide in-depth qualitative analyses and ablation studies to explain the workings of our model and grasp the clinical relevance for RRG. This work presents the first application of preference optimization in continual RRG, representing a significant advancement in automating clinically reliable report generation. By reducing cognitive burdens on radiologists through AI-powered reasoning and alignment in LLMs, the proposed model improves decision-making, perception, and diagnostic precision, streamlining workflows and enhancing patient care.</p>

![Architecture](assets/architecture.png)

## :black_nib: Citation
If you find our work and/or this repository useful in your research, please consider giving a star ⭐ and a citation.

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

## :hammer_and_wrench: Reproducibility
### General requirements:
- `Workstation`:
  - `OS: Ubuntu22.04`
  - `Single node with single GPU`:
    - `RAM >= 16GB`
    - `GPU vRAM >= 24GB`
    - `Storage >= 200GB`
- `System modules:`
  - `miniconda`
  - `cuda >= 12.1`
- `Huggingface access token`

### Setup environment and dependencies:
- Use the following commands in the terminal:
````bash
# Clone the repository
git clone https://github.com/AI-14/r2gpoallm.git

# Get in the repository
cd r2gpoallm

# Setup conda environment and install dependencies
conda create -n env python=3.10 --yes
source activate env
conda install pip
pip install -r requirements.txt

# Update this specific package
pip install -U datasets
````

### Setup embryonic experimental results:
- For each dataset i.e. IUXRAY & COVCTR, follow:
  - Follow steps given in [PKATransNet](https://github.com/AI-14/pkatransnet) and get the results after running their experiments. In `fg-res` directory, rename `beam-search-predictions.csv` file to `test_p.csv` file and move this file to corresponding sub-directory in `datasets` directory.
- Move the `datasets` directory from PKATransNet into this repository.

### Experiments with IUXRAY dataset:
- Paste your huggingface access token in front of `--hf_token` in `scripts/iuxray/base_sft_like.sh`, `scripts/iuxray/base_po_like.sh`, `scripts/iuxray/sft.sh`, and `scripts/iuxray/po.sh` files.
- Use the following commands in the terminal:
```bash
# Preprocess
source scripts/iuxray/pp.sh

# Test base sft-like
source scripts/iuxray/base_sft_like.sh

# Test base po-like
source scripts/iuxray/base_po_like.sh

# Train and test sft
source scripts/iuxray/sft.sh

# Train and test po
source scripts/iuxray/po.sh
```

### Experiments with COVCTR dataset:
- Paste your huggingface access token in front of `--hf_token` in `scripts/covctr/base_sft_like.sh`, `scripts/covctr/base_po_like.sh`, `scripts/covctr/sft.sh`, and `scripts/covctr/po.sh` files.
- Use the following commands in the terminal:
```bash
# Preprocess
source scripts/covctr/pp.sh

# Test base sft-like
source scripts/covctr/base_sft_like.sh

# Test base po-like
source scripts/covctr/base_po_like.sh

# Train and test sft
source scripts/covctr/sft.sh

# Train and test po
source scripts/covctr/po.sh
```

### Clean up:
- Use the following commands in the terminal:
```bash
# Get to parent level directory 
cd ..

# Deactivate and delete the environment 
conda deactivate
conda remove --name env --all

# Remove the repository
rm -r r2gpoallm
```

> NOTE: The variance in results for these datasets is high due to their small size and sensitivity to random seed initialization. As a result, exact reproducibility of outcomes may not be guaranteed, with slight fluctuations expected.
