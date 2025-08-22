#  iBitter-GRE

**iBitter-GRE** is a **bitter peptide predictor** that integrates **ESM-2 embeddings** with traditional physicochemical descriptors through a **stacked ensemble framework**.  
This tool enables accurate identification of bitter peptides, assisting research in **food chemistry, bioinformatics, and peptide drug discovery**.  

---

##  Reference

- Lv, J., Geng, A., Pan, Z., Wei, L., Zou, Q., Zhang, Z., & Cui, F. (2025).  
  *iBitter-GRE: A Novel Stacked Bitter Peptide Predictor with ESM-2 and Multi-View Features.*  
  **Journal of Molecular Biology, 437(8), 169005.**  
  [https://doi.org/10.1016/j.jmb.2025.169005](https://doi.org/10.1016/j.jmb.2025.169005)

---

##  Web Server

The **iBitter-GRE web server** is freely available at:  
 [iBitter-GRE](http://121.36.197.223:45107/)

 **Notes:**  
> 1. The web server only accepts protein sequences with length **< 1024**.  
> 2. For any questions, please contact us via email: *your_email@domain.com*  

---

##  Framework

<img width="865" height="844" alt="image" src="https://github.com/user-attachments/assets/c5b298e4-2d96-44c5-bd1e-ebc8d92ed1b0" />


**Figure 1. Overall framework of iBitter-GRE.**  
(A) Dataset construction.  
(B) Feature representation.  
(C) Stacking model framework.  
(D) Web server development.  

---

##  Dataset Statistics

<img width="865" height="309" alt="image" src="https://github.com/user-attachments/assets/1bdefd25-1476-401d-beed-e784176b4bdb" />


**Figure 2. Distribution of training and testing datasets.**  
(A) Amino acid frequency distribution in the training and testing sets.  
(B) Sequence length distribution in the training and testing sets.  

---

##  Performance

###  Comparative Performance with Existing Predictors

**Table 1. Performance comparison of iBitter-GRE with state-of-the-art bitter peptide predictors.**

| Model              | ACC   | AUC   | Sn    | Sp    | MCC   |
|--------------------|-------|-------|-------|-------|-------|
| iBitter-SCM        | 0.844 | 0.904 | 0.844 | 0.844 | 0.688 |
| iBitter-Fuse       | 0.930 | 0.933 | 0.938 | 0.922 | 0.859 |
| Bert4Bitter        | 0.922 | 0.964 | 0.938 | 0.906 | 0.844 |
| iBitter-DRLF       | 0.945 | 0.977 | 0.922 | **0.969** | 0.892 |
| Bitter-RF          | 0.938 | 0.978 | 0.938 | 0.938 | 0.875 |
| CPM-BP (BTP640)    | 0.836 | 0.836 | 0.773 | 0.903 | 0.680 |
| **iBitter-GRE**    | **0.961** | **0.978** | **0.984** | 0.938 | **0.923** |

 **Observation:**  
- iBitter-GRE achieves the **highest accuracy (96.1%), AUC (97.8%), Sn  (98.4%)** and **MCC (0.923)**.  
- It surpasses all existing predictors across multiple metrics, demonstrating strong robustness and reliability.  

---

##  Repository Structure

```bash
iBitter-GRE/
│── Data/                     # Datasets
│   ├── training.fasta
│   ├── testing.fasta
│   ├── training.csv
│   └── testing.csv
│
│── Extract features/          # Feature extraction scripts
│   ├── extract_features.py
│   ├── extract_features_use.py
│   └── index.csv
│
│── Prediction/                # Prediction scripts
│   ├── prediction.py
│   └── use_predictor.py
│
│── config.json                # Python version & dependencies
│── README.md                  # Project documentation
│── .gitignore
