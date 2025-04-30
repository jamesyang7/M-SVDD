
# Audio-Inertia Fusion for Anomaly Detection in Autonomous Mobile Robots Based on Mahalanobis SVDD  

This repository contains the implementation of our proposed unsupervised anomaly detection network that fuses **audio** and **IMU data** to detect anomalies, including **collisions** and **component failures**.  

Our approach leverages a *Mahalanobis distance-based Deep Support Vector Data Description (M-SVDD)** model trained solely on **normal operational data**, enabling cost-effective and robust anomaly detection **without the need for annotated fault datasets**.  

---

## The structure of the proposed network 
<div style="display: flex; justify-content: center; align-items: center; gap: 200px;">
    <img src="image/teaser.png" alt="The structure of the proposed network" width="400">
    <img src="image/data_collection.jpg" alt="Data collection platform" width="400">
</div>


---
## Preparation

### Environment Setup
To ensure compatibility and smooth execution, the following libraries and frameworks are required:
- **PyTorch**
- **TorchAudio**
- **DeepOD**


---
## Train and Validatation

### Set the Configuration  
Edit the configuration file located at: `config/config.json`  

### Train the Network  
Run the following command to train the model:  
```bash
python GSVDD_train.py
```  

### Evaluate the Network  
After training, evaluate the model's performance using:`GSVDD_test.ipynb`  

---

### Dataset
- The dataset used in this repository can be downloaded from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/yizhuo001_e_ntu_edu_sg/EbQAP08fM_5LvZqfEVBX7BUBNH7RfH1T1OE26DDRPsigow?e=gmIETT).  
- The ROS bag containing all modality data will be uploaded soon.

---
## Note
- The evaluation of the model on public datasets are presented in timeseries branch or [this link](https://anonymous.4open.science/r/GSVDD-853B).
___

## References  
- **TranAD**: [TranAD Repository](https://github.com/imperial-qore/TranAD)  
- **DeepOD**: [DeepOD Repository](https://github.com/xuhongzuo/DeepOD)  
---
