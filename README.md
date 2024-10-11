# Machine Learning-based Digitalization of Engineering Drawings

This repository presents a hybrid method as a complete solution for digitizing engineering drawings.

The architecture combines the You Only Look Once (**YOLO**), Probabilistic Hough Transform (**PHT**), Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**), and ruled-based methods. The complete pipeline of the proposed method is as follows:

![RFI2](https://github.com/user-attachments/assets/0f2f44c1-3075-4fee-974e-97aedeb7e8f8)

---

The first step in using the deep learning-based model for object detection is to reduce the dimensions of the drawings since the original projects have high resolution. This process is carried out using the sliding window method, the algorithm for which is presented [here](https://github.com/SFStefenon/Digital_ED/blob/main/Sliding%20Window/Sliding%20Window%20Compute.py).

---

YOLO is used for object detection considering a custom dataset (symbols, labels, and specifiers) from relay-based railway interlocking systems. The explanation of how YOLO is employed is presented [here](https://github.com/SFStefenon/Digital_ED/tree/main/YOLO). To ensure that the best architecture setup is considered, hypertuning is used for model selection (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x) and hyperparameters tuning.

To perform the hyperparameters tuning of YOLO the Optuna using a tree-structured Parzen estimator is used.


> If you want to use a local machine, you can follow this Python-based [algorithm](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/yolov8_insulator_exp1.py). Using a Cluster the study is gonna be saved and you can evaluate latter using [Colab](https://github.com/SFStefenon/WBF-HE-YOLO/blob/main/Hypertuning_Optuna/Cluster_Computing/Experiment_Results/Optuna_Results.ipynb).

OBS: Since the analysis is using a deep learning-based model, depending on your dataset, a high processing time will be required for the model to be trained (considering the defined number of epochs). The file that calls the dataset must be in the same main folder of the model.

---

, the PHT with DBSCAN is considered for segment detection, and the ruled-based methods apply considering the schematic rules of the drawings.


---

Wrote by **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, June 06, 2024
