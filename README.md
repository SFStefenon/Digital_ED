# Machine Learning-based Digitalization of Engineering Drawings

This repository presents a hybrid method as a complete solution for digitizing engineering drawings.

The architecture combines the You Only Look Once (**YOLO**), Probabilistic Hough Transform (**PHT**), Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**), and ruled-based methods. The complete pipeline of the proposed method is as follows:

![RFI2](https://github.com/user-attachments/assets/0f2f44c1-3075-4fee-974e-97aedeb7e8f8)

---

YOLO is used for object detection considering a custom dataset (symbols, labels, and specifiers) from relay-based railway interlocking systems. The explanation of how YOLO is employed is presented [here](https://github.com/SFStefenon/Digital_ED/tree/main/YOLO). To ensure that the best architecture setup is considered, hypertuning is used for model selection (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x) and hyperparameters tuning, as it explained here.

, the PHT with DBSCAN is considered for segment detection, and the ruled-based methods apply considering the schematic rules of the drawings.


---

Wrote by **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, June 06, 2024
