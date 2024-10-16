# Machine Learning-based Digitalization of Engineering Drawings

This repository presents a hybrid method as a complete solution for digitizing engineering drawings.

The architecture combines the You Only Look Once (**YOLO**), Probabilistic Hough Transform (**PHT**), Density-Based Spatial Clustering of Applications with Noise (**DBSCAN**), and ruled-based methods. The complete pipeline of the proposed method is as follows:

![RFI2](https://github.com/user-attachments/assets/0f2f44c1-3075-4fee-974e-97aedeb7e8f8)

---

The first step in using deep learning-based models for object detection is to reduce the dimensions of the drawings since the original projects have high resolution. Thus, cropouts of the drawings are created and a dataset with several smaller images is obtained. This process is carried out using the sliding window method, the algorithm for which is presented [here](https://github.com/SFStefenon/Digital_ED/blob/main/Sliding%20Window/Sliding%20Window%20Compute.py).

---

YOLO is used for object detection considering a custom dataset (symbols, labels, and specifiers) from relay-based railway interlocking systems. The explanation of how YOLO is employed is presented [here](https://github.com/SFStefenon/Digital_ED/tree/main/YOLO). To ensure that the best architecture is considered, the model selection and hyperparameters tuning are based on the Optuna using a tree-structured Parzen estimator (an example is available [here](https://github.com/SFStefenon/Digital_ED/blob/main/Optuna/yolov8-optuna-sd2.py)). 

---

For segment detection PHT is used, considering that the detected segments are dashed, DBSCAN is used for clustering the segments to create a continuous line.
For PHT the preprocessing techniques canny edge detection, Sobel edge detection, binarization threshold, Otsu Riddler-Calvard threshold, or adaptive threshold can be used.
To compare the DBSCAN clustering, the KMeans, agglomerative clustering, and ordering points to identify the clustering structure (OPTICS) can be considered.

Considering the detected lines, orthogonal lines are merged and identifiers (ID) are used to label them. Using the IDs a graph is built to connect lines to symbols having a graph of the electrical connections. Based on the coordinates from YOLO and the electrical connections from the graph an output that can be read by [NORMA Tool](https://doi.org/10.1007/978-3-030-99524-9_7) is created. The algorithm that identifies the lines, aggregates the lines based on rules to create the graph and generates the readable output is [here](https://github.com/SFStefenon/Digital_ED/blob/main/PHT-DBSCAN/Line_Detection_Graph_Readable_Ouput.py). An example of the results of the proposed method is presented below:

![a](https://github.com/user-attachments/assets/ef58c561-96ca-44be-8b06-791539d81da4)

To perform a complete digitization of engineering drawings, white rectangles are initially drawn over the objects under consideration (symbols, labels, and specifiers), resulting in an image that only has the lines to be detected (without components). PHT is then applied to detect segments, and since this method detects small segments (dashed lines), DBSCAN is used to redraw the lines. Identifiers are assigned to each line and object, and based on geometric and design rules, a Graph is constructed as links from symbols to lines, and then symbols to symbols. The Graph of symbols is used to create a readable output for Norma, the graphical interface used to manage rail network projects.

---

Since the cropouts have 640 by 640 pixels, it is necessary to join the segments and define the position of the objects-based cropouts from the original image, this is done by this [algorithm](https://github.com/SFStefenon/Digital_ED/blob/main/Graph/Load_Complete_Graph_Full_Image_Annotations_and_Segments.py). The complete graph that combines all electrical connections is available [here](https://github.com/SFStefenon/Digital_ED/blob/main/Graph/Create_Complete_Graph_Full_Image_to_Save_Annotations_and_Segments.py).  


---

Wrote by Dr. **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, October 10, 2024
