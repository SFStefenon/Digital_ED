The PHT-DBSCAN is a hybrid method that was written from scratch for segment detection. 
The method consists of detecting dashed segments using the PHT. Using the DBSCAN method, neighboring segments are clustered, resulting in a new line that is based on the union of the segments detected by the PHT.
An example of the results of using the method is presented below.

![b](https://github.com/user-attachments/assets/3778928c-32eb-4061-8e42-22ef567fcac0)

Taking into account the segments reconstructed by PHT-DBSCAN and the YOLO bounding boxes, a [graph](https://github.com/SFStefenon/Digital_ED/tree/main/Graph) is constructed based on the distances between segments and specific symbols and rules for the considered engineering drawings.

---

Wrote by Dr. **Stefano Frizzo Stefenon**

Fondazione Bruno Kessler

Trento, Italy, October 10, 2024
