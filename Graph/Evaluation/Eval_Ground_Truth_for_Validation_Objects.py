
''' 
Algoritm wrote by Stefano Frizzo Stefenon

Fondazione Bruno Kessler
Trento, May 15, 2023.

'''

# Import libs
import glob
import numpy as np

# Definition
# tp = true posotive (the correct edges identified)
# fn = false negative (what is missing from the ground_truth)
# fp = false positive (what is identified and shouldn't be)

###############################################################################
###############################################################################
# Load all graphs
###############################################################################
###############################################################################
#path: str = '../PrePro/Dataset_Graphs/Ground_truth/Letters/*.csv'
path: str = '../PrePro/Dataset_Graphs/Ground_truth/Objects/*.csv'
ground_truth = glob.glob(path, recursive=True)

#path: str = '../PrePro/Dataset_Graphs/Graph_segments/Letters/110d2/15_15/*.csv'
#path: str = '../PrePro/Dataset_Graphs/Graph_segments/Letters/IoU/5_0/*.csv'
path: str = '../PrePro/Dataset_Graphs/Graph_segments/Objects/01_01/*.csv'
graph_names = glob.glob(path, recursive=True)

# Definition of Train and Test Slip
slip = 1 # 70% of the data to train (define the settings of the parameters)

# Train
#ground_truth = ground_truth[:int(slip*len(ground_truth))]

# Test
#ground_truth = ground_truth[int(slip*len(ground_truth)):]

all_recall=[]
all_precision=[]
all_tp=[]
all_fp=[]
all_fn=[]
all_edges=[]

for graph_gt in ground_truth:
    mydata_gt = np.loadtxt(graph_gt, delimiter=",", dtype=np.float32, skiprows=0)

    # Rule to update the data
    save=[]

    # Calculate the measures of performance
    # mydata = mydata # Updated Data
    # mydata = mydata # Original data
    for graph in graph_names:
        mydata = np.loadtxt(graph, delimiter=",", dtype=np.float32, skiprows=1, usecols=(1, 2, 3, 4))        
        if graph_gt[46:] == graph[57:]:
            tp=0
            fp=0
                       
            for i in range(0,len(mydata)):
                not_found=0     
                
                # If there is only one edge in the gt and in mydata
                if len(mydata_gt.flatten()) == 2 and len(mydata.flatten()) == 4: 
                    if ((mydata_gt[0] == mydata[0]) and (mydata_gt[1] == mydata[1]))  or ((mydata_gt[1] == mydata[0]) and (mydata_gt[0] == mydata[1])) and i==0: # (do it once)
                        tp+=1
                    elif (i==0):
                        fp+=1
                        print('There is a strange in', graph[57:], 'which is', mydata)  

                # If there is only one edge in mydata
                elif len(mydata_gt.flatten()) > 2 and len(mydata.flatten()) == 4:  
                    for check in mydata_gt:   
                        if ((check[0] == mydata[0]) and (check[1] == mydata[1]))  or ((check[1] == mydata[0]) and (check[0] == mydata[1])) and i==0: # (do it once)
                            tp+=1   
                        elif (i==0):
                            not_found+=1                          
                    if not_found == len(mydata_gt):
                        fp+=1
                        print('There is a strange in', graph[57:], 'which is', mydata)

                # If there is only one edge in gt
                elif len(mydata_gt.flatten()) == 2 and len(mydata.flatten()) > 4:  
                    if ((mydata_gt[0] == mydata[i][0]) and (mydata_gt[1] == mydata[i][1]))  or ((mydata_gt[1] == mydata[i][0]) and (mydata_gt[0] == mydata[i][1])) and i==0: # (do it once)
                        tp+=1
                    elif (i==0):
                        fp+=1
                        print('There is a strange in', graph[57:], 'which is', mydata)  

                # For more edges the position is line by line    
                else: 
                    for check in mydata_gt:
                        if ((check[0] == mydata[i][0]) and (check[1] == mydata[i][1]))  or ((check[1] == mydata[i][0]) and (check[0] == mydata[i][1])):
                            tp+=1          
                        else:
                            not_found+=1
                    if not_found == len(mydata_gt):
                        fp+=1
                        print('There is a strange in', graph[57:], 'which is', mydata[i])
            
            if len(mydata_gt.flatten()) == 2:
                all_edges.append(1)
                fn = abs(1 - tp)
            else:
                all_edges.append(len(mydata_gt))
                fn = abs(len(mydata_gt) - tp)
                
            # Compute           
            recall = tp/(tp+fn)
            if tp > 0 or fp > 0:
                precision = tp/(tp+fp)
            print(f'{graph_gt[46:-4]}: Prec. = {precision:.3f}, Rec. = {recall:.3f}')
            all_recall.append(recall)
            all_precision.append(precision)
            all_tp.append(tp)
            all_fp.append(fp)
            all_fn.append(fn)
            
m_recall = np.mean(all_recall)
m_precision = np.mean(all_precision) 
m_f1_score = m_precision*m_recall*2 /(m_precision +m_recall)

print('\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') 
print('Edges = all edges of the ground truth')
print('Correct = right identified edges from the ground truth')
print('Strange = identified edges that are not in the ground truth')
print('Missing = edges that are not found in the ground truth')
print(f'From {len(ground_truth)} drawings I have:')
print(f'Edges = {sum(all_edges)}, Correct (TP) = {sum(all_tp)}, Strange (FP) = {sum(all_fp)}, Missing (FN) = {sum(all_fn)}')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')   

print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
print(f'Precision = {m_precision:.5f}, Recall = {m_recall:.5f}, F1_score = {m_f1_score:.5f}')
print(f'& {m_precision:.5f} & {m_recall:.5f} & {m_f1_score:.5f}')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')    























'''
save=[]
for graph in graph_names:
    mydata = np.loadtxt(graph, delimiter=",", dtype=np.float32, skiprows=1, usecols=(1, 2, 3, 4))
    # mydata2 = mydata
    for i in range(0,len(mydata)):
        count: int = 0
        for k in range(0,len(mydata)):
            if (mydata[k][0] == mydata[i][0]) and i != k:
                count +=1
                
                if count>0 and (mydata[k][3] == mydata[i][3]):
                    
                    if mydata[k][2]>mydata[i][2]:
                        
                        print(mydata[k], k)
                        save.append(k)
mydata2 = np.delete(mydata, save, 0)
'''

'''
all_recall=[]
all_precision=[]
all_f1_score=[]

for graph in graph_names:
    mydata = np.loadtxt(graph, delimiter=",", dtype=np.float32, skiprows=1, usecols=(1, 2))
    for graph_gt in ground_truth:
        mydata_gt = np.loadtxt(graph_gt, delimiter=",", dtype=np.float32, skiprows=0)
        if graph[51:] == graph_gt[43:]:
            # print(graph_gt[43:])
            tp=0
            for i in range(0,len(mydata)):
                for check in mydata_gt:                 
                    if (check[0] == mydata[i][0]) and (check[1] == mydata[i][1])  or (check[1] == mydata[i][0]) and (check[0] == mydata[i][1]):
                        tp+=1
            fn = abs(len(mydata_gt) - tp)
            fp = abs(len(mydata) - tp)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            f1_score = precision*recall*2 / (precision+recall)
            # print(f'Precision = {precision}, Recall = {recall}, f1_score = {f1_score}')
            all_recall.append(recall)
            all_precision.append(precision)
            all_f1_score.append(f1_score)
            
m_recall = np.mean(all_recall)
m_precision = np.mean(all_precision) 
m_f1_score =   np.mean(all_f1_score)     

    
print(f'Mean Precision = {m_precision:.5f}, Mean Recall = {m_recall:.5f}, Mean F1_score = {m_f1_score:.5f}')
print(f' {m_precision:.5f} & {m_recall:.5f} & {m_f1_score:.5f}')
'''





