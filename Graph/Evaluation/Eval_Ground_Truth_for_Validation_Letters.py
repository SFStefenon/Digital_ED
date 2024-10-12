
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
path: str = '../PrePro/Dataset_Graphs/Ground_truth/Letters/*.csv'
ground_truth = glob.glob(path, recursive=True)

path: str = '../PrePro/Dataset_Graphs/Graph_segments/Letters/110d2/15_15/*.csv'
#path: str = '../PrePro/Dataset_Graphs/Graph_segments/Letters/IoU/5_0/*.csv'

graph_names = glob.glob(path, recursive=True)

# Definition of Train and Test Slip
slip = 0.7 # 70% of the data to train (define the settings of the parameters)


# Train
#ground_truth = ground_truth[:int(slip*len(ground_truth))]

# Test
#ground_truth = ground_truth[int(slip*len(ground_truth)):]


all_recall=[]
all_precision=[]

for graph in graph_names:
    # Rule to update the data
    save=[]
    mydata = np.loadtxt(graph, delimiter=",", dtype=np.float32, skiprows=1, usecols=(1, 2, 3, 4))
    for i in range(0,len(mydata)):
        count: int = 0
        for k in range(0,len(mydata)):
            # IF there is repeated ID and it is not itself count it
            if (mydata[k][0] == mydata[i][0]) and i != k:
                count +=1
                # Where there is the repeated ID
                if count>0 and (mydata[k][3] == mydata[i][3]):
                    # For the repeated ID that is further apart
                    if mydata[k][2]>mydata[i][2]:
                        # Save its position
                        save.append(k)
    # Delete all the repeated ID that are further apart
    mydata2 = np.delete(mydata, save, 0)

    # Calculate the measures of performance
    mydata = mydata # Updated Data
    # mydata = mydata # Original data
    for graph_gt in ground_truth:
        mydata_gt = np.loadtxt(graph_gt, delimiter=",", dtype=np.float32, skiprows=0)
        if graph[-19:] == graph_gt[-19:]:
            tp=0
            for i in range(0,len(mydata)):
                for check in mydata_gt:                 
                    if (check[0] == mydata[i][0]) and (check[1] == mydata[i][1])  or (check[1] == mydata[i][0]) and (check[0] == mydata[i][1]):
                        tp+=1
            fn = abs(len(mydata_gt) - tp)
            fp = abs(len(mydata) - tp)
            recall = tp/(tp+fn)
            precision = tp/(tp+fp)
            print(f'{graph_gt[-19:]}: Prec. = {precision:.3f}, Rec. = {recall:.3f}')
            all_recall.append(recall)
            all_precision.append(precision)
            
m_recall = np.mean(all_recall)
m_precision = np.mean(all_precision) 
m_f1_score = m_precision*m_recall*2 / (m_precision +m_recall)

print('\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print(f'Mean Precision = {m_precision:.5f}, Mean Recall = {m_recall:.5f}, Mean F1_score = {m_f1_score:.5f}')
print(f'& {m_precision:.5f} & {m_recall:.5f} & {m_f1_score:.5f}')
print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')    





























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





