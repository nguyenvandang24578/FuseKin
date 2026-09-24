import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_spatial_graph(num_node, self_link, inward, outward, virtual_num=0):
    I = np.pad(edge2mat(self_link, num_node), ((0, virtual_num), (0, virtual_num)))
    In = np.pad(edge2mat(inward, num_node), ((0, virtual_num), (0, virtual_num)))
    Out = np.pad(edge2mat(outward, num_node), ((0, virtual_num), (0, virtual_num)))
    
    # Virtual node is fully connected to all physical nodes in In and Out
    # and has a self-loop in I
    for i in range(virtual_num):
        I[num_node + i, num_node + i] = 1
        In[:num_node, num_node + i] = 1
        Out[num_node + i, :num_node] = 1
        
    A = np.stack((I, In, Out))
    return A

def get_coco19_adjacency(virtual_num=1):
    num_node = 19
    self_link = [(i, i) for i in range(num_node)]
    
    # 0: Nose, 1: L_Eye, 2: R_Eye, 3: L_Ear, 4: R_Ear
    # 5: L_Shoulder, 6: R_Shoulder, 7: L_Elbow, 8: R_Elbow, 9: L_Wrist, 10: R_Wrist
    # 11: L_Hip, 12: R_Hip, 13: L_Knee, 14: R_Knee, 15: L_Ankle, 16: R_Ankle
    # 17: Pelvis, 18: Neck

    # Inward edges (hướng về phía Pelvis - root)
    inward = [
        (0, 18), # Nose -> Neck
        (1, 0),  # L_Eye -> Nose
        (2, 0),  # R_Eye -> Nose
        (3, 1),  # L_Ear -> L_Eye
        (4, 2),  # R_Ear -> R_Eye
        
        (18, 17), # Neck -> Pelvis
        
        (5, 18),  # L_Shoulder -> Neck
        (6, 18),  # R_Shoulder -> Neck
        (7, 5),   # L_Elbow -> L_Shoulder
        (9, 7),   # L_Wrist -> L_Elbow
        (8, 6),   # R_Elbow -> R_Shoulder
        (10, 8),  # R_Wrist -> R_Elbow
        
        (11, 17), # L_Hip -> Pelvis
        (12, 17), # R_Hip -> Pelvis
        (13, 11), # L_Knee -> L_Hip
        (15, 13), # L_Ankle -> L_Knee
        (14, 12), # R_Knee -> R_Hip
        (16, 14)  # R_Ankle -> R_Knee
    ]
    
    outward = [(j, i) for (i, j) in inward]
    
    A = get_spatial_graph(num_node, self_link, inward, outward, virtual_num=virtual_num)
    return A
