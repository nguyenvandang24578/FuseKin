import os, sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
def build_human_H_matrix():
    # Khởi tạo ma trận H: 24 khớp x 5 siêu cạnh (toàn số 0)
    H = np.zeros((24, 5))
    
    # Giả sử index các khớp trong chuẩn SMPL (Bạn tự chỉnh lại theo đúng index của bạn nhé):
    # 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1...
    
    # Siêu cạnh 0: Thân mình (Pelvis, Spine, Neck, Head...)
    torso_joints = [0, 3, 6, 9, 12, 15] 
    H[torso_joints, 0] = 1.0
    
    # Siêu cạnh 1: Chân trái (L_Hip, L_Knee, L_Ankle, L_Foot)
    left_leg_joints = [1, 4, 7, 10]
    H[left_leg_joints, 1] = 1.0
    
    # Siêu cạnh 2: Chân phải
    right_leg_joints = [2, 5, 8, 11]
    H[right_leg_joints, 2] = 1.0
    
    # Siêu cạnh 3: Tay trái (L_Collar, L_Shldr, L_Elbow, L_Wrist, L_Hand)
    left_arm_joints = [13, 16, 18, 20, 22]
    H[left_arm_joints, 3] = 1.0
    
    # Siêu cạnh 4: Tay phải
    right_arm_joints = [14, 17, 19, 21, 23]
    H[right_arm_joints, 4] = 1.0
    
    # Lưu ý: Các siêu cạnh CÓ THỂ GIAO NHAU (Ví dụ Pelvis vừa thuộc Thân, vừa thuộc Chân) 
    # Bạn cứ thoải mái thêm các joint giao cắt vào mảng nhé.
    
    return H
# . Hàm vẽ
def visualize_G():
    # Khởi tạo G
    H_matrix = build_human_H_matrix()
    G_matrix = _generate_G_from_H(H_matrix)
    G_array = np.asarray(G_matrix)

    # Tên 24 khớp SMPL chuẩn để gắp vào trục tung/hoành
    smpl_joints = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 
        'R_Collar', 'Head', 'L_Shldr', 'R_Shldr', 'L_Elbow', 'R_Elbow', 
        'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    # Setup hình vẽ
    plt.figure(figsize=(12, 10))
    sns.heatmap(G_array, 
                cmap='YlGnBu', 
                square=True, 
                xticklabels=smpl_joints, 
                yticklabels=smpl_joints,
                linewidths=0.5,     # Kẻ vạch cho dễ nhìn
                annot=False)        # Bỏ số đi cho đỡ rối, nhìn màu là đủ
    
    plt.title("Static Hypergraph Matrix G (24x24)", fontsize=16)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("Check_Static_G.png", dpi=300)
    print("Đã lưu ảnh Check_Static_G.png thành công!")

if __name__ == '__main__':
    visualize_G()