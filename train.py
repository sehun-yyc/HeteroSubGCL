import torch
from torch_geometric.data import Data, HeteroData
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import numpy as np
import scipy.sparse as sp
from model.heco import HeCo
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, auc, precision_recall_curve
from sklearn.neural_network import MLPClassifier
from model.contrast import Contrast
from sklearn import svm
import scipy.io as sio
from sklearn import preprocessing
import torch.backends.cudnn as cudnn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import produce_adjacent_matrix
import edge_dict
import time
from sklearn.feature_selection import SelectFromModel, SelectPercentile, RFE, f_classif, VarianceThreshold
import time
# print(produce_adjacent_matrix.a)
# 建议写成类、函数等模块化程序
time0 = time.time()
# 固定种子
seed = 54321
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
cudnn.benchmark = False
cudnn.deterministic = True

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')

# # 加载数据
data = sio.loadmat('BRCA.mat')  # 根据实际数据文件名调整
features = data['BRCA_Gene_Expression'].T
Methy_features = data['BRCA_Methy_Expression'].T
Mirna_features = data['BRCA_Mirna_Expression'].T
labels = data['BRCA_clinicalMatrix']
indexes = data['BRCA_indexes']

# data = sio.loadmat('GBM.mat')  # 根据实际数据文件名调整
# features = data['GBM_Gene_Expression'].T
# Methy_features = data['GBM_Methy_Expression'].T
# Mirna_features = data['GBM_Mirna_Expression'].T
# labels = data['GBM_clinicalMatrix']
# indexes = data['GBM_indexes']
# 数据预处理
features = preprocessing.scale(features)
Methy_features = preprocessing.scale(Methy_features)
Mirna_features = preprocessing.scale(Mirna_features)

labels = labels.reshape(labels.shape[0])

path = "data/"
cites1 = path + "edges_gene_brca.csv"
cites2 = path + "edges_methy_brca.csv"
cites3 = path + "edges_mirna_brca.csv"

# cites1 = path + "edges_gene_gbm.csv"
# cites2 = path + "edges_methy_gbm.csv"
# cites3 = path + "edges_mirna_gbm.csv"

# 索引字典，转换到从0开始编码
index_gene_dict = dict()
edge_gene_index = []

for i in range(indexes.shape[0]):
    index_gene_dict[int(indexes[i])] = len(index_gene_dict)

with open(cites1, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.strip().split(',')
        edge_gene_index.append([index_gene_dict[int(start)], index_gene_dict[int(end)]])

index_methy_dict = dict()
edge_methy_index = []

for i in range(indexes.shape[0]):
    index_methy_dict[int(indexes[i])] = len(index_methy_dict)

with open(cites2, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.strip().split(',')
        edge_methy_index.append([index_methy_dict[int(start)], index_methy_dict[int(end)]])

index_mirna_dict = dict()
edge_mirna_index = []

for i in range(indexes.shape[0]):
    index_mirna_dict[int(indexes[i])] = len(index_mirna_dict)

with open(cites3, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.strip().split(',')
        edge_mirna_index.append([index_mirna_dict[int(start)], index_mirna_dict[int(end)]])

# 将边索引转换为张量并移动到设备
edge_gene_index = torch.LongTensor(edge_gene_index).t().to(device)
edge_methy_index = torch.LongTensor(edge_methy_index).t().to(device)
edge_mirna_index = torch.LongTensor(edge_mirna_index).t().to(device)

# 训练前的准备：将特征和标签转换为张量并移动到设备
labels = torch.LongTensor(labels).to(device)  # 确保 labels 在 GPU
features1 = torch.FloatTensor(features).to(device)
features2 = torch.FloatTensor(Methy_features).to(device)
features3 = torch.FloatTensor(Mirna_features).to(device)

# 创建mask并移动到 GPU
mask = torch.randperm(len(index_gene_dict)).to(device)

# 填充 edge_dict
edge_dict.edge_dict = {
    'gene_rel': edge_gene_index,
    'methy_rel': edge_methy_index,
    'mirna_rel': edge_mirna_index,
}

# 定义一个通用的特征选择函数
def select_features(X, y, method='model_based', threshold='median', percentile=40, step=0.1):
    """
    通用特征选择函数，支持多种方法。
    
    参数:
    - X: 特征矩阵 (numpy array)
    - y: 标签向量 (numpy array)
    - method: 特征选择方法 ('model_based', 'percentile', 'rfe')
    - threshold: 用于选择特征的阈值，依据不同方法可能有所不同
    - percentile: 在 'percentile' 方法中使用，表示选择的特征百分比
    - step: 在 'rfe' 方法中使用，表示每次移除特征的数量或比例
    
    返回:
    - X_selected: 选择后的特征矩阵 (numpy array)
    """
    # 移除常数特征
    selector_var = VarianceThreshold(threshold=0.0)
    X_var = selector_var.fit_transform(X)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)
    
    if method == 'model_based':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_scaled, y)
        selector = SelectFromModel(clf, threshold=threshold, prefit=True)
        X_selected = selector.transform(X_scaled)
    elif method == 'percentile':
        selector = SelectPercentile(score_func=f_classif, percentile=percentile)
        X_selected = selector.fit_transform(X_scaled, y)
    elif method == 'rfe':
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rfe = RFE(estimator=clf, n_features_to_select=1, step=step)  # n_features_to_select=1 表示选择所有特征，直到剩下一个
        rfe.fit(X_scaled, y)
        X_selected = rfe.transform(X_scaled)
    else:
        raise ValueError("Unsupported feature selection method.")
    
    return X_selected

# 全局特征选择
features1_selected = select_features(
    features1.cpu().numpy(), 
    labels.cpu().numpy(), 
    method='percentile', 
    percentile=40
)
print(f'Features1 selected shape: {features1_selected.shape}')

features2_selected = select_features(
    features2.cpu().numpy(), 
    labels.cpu().numpy(), 
    method='percentile', 
    percentile=40
)
print(f'Features2 selected shape: {features2_selected.shape}')

features3_selected = select_features(
    features3.cpu().numpy(), 
    labels.cpu().numpy(), 
    method='percentile', 
    percentile=40
)
print(f'Features3 selected shape: {features3_selected.shape}')

# 转换回张量并移动到 GPU
features1_selected = torch.FloatTensor(features1_selected).to(device)
features2_selected = torch.FloatTensor(features2_selected).to(device)
features3_selected = torch.FloatTensor(features3_selected).to(device)

# 构建 HeteroData 对象
coarse_view = HeteroData()
coarse_features_selected = torch.cat([features1_selected, features2_selected, features3_selected], dim=1)
coarse_view['patient'].x = coarse_features_selected
coarse_view['patient', 'gene_rel', 'patient'].edge_index = edge_gene_index
coarse_view['patient', 'methy_rel', 'patient'].edge_index = edge_methy_index
coarse_view['patient', 'mirna_rel', 'patient'].edge_index = edge_mirna_index
coarse_view['patient'].y = labels

# 构建其他视图的 HeteroData 对象（根据全局选择后的特征）
data_gene_methy_selected = HeteroData()
data_gene_methy_selected['patient'].x = torch.cat([features1_selected, features2_selected], dim=1)
data_gene_methy_selected['patient', 'gene_rel', 'patient'].edge_index = edge_gene_index
data_gene_methy_selected['patient', 'methy_rel', 'patient'].edge_index = edge_methy_index
data_gene_methy_selected['patient'].y = labels

data_gene_mirna_selected = HeteroData()
data_gene_mirna_selected['patient'].x = torch.cat([features1_selected, features3_selected], dim=1)
data_gene_mirna_selected['patient', 'gene_rel', 'patient'].edge_index = edge_gene_index
data_gene_mirna_selected['patient', 'mirna_rel', 'patient'].edge_index = edge_mirna_index
data_gene_mirna_selected['patient'].y = labels

data_methy_mirna_selected = HeteroData()
data_methy_mirna_selected['patient'].x = torch.cat([features2_selected, features3_selected], dim=1)
data_methy_mirna_selected['patient', 'methy_rel', 'patient'].edge_index = edge_methy_index
data_methy_mirna_selected['patient', 'mirna_rel', 'patient'].edge_index = edge_mirna_index
data_methy_mirna_selected['patient'].y = labels

data_gene_selected = HeteroData()
data_gene_selected['patient'].x = features1_selected
data_gene_selected['patient', 'gene_rel', 'patient'].edge_index = edge_gene_index
data_gene_selected['patient'].y = labels

data_methy_selected = HeteroData()
data_methy_selected['patient'].x = features2_selected
data_methy_selected['patient', 'methy_rel', 'patient'].edge_index = edge_methy_index
data_methy_selected['patient'].y = labels

data_mirna_selected = HeteroData()
data_mirna_selected['patient'].x = features3_selected
data_mirna_selected['patient', 'mirna_rel', 'patient'].edge_index = edge_mirna_index
data_mirna_selected['patient'].y = labels

# 初始化评估指标
p_mean = np.zeros(10)
r_mean = np.zeros(10)
f1score_mean = np.zeros(10)
ACC_mean = np.zeros(10)
ARS_mean = np.zeros(10)
MCC_mean = np.zeros(10)
AUC_mean = np.zeros(10)
PR_AUC_mean = np.zeros(10)
DBI_mean = np.zeros(10)
SS_mean = np.zeros(10)

k = 5
f_name = './results/'

for n in range(10):
    # 初始化每次外循环的评估指标
    p = np.zeros(k)
    r = np.zeros(k)
    f1score = np.zeros(k)
    ACC = np.zeros(k)
    ARS = np.zeros(k)
    MCC = np.zeros(k)
    AUC = np.zeros(k)
    PR_AUC = np.zeros(k)
    DBI = np.zeros(k)
    SS = np.zeros(k)
    
    m = 0
    kfold = KFold(n_splits=k, shuffle=True, random_state=n*n+1)
    
    for train_idx, test_idx in kfold.split(mask):
        train_mask = mask[train_idx]
        test_mask = mask[test_idx]
        
        # 确保 train_mask 和 test_mask 在 GPU
        train_mask = train_mask.to(device)
        test_mask = test_mask.to(device)
        
        # 打印设备信息（可选，用于调试）
        print(f'labels device: {labels.device}')
        print(f'train_mask device: {train_mask.device}')
        
        # 初始化模型
        model = HeCo(
            num_feature_coarse=coarse_features_selected.shape[1],  # 更新后的特征维度
            num_feature_medium1=data_gene_methy_selected['patient'].x.shape[1],
            num_feature_medium2=data_gene_mirna_selected['patient'].x.shape[1],
            num_feature_medium3=data_methy_mirna_selected['patient'].x.shape[1],
            num_feature_fine1=data_gene_selected['patient'].x.shape[1],
            num_feature_fine2=data_methy_selected['patient'].x.shape[1],
            num_feature_fine3=data_mirna_selected['patient'].x.shape[1]
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
        criterion = Contrast(128, 0.3, 0.7).to(device)
        
        # 训练模型
        for epoch in range(130):
            model.train()
            optimizer.zero_grad()
            embeds = model(
                coarse_view, 
                data_gene_methy_selected, 
                data_gene_mirna_selected, 
                data_methy_mirna_selected, 
                data_gene_selected, 
                data_methy_selected, 
                data_mirna_selected
            )
            pos = (labels[train_mask].unsqueeze(0) == labels[train_mask].unsqueeze(1)).float().to(device)
            loss = criterion(
                embeds['coarse'][train_mask], 
                embeds['medium1'][train_mask], 
                embeds['medium2'][train_mask], 
                embeds['medium3'][train_mask], 
                embeds['fine1'][train_mask], 
                embeds['fine2'][train_mask], 
                embeds['fine3'][train_mask], 
                pos
            )
            if epoch % 10 == 0:  # 每10个epoch打印一次
                print(f'Fold {n+1}, Epoch {epoch}, Loss: {loss.item():.4f}')
            loss.backward()
            optimizer.step()
        
        # 获取嵌入
        model.eval()
        with torch.no_grad():
            embeds = model.get_embeds(
                coarse_view, 
                data_gene_methy_selected, 
                data_gene_mirna_selected, 
                data_methy_mirna_selected, 
                data_gene_selected, 
                data_methy_selected, 
                data_mirna_selected
            )
            weight1, weight2, weight3, weight4 = 0.6, 0.1, 0.1, 0.1
            embeds_train = (
                weight4 * embeds['coarse'][train_mask].detach().cpu().numpy()
                + weight1 * embeds['fine1'][train_mask].detach().cpu().numpy()
                + weight2 * embeds['fine2'][train_mask].detach().cpu().numpy() 
                + weight3 * embeds['fine3'][train_mask].detach().cpu().numpy()
            )
            embeds_test = (
                weight4 * embeds['coarse'][test_mask].detach().cpu().numpy()
                + weight1 * embeds['fine1'][test_mask].detach().cpu().numpy()
                + weight2 * embeds['fine2'][test_mask].detach().cpu().numpy()
                + weight3 * embeds['fine3'][test_mask].detach().cpu().numpy()
            )
            targets_train = labels[train_mask].cpu().numpy()
            targets_test = labels[test_mask].cpu().numpy()
        
        # 使用分类器
        classifier = svm.SVC(
            C=1,
            degree=2,
            kernel='rbf',
            gamma='scale',
            probability=True,  # 启用概率估计
            decision_function_shape='ovr'
        )
        classifier.fit(embeds_train, targets_train)
        Y_test = classifier.predict(embeds_test)
        
        # 计算评估指标
        p[m] = precision_score(targets_test, Y_test, average='macro')
        r[m] = recall_score(targets_test, Y_test, average='macro')
        f1score[m] = f1_score(targets_test, Y_test, average='macro')
        ACC[m] = accuracy_score(targets_test, Y_test)
        ARS[m] = metrics.adjusted_rand_score(targets_test, Y_test)
        MCC[m] = matthews_corrcoef(targets_test, Y_test)
        DBI[m] = metrics.davies_bouldin_score(embeds_test, Y_test)
        SS[m] = metrics.silhouette_score(embeds_test, Y_test)
        
        # 计算AUC和PR-AUC
        n_class = len(np.unique(targets_train))
        y_one_hot = label_binarize(targets_test, classes=np.arange(n_class))
        y_score = classifier.predict_proba(embeds_test)
        
        # 多分类AUC
        fpr, tpr, _ = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        AUC[m] = metrics.auc(fpr, tpr)
        
        # 计算PR AUC
        pr, re, _ = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
        PR_AUC[m] = auc(re, pr)
        
        # 保存每次交叉验证的结果
        with open(f'{f_name}each_best_result_brca_svm.txt', "a") as f:
            print(f'第{n+1}次十倍交叉，折数{m+1}', file=f)
            print(f'Precision : {p[m]:.4f}', file=f)
            print(f'Recall : {r[m]:.4f}', file=f)
            print(f'f1score : {f1score[m]:.4f}', file=f)
            print(f'ACC : {ACC[m]:.4f}', file=f)
            print(f'ARI : {ARS[m]:.4f}', file=f)
            print(f'MCC : {MCC[m]:.4f}', file=f)
            print(f'AUC : {AUC[m]:.4f}', file=f)
            print(f'PR_AUC : {PR_AUC[m]:.4f}', file=f)
            print(f'silhouette_width : {SS[m]:.4f}', file=f)
            print(f'DBI : {DBI[m]:.4f}', file=f)
        m += 1
    
    # 计算并保存每次外循环的平均结果
    p_mean[n] = np.mean(p)
    r_mean[n] = np.mean(r)
    f1score_mean[n] = np.mean(f1score)
    ACC_mean[n] = np.mean(ACC)
    ARS_mean[n] = np.mean(ARS)
    MCC_mean[n] = np.mean(MCC)
    AUC_mean[n] = np.mean(AUC)
    PR_AUC_mean[n] = np.mean(PR_AUC)
    DBI_mean[n] = np.mean(DBI)
    SS_mean[n] = np.mean(SS)
    
    with open(f'{f_name}each_best_result_n_mean_brca_svm.txt', "a") as f:
        print(f'第{n+1}次十倍交叉平均结果', file=f)
        print(f'Precision : {p_mean[n]:.4f}', file=f)
        print(f'Recall : {r_mean[n]:.4f}', file=f)
        print(f'f1score : {f1score_mean[n]:.4f}', file=f)
        print(f'ACC : {ACC_mean[n]:.4f}', file=f)
        print(f'ARI : {ARS_mean[n]:.4f}', file=f)
        print(f'MCC : {MCC_mean[n]:.4f}', file=f)
        print(f'AUC : {AUC_mean[n]:.4f}', file=f)
        print(f'PR_AUC : {PR_AUC_mean[n]:.4f}', file=f)
        print(f'DBI : {DBI_mean[n]:.4f}', file=f)
        print(f'silhouette_width : {SS_mean[n]:.4f}', file=f)
    
# 计算十次外循环的平均结果
s_p_mean = np.mean(p_mean)
s_r_mean = np.mean(r_mean)
s_f1score_mean = np.mean(f1score_mean)
s_ACC_mean = np.mean(ACC_mean)
s_ARS_mean = np.mean(ARS_mean)
s_MCC_mean = np.mean(MCC_mean)
s_AUC_mean = np.mean(AUC_mean)
s_PR_AUC_mean = np.mean(PR_AUC_mean)
s_DBI_mean = np.mean(DBI_mean)
s_SS_mean = np.mean(SS_mean)

with open(f'{f_name}each_best_result_svm_brca_mean.txt', "a") as f:
    time1 = time.time()
    train_time = time1 - time0
    print('训练时间：{:.2f} 秒'.format(train_time), file=f)
    print('十次五倍交叉平均结果', file=f)
    print(f'Precision : {s_p_mean:.4f}', file=f)
    print(f'Recall : {s_r_mean:.4f}', file=f)
    print(f'f1score : {s_f1score_mean:.4f}', file=f)
    print(f'ACC : {s_ACC_mean:.4f}', file=f)
    print(f'ARI : {s_ARS_mean:.4f}', file=f)
    print(f'MCC : {s_MCC_mean:.4f}', file=f)
    print(f'AUC : {s_AUC_mean:.4f}', file=f)
    print(f'PR_AUC : {s_PR_AUC_mean:.4f}', file=f)
    print(f'DBI : {s_DBI_mean:.4f}', file=f)
    print(f'silhouette_width : {s_SS_mean:.4f}', file=f)
