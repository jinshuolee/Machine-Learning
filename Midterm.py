# XGBoost Classification
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

filename = 'train.csv'
filename2 = 'X_test.csv'

names=['D_Dbuild','D_StructureSystem_RC','D_StructureSystem_Precast_Concrete',
       'D_StructureSystem_Steel','D_StructureSystem_Reinfored_Brick','D_StructureSystem_Brick_Wall',
       'D_StructureSystem_Brick_wall_with_Wooden_pillar','D_StructureSystem_Wood_or_Synthetic_Resins',
       'D_StructureSystem_SRC','D_StructureSystem_NaN','D_structure_6','D_structure_5','D_structure_4',
       'D_structure_3','D_structure_2','D_structure_1','D_floor','D_floorTA','D_floorTAGround',
       'D_1floorCorridorCol','D_1floorCorridorColA','D_1floorClassCol','D_1floorClassColA',
       'D_1floorInsideCol','D_1floorInsideColA','D_X4brickwall','D_X3brickwall','D_YRCwallA',
       'D_Y4brickwall','D_Y3brickwall','D_basintype','D_475Acc','D_I','D_Demand','D_Tx','D_Ty',
       'D_sds','D_sd1','D_td0','D_sms','D_sm1','D_tm0','D_windows','D_patitionwall','D_nonstructure',
       'D_CLlarge','D_CLsmall','D_MaxCl','D_NeutralDepth','D_Ra_Capacity','D_Ra_CDR','Total_Height',
       'Total_DeadLoad','Total_LiveLoad','Total_Floor','Avg_Confc','Avg_MBfy','Avg_stify','D_isR']

dataframe = read_csv(filename, names=names)
dataframe2 = read_csv(filename2, names=names)
array = dataframe.values
array2 = dataframe2.values

X_train = array[1:,0:58].astype(float)
Y_train = array[1:,58].astype(int)
X_test = array2[1:,0:58].astype(float)


X_train2=X_train.copy()
X_test2=X_test.copy()
names2=np.delete(names, 58)

#==========================================

#============前處理

# # delete nan col
columns_to_delete = [39, 40, 41, 49, 50, 36, 37, 38]
X_train2 = np.delete(X_train2, columns_to_delete, 1)
X_test2 = np.delete(X_test2, columns_to_delete, 1)
names2 = np.delete(names2, columns_to_delete)

# # yesrs
[h,w]=X_train2.shape
for i in range(0,h):
    if X_train2[i,0]>3000 or X_train2[i,0]<-3000:
        X_train2[i,0]=X_train2[i,0]/525960/60+1970
    

# # Handle extreme data
percentile_90 = np.nanpercentile(X_train2, 90, axis=0)
X_train2[X_train2 < 0] = 0
X_train2[X_train2 > percentile_90 * 20] = np.nan

X_train_copy=X_train2.copy()
X_test_copy=X_test2.copy()
names_copy=names2.copy()

# # imputer
imputer = IterativeImputer(estimator=BayesianRidge(max_iter=300,tol=1e-2),imputation_order='ascending',random_state=42)
imputed_X_train = imputer.fit_transform(X_train2)
imputed_X_test = imputer.transform(X_test2)

# # RobustScaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(imputed_X_train)
X_test_scaled = scaler.transform(imputed_X_test)


# importances_mean
from sklearn.inspection import permutation_importance
model_imp=XGBClassifier(n_estimators=350, max_depth=15,learning_rate=0.1,gamma=0.2,scale_pos_weight=670/2664)
model_imp.fit(X_train_scaled, Y_train)

feature_importance = model_imp.feature_importances_

results = cross_val_score(model_imp, X_train_scaled, Y_train, cv=5,scoring='accuracy')
print(results.mean())

# # plot
# plt.bar(range(len(model_imp.feature_importances_)), model_imp.feature_importances_)
# plt.show()

# plot
# perm_importance_knn_sc = permutation_importance(model_imp, X_train_scaled, Y_train)
# plt.bar(range(len(model_imp.feature_importances_)), perm_importance_knn_sc.importances_mean)
# plt.xlabel("Permutation Importance")
# plt.show()

# plot
perm_importance = permutation_importance(model_imp, X_train_scaled, Y_train)
importances_mean = perm_importance.importances_mean
sorted_indices = importances_mean.argsort()[::-1]
top_10_indices = sorted_indices[:10]
top_10_importances = importances_mean[top_10_indices]
plt.bar(range(10), top_10_importances)
plt.xticks(range(10), top_10_indices)
plt.xlabel("Feature Index")
plt.ylabel("Permutation Importance")
plt.show()

# # 只留重要的colume
col=[48,46,42,12,40,30,22,31,24]
X_train3 = X_train_copy[:,col]
X_test3=X_test_copy[:,col]
name3=names_copy[col]

# # imputer
imputer = IterativeImputer(estimator=BayesianRidge(max_iter=300,tol=1e-2),imputation_order='ascending',random_state=42)
imputed_X_train3 = imputer.fit_transform(X_train3)
imputed_X_test3 = imputer.transform(X_test3)

# # RobustScaler
scaler = RobustScaler()
X_train_scaled3 = scaler.fit_transform(imputed_X_train3)
X_test_scaled3 = scaler.transform(imputed_X_test3)

# # 加入smote
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42,k_neighbors=8)
X_train_final, Y_train_final = smote.fit_resample(X_train_scaled3, Y_train)


# # XGBClassifier
model = XGBClassifier(n_estimators=350, max_depth=15, learning_rate=0.1, gamma=0.2, scale_pos_weight=1)
model.fit(X_train_final, Y_train_final)

pred = model.predict(X_test_scaled3)

results1 = cross_val_score(model, X_train_final, Y_train_final, cv=5,scoring='accuracy')
results2 = cross_val_score(model, X_train_final, Y_train_final, cv=5,scoring='f1')

print("accuracy = ",results1.mean())
print("f1 = ",results2.mean())

sum0=0
for i in range(len(pred)):
    if pred[i]==0:
        sum0+=1
sum1=len(pred)-sum0
print(f"0={sum0}")
print(f"1={sum1}")

#==============================================================

# # print excel
# list1 = np.arange(1,1267)
# result_df = pd.DataFrame({'id': list1, 'label': pred})
# result_df.to_csv('prediction.csv', index=False)

#=============================================================
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X_train_final,Y_train_final,test_size=0.2,random_state=42)
model = XGBClassifier(n_estimators=350, max_depth=15, learning_rate=0.1, gamma=0.2, scale_pos_weight=1)

model.fit(x_train, y_train)
y_train_hat = model.predict(x_train)
y_test_hat = model.predict(x_test)


print('Test performance')
print('-------------------------------------------------------')
print(classification_report(y_test, y_test_hat))


print('')

print('Confusion matrix')
print('-------------------------------------------------------')
print(confusion_matrix(y_test, y_test_hat))

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns

cm = confusion_matrix(y_test, y_test_hat)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=.5)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

precision = precision_score(y_test, y_test_hat)
recall = recall_score(y_test, y_test_hat)
f1 = f1_score(y_test, y_test_hat)

print('')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')
