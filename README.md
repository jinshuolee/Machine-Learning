<img width="1162" height="542" alt="image" src="https://github.com/user-attachments/assets/5c24cffb-daa2-4234-a00d-674bbd59af56" /># 用機器學習進行校舍耐震能力評估

介紹
https://github.com/jinshuolee/Machine-Learning/blob/main/figure%201.png?raw=true

資料集
https://github.com/jinshuolee/Machine-Learning/blob/main/figure%202.png?raw=true

資料前處理

1.	讀取資料。
2.	將過少的資料進行刪除。
3.	進行年代轉換，原本資料為Unixtime格式，使用公式轉換成西元年。
4.	處理極值問題，將小於0的值設為0，過大的資料設為空值。
5.	使用imputer進行缺失值填充。
6.	使用RobustScaler將資料進行縮放。
7.	計算模型的各特徵重要性，將原始資料中重要性低的資料刪除。
8.	原始資料經過刪除不重要特徵後，處理過的資料再進行一次imputer和RobustScaler。
9.	使用smote方法處理不平衡數據，透過smote方法採樣得到的新樣本。

使用的模型為XGBClassifier，最後的最佳訓練結果參數為XGBClassifier(n_estimators=350, max_depth=15, learning_rate=0.1, gamma=0.2, scale_pos_weight=1)

在尋找hyper-parameter tuning的過程中，中間嘗試很多次使用GridSearch和Cross-Validation來調整最佳參數，但是結果不如預期，因此就嘗試將樹的數量和深度調高，學習率調低一點來調整參數，重要特徵值的部分和同學討論後都不同，因此也是多嘗試用不同的特徵值下去訓練，得到最好的結果。驗證模型的訓練結果就是察看分數，由於F1數值與kaggle的分數不一定會一樣，於是先用程式同時查看F1 score 和accuracy分數避免單一分數影響過大，通常越高分訓練結果也會越好，查看訓練的結果後再調整參數。

