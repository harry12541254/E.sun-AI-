# E.sun-AI-

本次信用卡詐欺偵測專案專案主要分成兩部分，分別為

1. data_preprocessed：
將原有資料表的欄位特徵，透過一些組合新增七項組合過後的欄位，針對信用卡的靜態以及動態資料進行整理，例如卡號的刷卡頻率、交易金額或是交易模式改變等，
以利在新樣本進入時能有更多判斷依據。

2. model：
本次模型因為有較多的類別變數，因此在模型選擇上採用了CatBoost的提升梯度算法模型，並且在區分訓練和驗證資料時，採分層抽樣以利模型能夠學習詐欺資料。
Boosting演算法在處理非線型模型有非常優秀的表現，且Catboost更是針對類別變數有良好的處理方法，搭配上強大的運算效能，讓Catboost能有不錯的預測結果。

備註：因最後上傳的超參數未妥善儲存，導致貴單位在復現模型時，可能會產生些許的誤差，但超參數值皆是採用與資料夾內接近的數值，且參數設置不變，
數據前處理上仍維持不變，對於缺失非常抱歉。
