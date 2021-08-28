# CTR学习笔记

The code is not rigorously tested, if you find a bug, welcome PR ^_^ ~

- Run: python main.py --model DeepFM --step train --dataset census --clear_model 1
- Requirement: tensorflow 1.15

1. 已完成模型列表[支持数据集]

- FM [census]
- FFM [census]
- Embedding+MLP [census]
- wide & Deep [census]
- FNN [census]
- PNN [census]
- DeepFM [census & frappe]
- AFM [census & frappe]
- NFM [census & frappe]
- Deep Crossing [census]
- Deep & Cross [census & frappe]
- xDeepFM [census & frappe]
- FiBiNET [census & frappe]
- DIN [amazon]

2. 数据集
当前支持census, frappe数据集，详情见data目录，training parameter和preprocess与数据集绑定

3. 参考论文列表
- [GBDT+LR] Practical Lessons from Predicting Clicks on Ads at Facebook
- [FM] S. Rendle, Factorization machines
- [FM Model] Fast Context-aware Recommendations with Factorization Machines
- [FFM] Yuchin Juan，Yong Zhuang，Wei-Sheng Chin，Field-aware Factorization Machines for CTR Prediction
- [NCF] Neural Collaborative Filtering
- [Wide&Deep] Cheng H T, Koc L, Harmsen J, et al. Wide & deep learning for recommender systems
- [FNN] Weinan Zhang, Tianming Du, and Jun Wang. Deep learning over multi-field categorical data - - A case study on user response
- [PNN] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction
- [DeepFM] Huifeng Guo et all. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
- [AFM] Attentional Factorization Machines - Learning the Weight of Feature Interactions via Attention Networks
- [NFM] Neural Factorization Machines for Sparse Predictive Analytics
- [DCN] Deep & Cross Network for Ad Click Predictions
- [Deep Crossing] Deep Crossing - Web-Scale Modeling without Manually Crafted Combinatorial Features
- [xDeepFM] xDeepFM- Combining Explicit and Implicit Feature Interactions for Recommender Systems
- [FiBiNET]- Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction
- [AutoInt]- Automatic Feature Interaction Learning via Self-Attentive Neural Networks
- [DIN] Deep Interest Network for Click-Through Rate Prediction.
- [DIEN] Deep Interest Evolution Network for Click-Through Rate Prediction

4. 总结博客
- [CTR学习笔记&代码实现1-深度学习的前奏 LR->FFM](https://www.cnblogs.com/gogoSandy/p/12501846.html)
- [CTR学习笔记&代码实现2-深度ctr模型 MLP->Wide&Deep](https://www.cnblogs.com/gogoSandy/p/12658051.html)
- [CTR学习笔记&代码实现3-深度ctr模型 FNN->PNN->DeepFM](https://www.cnblogs.com/gogoSandy/p/12742417.html)
- [CTR学习笔记&代码实现4-深度ctr模型 NFM/AFM](https://www.cnblogs.com/gogoSandy/p/12814804.html)
- [CTR学习笔记&代码实现5-深度ctr模型 DeepCrossing -> DCN](https://www.cnblogs.com/gogoSandy/p/12892973.html)
- [CTR学习笔记&代码实现6-深度ctr模型 后浪 xDeepFM/FiBiNET](https://www.cnblogs.com/gogoSandy/p/13023265.html)
