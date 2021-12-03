# 逻辑回归模型训练 logistic_regression

## 说明：

- trainset.json: 内包含训练集数据，由1000条数据组成的列表，每条数据由一组输入数据和一个label值，输入数据为一个长度为9的数组，label为0则代表该输入数据属于第0类，label为1则代表该输入数据属于第1类。

- devset.json: 内包含验证集数据，格式同trainset.json，由200条数据组成的列表，每条数据由一组输入数据和一个label值，输入数据为一个长度为9的数组，label为0则代表该输入数据属于第0类，label为1则代表该输入数据属于第1类。

请基于以上两组数据，使用梯度下降法训练一个逻辑回归模型，并使用此模型对testset.json中的数据进行预测。

- testset.json: 内包含测试数据，是一个由100条测试数据组成的二维数组，形状为[100,9], 在完成函数拟合后，请使用得出的二分类模型对这100条测试数据进行预测，并将此文件补充为与trainset.json相同格式的文件。

## 要求：
1. 仅使用numpy矩阵运算库实现
2. 给出train训练集和valid验证集的准确率：运行后有终端输出和文件输出（./output/accuracy.txt）
3. 根据每次迭代的loss绘制train和valid的loss曲线图，横坐标为iterations，纵坐标为loss：（./output/loss.jpg）
4. 完成后请将源代码和补充完整的testset.json一起打包发送到微信群：补充完整的testset.json（./output/fill-testset.json）
