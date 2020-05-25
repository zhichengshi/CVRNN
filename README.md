#### CVRNN
该项目主要利用卷积和序列神经网络解决了两个任务，代码分类和近似代码搜索

执行步骤：

（1）git clone https://github.com/zhichengshi/CVRNN

（2）该项目所使用的数据已经被处理好存放在dataset文件夹中，链接：https://pan.baidu.com/s/1_9JbeNL0Ua4ffY0kpz8uzg 提取码：nda8，将dataset文件夹放到CVRNN文件夹下

#### 代码分类任务：
cvrnn模型直接运行cvRnn文件夹下的trainWithValidate.py文件即可。

tbcnn模型直接运行tbcnn文件夹下的trainWithValidate文件即可

#### 相似代码搜索
#### cvrnn相似代码搜索
(1)首先使用代码分类任务训练模型，即执行cvRnn文件夹下的trainWithValidate.py

(2)然后执行cvRnn/generateCodeMatrix.py生成代码向量，代码向量会放到dataset/matrix/cvrnn中，negative.pkl表示负样本向量，positive.pkl（代码向量数目为200）表示正样本向量

(3)然后在codeSearch/codeSearch.py中指定上面得到的negative.pkl以及positive.pkl的路径

#### tbcnn相似代码搜索
(1)首先使用代码分类任务训练模型，即执行tbcnn文件夹下的trainWithValidate.py

(2)然后执行tbcnn/generateCodeMatrix.py生成代码向量，代码向量会放到dataset/matrix/tbcnn中，negative.pkl表示负样本向量，positive.pkl（代码向量数目为200）表示正样本向量

(3)然后在codeSearch/codeSearch.py中指定上面得到的negative.pkl以及positive.pkl的路径

#### 绘制leetcode代码向量的二维打印图
运行cvRnn文件夹下的drawPicture.py即可

