## 3.1 基本配置

### 常用的包的导入

```Python
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,  DataLoader
import torch.optim as optimizer
```
### 超参数的设置

- batch size
- learning rate (初始学习率)
- 训练次数 epochs
- GPU设置

GPU的设置有两种常见的方式：

``` Python
# 方案一：使用os.environ, 这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# 方案二：使用“device”, 后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

## 3.2 数据的读入

PyTorch 的数据读入通常是通过 `Dataset + Dataloader` 的形式完成的,  Dataset定义好数据的格式和数据变换形式,  DataLoader 用 iterative 的方式不断读入批次数据.

我们可以定义自己的Dataset类来实现灵活的数据读取,  定义的类需要继承PyTorch自身的Dataset类. 主要包含三个函数: 

-   `__init__`: 用于向类中传入外部参数,  同时定义样本集
    
-   `__getitem__`: 用于逐个读取样本集合中的元素,  可以进行一定的变换,  并将返回训练/验证所需的数据
    
-   `__len__`: 用于返回数据集的样本数

e.g. 自定义的 Dataset 格式

```Python
class MyDataset(Dataset):
    def __init__(self,  data_dir,  info_csv,  image_list,  transform=None):
        """
        Args:
            data_dir: path to image directory.
            info_csv: path to the csv file containing image indexes
                with corresponding labels.
            image_list: path to the txt file contains image names to training/validation set
            transform: optional transform to be applied on a sample.
        """
        label_info = pd.read_csv(info_csv)
        image_file = open(image_list).readlines()
        self.data_dir = data_dir
        self.image_file = image_file
        self.label_info = label_info
        self.transform = transform

    def __getitem__(self,  index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_file[index].strip('\n')
        raw_label = self.label_info.loc[self.label_info['Image_index'] == image_name]
        label = raw_label.iloc[:, 0]
        image_name = os.path.join(self.data_dir,  image_name)
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image,  label

    def __len__(self):
        return len(self.image_file)
```

## 3.3 模型构建

### 神经网络的构造

构建神经网络一般都基于 `nn.Module` 完成构建. Module 类是 nn 模块里提供的一个模型构造类, 是所有神经⽹网络模块的基类, 我们可以继承它来定义我们想要的模型.

e.g. MLP (多重感知机) 的构造

```Python
import torch
from torch import nn

class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)
```

注意，这里并没有将 Module 类命名为 Layer (层)或者 Model (模型)之类的名字，这是因为该类是一个可供⾃由组建的部件。它的子类既可以是⼀个层(如PyTorch提供的 Linear 类)，⼜可以是一个模型(如这里定义的 MLP 类)，或者是模型的⼀个部分。

### 层

深度学习的一个魅力在于神经网络中各式各样的层，例如全连接层、卷积层、池化层与循环层等等。虽然PyTorch提供了⼤量常用的层，但有时候我们依然希望⾃定义层。

- **不含模型参数的层**

我们先介绍如何定义一个不含模型参数的自定义层。下⾯构造的 MyLayer 类通过继承 Module 类自定义了一个**将输入减掉均值后输出**的层，并将层的计算定义在了 forward 函数里。这个层里不含模型参数。

```Python
import torch
from torch import nn

class MyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()  
```

测试，实例化该层，然后做前向计算
```Python
layer = MyLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))
```

```Python
tensor([-2., -1.,  0.,  1.,  2.])
```

- **含模型参数的层**

我们还可以自定义含模型参数的自定义层。其中的模型参数可以通过训练学出。

Parameter 类其实是 Tensor 的子类，如果一 个 Tensor 是 Parameter ，那么它会⾃动被添加到模型的参数列表里。所以在⾃定义含模型参数的层时，我们应该将参数定义成 Parameter ，除了直接定义成 Parameter 类外，还可以使⽤ ParameterList 和 ParameterDict 分别定义参数的列表和字典。

- **二维卷积层**

二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。

```Python
import torch
from torch import nn

# 卷积运算（二维互相关）
def corr2d(X, K): 
    h, w = K.shape
    X, K = X.float(), K.float()
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

卷积窗口形状为 $(p \times q)$ 的卷积层称为$(p \times q)$ 卷积层。同样，$(p \times q)$ 卷积或 $(p \times q)$ 卷积核说明卷积核的高和宽分别为 p 和 q 。

填充(padding)是指在输⼊高和宽的两侧填充元素(通常是0元素)。

下面的例子里我们创建一个⾼和宽为3的二维卷积层，然后设输⼊高和宽两侧的填充数分别为1。给定一 个高和宽为8的输入，我们发现输出的高和宽也是8。

```Python
import torch
from torch import nn

# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:]) # 排除不关心的前两维:批量和通道

# 注意这里是两侧分别填充1⾏或列，所以在两侧一共填充2⾏或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3,padding=1)

X = torch.rand(8, 8)
comp_conv2d(conv2d, X).shape

torch.Size([8, 8])

当卷积核的高和宽不同时，我们也可以通过设置高和宽上不同的填充数使输出和输入具有相同的高和宽。

# 使用高为5、宽为3的卷积核。在⾼和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

torch.Size([8, 8])
```

在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上往下 的顺序，依次在输⼊数组上滑动。我们将每次滑动的行数和列数称为步幅(stride)。

填充可以增加输出的高和宽。这常用来使输出与输入具有相同的高和宽。

步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的 ( 为大于1的整数)。

- **池化层**

池化层每次对输入数据的一个固定形状窗口(⼜称池化窗口)中的元素计算输出。不同于卷积层里计算输⼊和核的互相关性，池化层直接计算池化窗口内元素的最大值或者平均值。该运算也 分别叫做最大池化或平均池化。在二维最⼤池化中，池化窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输⼊数组上滑动。当池化窗口滑动到某⼀位置时，窗口中的输入子数组的最大值即输出数组中相应位置的元素。

下面把池化层的前向计算实现在`pool2d`函数里。

```Python
import torch
from torch import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=torch.float)
pool2d(X, (2, 2))


pool2d(X, (2, 2), 'avg')
```

我们可以使用`torch.nn`包来构建神经网络。我们已经介绍了`autograd`包，`nn`包则依赖于`autograd`包来定义模型并对它们求导。一个`nn.Module`包含各个层和一个`forward(input)`方法，该方法返回`output`。

## 3.4 模型初始化

初始化是神经网络构建中非常重要的一部分. 通过合理地初始化权重, 可以让训练更加高效.

### torch.nn.init内容

通过访问torch.nn.init的官方文档[链接](https://pytorch.org/docs/stable/nn.init.html) ，我们发现`torch.nn.init`提供了以下初始化方法： 
1. `torch.nn.init.uniform_`(tensor, a=0.0, b=1.0) 
2. `torch.nn.init.normal_`(tensor, mean=0.0, std=1.0) 
3. `torch.nn.init.constant_`(tensor, val) 
4. `torch.nn.init.ones_`(tensor) 
5. `torch.nn.init.zeros_`(tensor) 
6. `torch.nn.init.eye_`(tensor) 
7. `torch.nn.init.dirac_`(tensor, groups=1) 
8. `torch.nn.init.xavier_uniform_`(tensor, gain=1.0) 
9. `torch.nn.init.xavier_normal_`(tensor, gain=1.0) 
10. `torch.nn.init.kaiming_uniform_`(tensor, a=0, mode='fan__in', nonlinearity='leaky_relu') 
11. `torch.nn.init.kaiming_normal_`(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu') 
12. `torch.nn.init.orthogonal_`(tensor, gain=1) 
13. `torch.nn.init.sparse_`(tensor, sparsity, std=0.01) 
14. `torch.nn.init.calculate_gain`(nonlinearity, param=None) 

关于计算增益如下表：
![](https://s1.vika.cn/space/2022/05/20/b1724c1bba82476fbc405cd34dcb6cd1)


我们可以发现这些函数除了`calculate_gain`，所有函数的后缀都带有下划线，意味着**这些函数将会直接原地更改输入张量的值**。

### 初始化函数的封装

人们常常将各种初始化方法定义为一个`initialize_weights()`的函数并在模型初始后进行使用。

```Python
def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()	
```

这段代码流程是遍历当前模型的每一层，然后判断各层属于什么类型，然后根据不同类型层，设定不同的权值初始化方法.

## 3.5 损失函数

在深度学习广为使用的今天，我们可以在脑海里清晰的知道，一个模型想要达到很好的效果需要学习，也就是我们常说的训练。一个好的训练离不开优质的负反馈，这里的损失函数就是模型的负反馈。

所以在PyTorch中，损失函数是必不可少的。它是数据输入到模型当中，产生的结果与真实标签的评价指标，我们的模型可以按照损失函数的目标来做出改进。

## 3.6 训练和评估

完成了上述设定后就可以加载数据开始训练模型了。首先应该设置模型的状态：如果是训练状态，那么模型的参数应该支持反向传播的修改；如果是验证/测试状态，则不应该修改模型参数。在PyTorch中，模型的状态设置非常简便.

验证/测试的流程基本与训练过程一致，不同点在于：

-   需要预先设置torch.no_grad，以及将model调至eval模式
    
-   不需要将优化器的梯度置零
    
-   不需要将loss反向回传到网络
    
-   不需要更新optimizer
    

一个完整的图像分类的训练过程如下所示：

```python
def train(epoch):
    model.train()
    train_loss = 0
    # 
    for data, label in train_loader:
        data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(label, output)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    train_loss = train_loss/len(train_loader.dataset)
		print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
```

对应的，一个完整图像分类的验证过程如下所示：

```python
def val(epoch):       
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
            running_accu += torch.sum(preds == label.data)
    val_loss = val_loss/len(val_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, val_loss))
```

