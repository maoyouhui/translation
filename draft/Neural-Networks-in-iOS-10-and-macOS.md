>* 原文链接 : [Neural Networks in iOS 10 and macOS](https://www.bignerdranch.com/blog/neural-networks-in-ios-10-and-macos/)
* 原文作者 : [Bolot Kerimbaev](https://www.bignerdranch.com/about-us/nerds/bolot-kerimbaev/)
* 译文出自 : [猫友会翻译计划](https://github.com/maoyouhui/translation)
* 译者 : [尹佳冀](http://www.taijicoder.com/)
* 校对者: [李叶伟](mailto:leeyevi@gmail.com)

# iOS10和macOS中的神经网络

很长一段时间以来苹果一直在他们的产品中使用机器学习：Siri回答我们的问题，娱乐大家，iPhoto中的人脸识别，Mail程序中的垃圾邮件检测。作为App开发者，我们有机会接触苹果提供的API，如人脸识别 ，并且从iOS 10开始，我们将获得的语音识别和SiriKit这些高层次的API。

有时候，我们想要超越平台内置API的的狭窄界限，创造独特的东西。多数时候，我们使用一些现成的库或直接用有快速计算能力的Accelerate 框架或Metal框架构建机器学习的东西。

例如，我的同事为我们的办公室构建了进入系统，使用一台iPad检测脸部图像，然后在Slack上发送一个GIF，并允许用户使用自定义的命令来开门。
![门铃人脸识别](https://www.bignerdranch.com/img/blog/2016/06/doorway.gif)

但是现在我们有了第一方的神经网络的直接支持：在WWDC 2016上，苹果推出了神经网络的API，而且是两个，称为Basic Neural Network Subroutines（BNNS）和 Convolutional Neural Networks（CNN）。

## 机器学习和神经网络
AI先驱[阿瑟·塞缪尔](https://en.wikipedia.org/wiki/Arthur_Samuel)定义了[机器学习](https://en.wikipedia.org/wiki/Machine_learning)是一个“在没有经过显式编码的情况下赋予电脑学习能力的研究领域。”机器学习系统可以很好的用来处理一些不能使用传统模型来简单描述的数据。

例如，我们可以很容易地编写一个计算房子平方英尺（面积）的程序，因为提供了所有的房间和其他空间的尺寸和形状，但对于计算房子的价值就不是我们可以通过一个公式计算能解决的了。但是，机器学习系统可以很好地解决这样的问题。通过提供已知的真实世界的数据给系统，如市场价值，房子的大小，卧室数量等，我们可以训练它去预测价格。

神经网络是构建机器学习系统最常用的模型之一。虽然神经网络的数学基础从20世纪40年代起已经发展超过了半个世纪，直到20世纪80年代并行计算的出现才让它更为可行，21世纪初对[深度学习](https://en.wikipedia.org/wiki/Deep_learning)的兴起才激起了神经网络的复兴潮流。

神经网络由多个层（layer）构成的，其中每一层由一个或多个节点（node）组成。最简单的神经网络有三层：输入层，隐含层和输出层。输入层节点可以表示为图像中的各个像素或者一些其他的参数。输出层节点是通常是分类的结果，如果我们试图自动检测照片的内容，分类的结果就是“狗”或“猫”。隐含层节点被配置为对输入执行操作或者应用激活函数（activation function）。

![神经网络图](https://www.bignerdranch.com/img/blog/2016/06/neural_network_diagram.png)

## 层的类型
三种常见的层是 池化层（pooling layer），卷积层（convolution layer）和全连接层（fully connected layer）。

池化层，典型地是使用输入的最大值或平均值，归并数据，来减少数据量的同时保留有用信息。一系列的卷积层和池化层连起来逐步把一张照片提取成层次越来越高的特征的集合。

卷积层通过对图像上的每个像素进行卷积操作来实现图像变换。如果你使用过Pixelmator或Photoshop滤镜，你很有可能已经用过了卷积矩阵运算。卷积矩阵通常是一个应用到输入图像像素邻域内的，为了计算输出图像新的像素值的3×3或5×5矩阵。为了获得输出像素的值，我们就乘上原图像中的窗口内邻域像素值，并计算平均值。

例如，这个卷积矩阵就会使图像模糊：

	1 1 1 
	1 1 1
	1 1 1
而这一个就会使图像锐化：

	0 -1 0 
	-1 5 -1 
	0 -1 0
神经网络的卷积层使用卷积矩阵来处理输入并为下一层生成数据，比如提取图像中的新的特征，像边缘。

一个全连接的层可以认为是一个和原始图像一样大小的滤波器卷积层。换句话说，你可以认为全连接层是一个会分配权重到图像各个像素，算出平均值，最后给出了一个输出值的函数。


## 训练和推理
每一层需要配置适当的参数。例如，卷积层需要输入和输出图像（尺寸，通道的数目等）的信息，以及卷积层的参数（内核大小，矩阵等）。全连接层则是由输入和输出向量，激活函数，和权重来定义的。

训练神经网络就是要获得这些参数的过程。这是通过向神经网络传递输入，确定输出，误差测量（即定义实际结果和预测结果差值有多大）来完成的，并通过反向传播（backpropagation）来调整权重。训练神经网络可能需要成百，上千甚至数百万的样本。

目前，苹果公司新的机器学习API，可以构建的神经网络，只能做推理，不能做训练。好消息是， [Big Nerd Ranch 可以做到](https://training.bignerdranch.com/classes/advanced-ios-bootcamp)。

##Accelerate：BNNS
第一个新的API是Accelerate框架的一部分，被称为BNNS，代表[Basic Neural Network Subroutines](https://developer.apple.com/reference/accelerate/1912851-bnns) 。BNNS作为BLAS（Basic Linear Algebra Subroutines）的补充，被用在机器学习第三方应用中。

BNNS定义的层在BNNSFilter类中。Accelerate支持三种类型的层：卷积层（由BNNSFilterCreateConvolutionLayer函数创建），全连接层（ BNNSFilterCreateFullyConnectedLayer ）和池化层（ BNNSFilterCreatePoolingLayer ）。

[MNIST数据库](https://en.wikipedia.org/wiki/MNIST_database)是一个广为人知的手写数字的数据集，包含了数以万计的被扫描的和调整大小到20*20像素的图像。

一种处理图像数据的方法是将图像转换成向量，并把它传给一个全连接层。对于MNIST数据，一个20×20的图像会成为400个值的向量。下面就展示了手写的数字“1”怎样转换到向量：
![手写转换为矢量](https://www.bignerdranch.com/img/blog/2016/06/handwritten_one_vector.png)

以下是一个样例代码，用于配置一个 由大小为400的向量作为输入的全连接层，采用sigmoid激活函数，并且输出大小为25的向量：

```
 // input layer descriptor
    BNNSVectorDescriptor i_desc = {
        .size = 400,
        .data_type = BNNSDataTypeFloat32,
        .data_scale = 0,
        .data_bias = 0,
    };

    // hidden layer descriptor
    BNNSVectorDescriptor h_desc = {
        .size = 25,
        .data_type = BNNSDataTypeFloat32,
        .data_scale = 0,
        .data_bias = 0,
    };

    // activation function
    BNNSActivation activation = {
        .function = BNNSActivationFunctionSigmoid,
        .alpha = 0,
        .beta = 0,
    };

    BNNSFullyConnectedLayerParameters in_layer_params = {
        .in_size = i_desc.size,
        .out_size = h_desc.size,
        .activation = activation,
        .weights.data = theta1,
        .weights.data_type = BNNSDataTypeFloat32,
        .bias.data_type = BNNSDataTypeFloat32,
    };

    // Common filter parameters
    BNNSFilterParameters filter_params = {
        .version = BNNSAPIVersion_1_0;   // API version is mandatory
    };

    // Create a new fully connected layer filter (ih = input-to-hidden)
    BNNSFilter ih_filter = BNNSFilterCreateFullyConnectedLayer(&i_desc, &h_desc, &in_layer_params, &filter_params);

    float * i_stack = bir; // (float *)calloc(i_desc.size, sizeof(float));
    float * h_stack = (float *)calloc(h_desc.size, sizeof(float));
    float * o_stack = (float *)calloc(o_desc.size, sizeof(float));

    int ih_status = BNNSFilterApply(ih_filter, i_stack, h_stack);
```

## Metal！

Metal框架能做的更多吗？事实上，确实，因为第二个神经网络API是Metal Performance Shaders（MPS）框架的一部分。Acclerate是在CPU上执行快速计算框架，而Metal则把GPU使用到了极致。Metal中的叫法是CNN，就是[卷积神经网络](https://en.wikipedia.org/wiki/Convolutional_neural_network)。

MPS拥有一个类似的API。创建一个卷积层需要使用MPSCNNConvolutionDescriptor和MPSCNNConvolution功能。对于池化层， MPSCNNPoolingMax会提供的参数。全连接层是由MPSCNNFullyConnected函数。激活函数是由以下子类来定义MPSCNNNeuron ： MPSCNNNeuronLinear ， MPSCNNNeuronReLU ， MPSCNNNeuronSigmoid ， MPSCNNNeuronTanH ， MPSCNNNeuronAbsolute 。

## BNNS和CNN对比

下表列出了Accelerate和Metal中激活函数：

| Accelerate/BNNS  | Metal Performance Shaders/CNN |
| ------------- | ------------- |
| BNNSActivationFunctionIdentity  |  |
| BNNSActivationFunctionRectifiedLinear  | MPSCNNNeuronReLU  |
| |MPSCNNNeuronLinear |
| BNNSActivationFunctionLeakyRectifiedLinear	 | |
| BNNSActivationFunctionSigmoid	| MPSCNNNeuronSigmoid |
| BNNSActivationFunctionTanh	| MPSCNNNeuronTanH |
| BNNSActivationFunctionScaledTanh	 | |
| BNNSActivationFunctionAbs	| MPSCNNNeuronAbsolute |

池化函数：

| Accelerate/BNNS  | Metal Performance Shaders/CNN |
| ------------- | ------------- |
| BNNSPoolingFunctionMax  | MPSCNNPoolingMax |
| BNNSPoolingFunctionAverage  | MPSCNNPoolingAverage  |

Accelerate和Metal框架提供一组非常相似的神经网络的的函数集，所以怎么选择取决于每个程序。虽然通常GPU的是各种机器学习所需计算的首选，数据的位置也有可能会导致Metal 的CNN 执行地比Accelerate BNNS版本 表现差。如果神经网络是操作GPU中的图像，比如使用MPSImage和新MPSTemporaryImage ，那么Metal是明显的选择。