# CNNs-and-Computer-Vision-CS231n
org:http://cs231n.github.io/

Content:
1.Assignment-1:
      In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

      understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
      understand the train/val/test splits and the use of validation data for hyperparameter tuning.
      develop proficiency in writing efficient vectorized code with numpy
      implement and apply a k-Nearest Neighbor (kNN) classifier
      implement and apply a Multiclass Support Vector Machine (SVM) classifier
      implement and apply a Softmax classifier
      implement and apply a Two layer neural network classifier
      understand the differences and tradeoffs between these classifiers
      get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)




      Q1: k-Nearest Neighbor classifier (20 points)
      The IPython Notebook knn.ipynb will walk you through implementing the kNN classifier.

      Q2: Training a Support Vector Machine (25 points)
      The IPython Notebook svm.ipynb will walk you through implementing the SVM classifier.

      Q3: Implement a Softmax classifier (20 points)
      The IPython Notebook softmax.ipynb will walk you through implementing the Softmax classifier.

      Q4: Two-Layer Neural Network (25 points)
      The IPython Notebook two_layer_net.ipynb will walk you through the implementation of a two-layer neural network classifier.

      Q5: Higher Level Representations: Image Features (10 points)
      The IPython Notebook features.ipynb will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

2. Assignment - 2:
        In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

        understand Neural Networks and how they are arranged in layered architectures
        understand and be able to implement (vectorized) backpropagation
        implement various update rules used to optimize Neural Networks
        implement Batch Normalization and Layer Normalization for training deep networks
        implement Dropout to regularize networks
        understand the architecture of Convolutional Neural Networks and get practice with training these models on data
        gain experience with a major deep learning framework, such as TensorFlow or PyTorch.


        Q1: Fully-connected Neural Network (20 points)
        The IPython notebook FullyConnectedNets.ipynb will introduce you to our modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

        Q2: Batch Normalization (30 points)
        In the IPython notebook BatchNormalization.ipynb you will implement batch normalization, and use it to train deep fully-connected networks.

        Q3: Dropout (10 points)
        The IPython notebook Dropout.ipynb will help you implement Dropout and explore its effects on model generalization.

        Q4: Convolutional Networks (30 points)
        In the IPython Notebook ConvolutionalNetworks.ipynb you will implement several new layers that are commonly used in convolutional networks.

        Q5: PyTorch / TensorFlow on CIFAR-10 (10 points)
        For this last part, you will be working in either TensorFlow or PyTorch, two popular and powerful deep learning frameworks. You only need to complete ONE of these two notebooks. You do NOT need to do both, and we will not be awarding extra credit to those who do.

        Open up either PyTorch.ipynb or TensorFlow.ipynb. There, you will learn how the framework works, culminating in training a convolutional network of your own design on CIFAR-10 to get the best performance you can.

        NOTE: The PyTorch notebook requires PyTorch version 0.4, which was released on 4/24/2018. You can install this version of PyTorch using conda or pip by following the instructions here: http://pytorch.org/

3. Assignment - 3

        In this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and also this model to implement Style Transfer. Finally, you will train a Generative Adversarial Network to generate images that look like a training dataset!
        The goals of this assignment are as follows:

        Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
        Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) networks.
        Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
        Explore various applications of image gradients, including saliency maps, fooling images, class visualizations.
        Understand and implement techniques for image style transfer.
        Understand how to train and implement a Generative Adversarial Network (GAN) to produce images that resemble samples from a dataset.


        You can do Questions 3, 4, and 5 in TensorFlow or PyTorch. There are two versions of each of these notebooks, one for TensorFlow and one for PyTorch. No extra credit will be awarded if you do a question in both TensorFlow and PyTorch.
        Q1: Image Captioning with Vanilla RNNs (25 points)
        The Jupyter notebook RNN_Captioning.ipynb will walk you through the implementation of an image captioning system on MS-COCO using vanilla recurrent networks.

        Q2: Image Captioning with LSTMs (30 points)
        The Jupyter notebook LSTM_Captioning.ipynb will walk you through the implementation of Long-Short Term Memory (LSTM) RNNs, and apply them to image captioning on MS-COCO.

        Q3: Network Visualization: Saliency maps, Class Visualization, and Fooling Images (15 points)
        The Jupyter notebooks NetworkVisualization-TensorFlow.ipynb /NetworkVisualization-PyTorch.ipynb will introduce the pretrained SqueezeNet model, compute gradients with respect to images, and use them to produce saliency maps and fooling images. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

        Q4: Style Transfer (15 points)
        In the Jupyter notebooks StyleTransfer-TensorFlow.ipynb/StyleTransfer-PyTorch.ipynb you will learn how to create images with the content of one image but the style of another. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awardeded if you complete both notebooks.

        Q5: Generative Adversarial Networks (15 points)
        In the Jupyter notebooks GANS-TensorFlow.ipynb/GANS-PyTorch.ipynb you will learn how to generate images that match a training dataset, and use these models to improve classifier performance when training on a large amount of unlabeled data and a small amount of labeled data. Please complete only one of the notebooks (TensorFlow or PyTorch). No extra credit will be awarded if you complete both notebooks.


