# cnn-lstm
reference  
 [cnn lstm tutorial and keras implement](https://machinelearningmastery.com/cnn-long-short-term-memory-networks/)

##prepair
### _keras model visualize_
 * 1.use model.summary

		model.summary()
* 2.use keras vislualize utils

		pip install graphviz

	for python3 use

		pip install pydot



## reference

* [Image denoising and restoration with CNN-LSTM Encoder Decoder with Direct Attention](https://arxiv.org/abs/1801.05141)
* [CONVOLUTIONAL, LONG SHORT-TERM MEMORY,
FULLY CONNECTED DEEP NEURAL NETWORKS](https://research.google.com/pubs/archive/43455.pdf)










__flatten layer__
>The last stage of a convolutional neural network (CNN) is a classifier. It is called a dense layer, which is just an artificial neural network (ANN) classifier.
>And an ANN classifier needs individual features, just like any other classifier. This means it needs a feature vector.
>Therefore, you need to convert the output of the convolutional part of the CNN into a 1D feature vector, to be used by the ANN part of it. This operation is called flattening. It gets the output of the convolutional layers, flattens all its structure to create a single long feature vector to be used by the dense layer for the final classification.


__reference__
[quora answer by Mehmet Ufuk Dalmis](https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network)



#keras issues

[How to use Different Batch Sizes when Training and Predicting with LSTMs](https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/)  
[What should be the batch for Keras LSTM CNN to process image sequence](https://stackoverflow.com/questions/46187124/what-should-be-the-batch-for-keras-lstm-cnn-to-process-image-sequence)  
[How are inputs fed into the LSTM/RNN network in mini-batch method?](https://www.quora.com/How-are-inputs-fed-into-the-LSTM-RNN-network-in-mini-batch-method)  
[What is batch size in neural network?](https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network)  

#scene text detection papers & resources
1. [Detecting and Recognizing Text in Natural Images](https://www.youtube.com/watch?v=Vb0adWGfWqo)  
2. [vikesh_final](http://cs231n.stanford.edu/reports/2015/pdfs/vikesh_final.pdf)  
3. [ICDAR2017 Competition on Reading Chinese Text in the Wild](https://vision.cornell.edu/se3/wp-content/uploads/2017/08/PID4980421.pdf)  
4. [Detecting Oriented Text in Natural Images by Linking Segments](https://www.researchgate.net/publication/315454607_Detecting_Oriented_Text_in_Natural_Images_by_Linking_Segments)  
5. [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://arxiv.org/abs/1507.05717)  
6. [TextBoxes: A Fast Text Detector with a Single Deep Neural Network](https://arxiv.org/abs/1611.06779)  
7. [AID: A Benchmark Dataset for Performance Evaluation of Aerial Scene Classification](https://arxiv.org/abs/1608.05167)  
8. [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915) 
