# Optical-Character-Recognition-with-Deep-Learning-keras-and-TensorFlow
Traditional OCR (Optical Character Recognition) yields very good results only on very specific use cases like converting printed Portable Document Format (PDF) documents to on-screen text. However, in general, it is still considered challenging for problems like handwritten documents.
Deep learning driven Neural Networks can be used to dramatically improve the accuracy of handwritten text recognition. In our project, we will be using Python as the programming language. We will be discussing how handwriting text recognition can be done using deep learning, keras, and TensorFlow. 
All codes were executed on Jupyter Notebook or Google Colab Notebook. We divided our work into two levels of complexity:
1.	The first level to recognize single digits and single English alphabet letters. 
2.	The second level to recognize handwritten words using different approaches and the recognition of vehicle license plate codes. 
Dataset Description

1)	Mnist data set: This data set from keras is used for digit recognition. It consists of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
2)	AZ-handwritten-alphabets-in-csv-format: This dataset is downloaded from Kaggle. It contains 26 folders (A-Z) containing handwritten images in size 28x28 pixels, each alphabet in each image is stored as Gray-level.
3)	Our own data set of lowercase a-z letters: We generated our own dataset of 1,200 lowercase characters using Google Images. JavaScript was used to gather the image URLs and a Python file used to download the images to a folder.
4)	Handwritten words: We have created our own dataset using our own handwritten words. We have used offline methods (input information is obtained through static information (images)) which involve recognizing text once it's written down.
Levels of Complexity

Our project of text recognition using TensorFlow, keras and Deep learning is divided into two levels of complexity:
1) Recognize single handwritten letters and digits.
2) Recognize handwritten words and recognize vehicles’ license plates. 

HOW OCR Recognizes Text
Handwritten Text Recognition (HTR) system receives and Interpret handwritten input from sources such as scanned images. We use tools like OpenCV and provide TensorFlow development services to build a Neural Network (NN) which is trained on line-images from the off-line HTR dataset. We performed hyperparameter tuning to improve accuracy of our model.

Level One - Recognize single handwritten letters and digits

Methodology
For the first level of complexity, we concatenated the MNIST dataset containing handwritten digits, the A-Z handwritten letters dataset from Kaggle and the lowercase a-z letters dataset we created from Google Images. Before we combined the three datasets, a descriptive analysis was done and it appeared that the MNIST dataset was relatively balanced, while the uppercase A-Z handwritten letters dataset and lowercase a-z letters dataset were quite imbalanced. In addition, there are significantly more samples of uppercase letters (372,450) compared to digits (70,000) and lowercase letters (248). The distribution graphs are shown below: 

Fig 1
In order to rectify the serve class imbalance, we used the Synthetic Minority Oversampling Technique (SMOTE).
Next, preprocessing techniques like gray scaling, resizing and normalizing were used on the images. We have set a total of 62 output classes that are applicable to our project. The 62 output classes can be summarized into: 
a)	capital letters from A-Z 
b)	small letters from a-z
c)	numbers from 0-9

We then built CNN models with different hyperparameters to identify the best performing model in terms of accuracy.
Prediction: To predict how well our model recognizes handwritten letters, we wrote letters from a-z on a piece of paper, took pictures of them, resized and reshaped them, then tried to predict some letters. While it predicted well for some, the model did not predict well for some others, which shows that the model did not learn well and needed to be improved on. We therefore increased the epoch and batch size and our model predicted them correctly.
 
Fig 2
Results
With hyperparameter tuning, we have achieved an accuracy of 96.65% with loss: 0.1595 (Fig3)
  
Fig 3
Below shows the model accuracy and model loss.
  
Fig 4
 
Level Two - Recognize Handwritten Words and Recognize Vehicles’ License Plate
Approach 1: Recognize handwritten words
Methodology
In order to extract text from a document image, word segmentation is required to locate objects and boundaries in an image. It includes vertical scanning of the image, pixel-row by pixel-row from left to right and top to bottom. At each pixel the intensity is tested. Depending on the values of the pixels we group pixels into multiple regions from the entire image. The different region indicates different content in the image file.
Before image segmentation, we need to bring the image in a specific format to simplify the subsequent processing. The preprocessing includes digitization, noise removal, binarization, normalization. The preprocessing stage yields a “clean” document with sufficient amount of shape information, high compression, and low noise in a normalized image.
The result of image segmentation is a set of segments that collectively cover the entire image, or a set of contours extracted from the image (edge detection). The outputs are bounding boxes that correspond to each character of the word as shown in Figure 6.
We used SGD optimizer from level 1 complexity with a learning rate of 0.05 for compiling Segmented Characters and predicting our final output. 
Individual character images are resized to 28 X 28 and reshaped to (1 X 784) format for our predictive model.

 
Fig 5
 
						Fig 6
Approach 2: Recognize handwritten words with keras-ocr
Methodology
This approach uses the keras-ocr package to recognize handwritten words. The package ships with easy-to-use implementations of the CRAFT text detection model and the CRNN recognition model. Weights can be loaded into the model attribute of the detector and the recognizer however we found the default pretrained weights provided a good level of accuracy.




License Plate Recognition
Methodology
Part 1: Recognizing Vehicle’s License Plate
 
 Fig 7
We implemented a pre-trained model to detect and extract license plates of vehicle images. A function was then created to read and pre-process our images. In order to be compatible with matplotlib, the function was executed to read the parsing image, convert it to RGB, and normalize the image data to a number between 0 to 1. We can also resize all the images to the same dimension so we can visualize them.
Next, we executed a function which processed the raw image, sent it to our model and returned the plate image and its coordinates. Given the coordinates, a bounding box was drawn around the detected license plate.
Part 2: Character Segmentation
After the license plate had been detected from the image, we preprocessed the image of the license plate. The preprocessing techniques included converting the image to 255 scale, converting to grayscale, blurring image, image thresholding, and dilation. Figure 8 below shows the preprocessing images.
 
Fig 8
Next, we segmented our license characters by using contour. Contour is a line joining all continuous points which share the same color and intensity. By using contour, we have successfully identified the coordinates of the license characters, in order, from left to right.
 
Fig 9
Part 3: Recognize license plate characters

After we segmented the license characters, we converted our input images to digital digits or letters by using a Neural Network model. Therefore, we trained a Neural Network to predict the segmented characters obtained from Part 2. Below shows an example of our prediction.

Final Prediction:
 
Fig 10 
Future Scope

●	In the next three months we would like to improve the accuracy of our handwritten words recognition model by using keras-ocr to train a new, custom model.
●	To improve our handwriting recognition accuracy, we will utilize Long Short-Term Memory networks (LSTMs), which can naturally handle connected characters. 
●	We will also add a third level of complexity where we will implement a model that is able to recognize words in handwritten paragraphs. 
●	The recommended approach to be used will be the Single Shot multibox Detection(SSD) Architecture. The input for the model will be an image of a passage that contains handwritten text, while the outputs will be bounding boxes corresponding to each line of the text. The SSD architecture will be used to detect the positions of each line of text. The architecture does this by taking the image features and repeatedly downscaling them to account for different scaling factors, then feeding the features into two CNNs: one to approximate the locations of bounding boxes relative to anchor points, and the other to approximate the probability of the bounding box encompassing the object.
