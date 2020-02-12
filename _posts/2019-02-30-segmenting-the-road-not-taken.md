---
permalink: /segmenting-the-road-not-taken/
title: "(Segmenting) The Road Not Taken"
type: posts
layout: single
author_profile: true
reading_time: true
comments: true
share: true
excerpt: "Deep Learning models for segmenting satellite road and highway images."

---

## Description

This work follows the development of our ability to implement complex algorithms on a high dimensional data set.  Our final model produced a dice coefficient of over 0.76 in the task of extracting roads from satellite images, placing among the top 10 teams in the class, and highlights our progression of knowledge in how to approach prediction problems in the field of machine learning. We began early in the project phase by implementing a naive K-nearest-neighbors logistic regression, based on results from our clustering algorithm. As we learned more about deep networks such as convolutional neural networks, we implemented a very basic U-Net, which outperformed the logistic regression by a great deal. We improved upon this classic architecture by adding various encoders to the U-Net's contraction path. We determined which methods worked best and, using those encoders, we were able to create a final ensemble of two models that noticeably improved the prediction accuracy of the original U-Net. 

## Exploratory Data Analysis and Initial Modeling Attempts

The goal of this project was to segment and extract road pixels from 512\(\times\)512 RGB satellite aerial images. We were given approximately 10,000 training images to train on and 2,170 test images for which our predictions will be evaluated. At first look, there was a large variety of images which ranged from highly urban (many buildings and roads), to mostly rural (many fields and not many buildings), to a mix of urban/rural areas. It was also clear that this was an unbalanced classification problem: 96\% of pixels were non-road, and only 4\% were roads. 

Neither of us had ever worked with image data, nor tackled an image segmentation problem.  However, we were quickly able to discern that image data can be interpreted as a feature matrix  based on color channels, and that image segmentation is simply a classification problem. Though the formulation appeared simple, the sheer size of the training data set coupled with our desire to produce a robust model for classification presented a difficult challenge.  

We understood that the best model would have to be trained using the most information that could be extracted on our collection of training images, but we realized that any common practice machine learning technique conducted on these data would be neither computationally efficient nor cheap. In order to train a model, there would be millions of parameters to store, along with millions to billions of computations to conduct. So, in our initial modeling attempts, we attempted to create a delicate balance between leveraging all of the information present in the data set with producing a model that was feasible to train given the time constraints of a semester project.

## Clustering: Combining Intuition and Unsupervised Learning Techniques

The obvious way to exploit the information contained in the training images was to cluster.  We felt that if we could effectively group the images, it would be possible to generalize the images by cluster, thereby reducing the scale of our training set. With clearly defined clusters, it might be possible to train models on a few images from each cluster,and only loose a marginal amount of information.

Intuition played a large role in coming up with an effective clustering scheme.  We realized that in the problem at hand, we were attempting to create a pixel by pixel mapping given a training image to its hand drawn mask:
\[f: (R,G,B) \rightarrow [0,1]\]

Now, there were certainly images in the training set with more road pixels in their masks than others, and equivalently, images with fewer road pixels.  There were also a fair share of images with a moderate number of roads (the hardest group to classify).  We could also easily quantify road concentration by simply computing the sum of the pixel values in an image's mask. For instance, images with many roads would (hopefully) have masks with many more 1's. So, we employed a three-Means clustering algorithm on each training images' mask sum, with the hope that the clusters would highlight 1) urban areas (images with generally more roads) 2) rural areas (images with fewer roads) 3) the pesky in between road level. 

The results were consistent, though not uniform, with this hypothesis. The images below show 5 randomly sampled images from each cluster. 

Satisfied with this clustering, the next step was an attempt at leveraging the information in each cluster to produce a prediction.

## Logistic Regression and the Limitations of Feature Engineering

At that point in the semester, we had not been exposed to deep learning, so at our disposal we had relatively weaker tools for classification.  We decided to use logistic regression given our wealth of experience implementing this technique.  In order to incorporate the information from our clusters, we devised the following algorithm for prediction on a new image:

Our intuition behind this algorithm was simple: fit a few models within each cluster to capture in-group variability, and then use a similarity metric to determine which model would be best to use to predict on an unseen image.  

The hyperparameters governing this algorithm, like the number of images to sample within each cluster and the degree of the polynomial basis expansion of the feature matrix were quite arbitrary.  This was really just a first effort to produce a prediction, and the results speak to that (which can be seen in Figure 7).  However, we uncovered some important information as to why a method like single layer logistic regression would never produce good predictions given the problem. Below is an example of a prediction produced by this algorithm:

As you can see, once the model for the "closest" image is chosen, and predictions are made, they are entirely based on the RGB channels of individual pixels.  Areas that have colors similar to the roads in the picture, which in this example are beige (for instance the model classifies the field from (([300,500],[100,200]) as one big road), are classified as roads even though they are clearly not.  

Therefore, this type of classifier only makes its predictions based on pixel colors, which will not generalize given the varying colors of roads in the images. In fact, we realized that no amount of feature engineering on the RGB values of a given pixel would produce models that would predict well. The classification of a pixel as a road or not is a far more sophisticated problem than identifying connections between color channels.

## Progression to Convolutional Neural Networks: the U-Net

As we could not rely on manually engineering a basis to map the information available in an image to its corresponding mask, we turned to deep neural networks to better approximate this function. Further research on effective network architectures highlighted the U-Net, a convolutional neural network, as the best performing network structure for the task of image segmentation \cite{unet}.

This network essentially has two stages. First a series of 3x3 convolutions followed by non-linear activation functions and pooling contracts the image with the goal of feature extraction.  In each subsequent layer of the "downward" pass, the number of feature maps is doubled while the image is shrunken.  Once a sufficient number of feature maps have been created, the image is then expanded through 2$\times$2 up convolutions and parameter concatenation with corresponding contraction layers until it reaches its original input size. The output is then passed through a fully connected layer with a sigmoid activation to create a segmentation map.    

Since this method has been well explored, implementing this architecture was not difficult. What proved to be the most challenging part of creating the best model was hyper-parameter tuning. There are numerous design decisions that need be made when constructing such a network. Developing intuition about which ones will result in the model with the highest predictive power is quite difficult given the hundreds of millions of parameters present in the model, the time required for training, and our limited knowledge of the functionality of convolutional neural networks. 

## Building up to a Successful Model

For our first convolutional neural network, we adapted the U-Net tutorial code written by Marko Jocic for the 2015 Ultrasound Nerve Segmentation Kaggle Competition. Throughout our project, we used a combination of AWS p2.xlarge EC2 instances and a Microsoft Azure Data Science VM with a Tesla K80 GPU (while we had issues getting our AWS EC2 instances to use the built-in GPU). The first and biggest problem we were confronted with was the long training time, given the size of the images and the complexity of the U-Net. To get around this, we downsized the original 512$\times$512 images to 128$\times$128, and reduced the number of filters in each layer. Using Keras, this gave us reasonable training times and allowed us to monitor the performance of the network in real-time. 

We used this trial time to do some preliminary tuning, and established an optimizer (Adam) with a reasonable learning rate of 1e-4, ReLU activation functions following each convolution layer, batch size of 8, layer widths, and a working loss function. Since the competition metric we were trying to maximize was the dice coefficient, we initially chose the negative dice coefficient (equation \ref{eq:1}) as our loss function. 

\begin{equation}\label{eq:1}
Dice =\frac{2\,\, |X\cap Y|}{|X| \,+ \,|Y|}\\    
\end{equation} 

However, this loss function seemed to be only training until the validation and training dice were capped at around 0.06 (all the predictions were the same), so we switched to using the soft dice coefficient (equation \ref{eq:2}) as the loss function for much of the rest of our training. This also solved differentiability issues that the original dice coefficient was known to suffer from. We further elaborate on hyperparameter tuning in Section 3.1. 


\begin{equation}\label{eq:2}
Soft\,Dice =\frac{2\,\,|X\cap Y|+1}{|X|+|Y|+1}
\end{equation}

After we were confident in our model, we began using the training images with their original sizes and increased the number of filters in each layer to accommodate the larger image sizes. We thought that a wider network would produce more feature maps, and therefore increase the predictive power of the model. We were able to achieve a baseline of $\sim$0.65 dice with this model, without post-processing. 

## Fine Tuning the U-Net: Hyperparameters and Encoders

After identifying an existing architecture that had achieved comparative success in the image segmentation problem, we were eager to test its ability on our own data set and hoped to find a variation that would produce great results.  

We did not anticipate the challenges that arose out of the seemingly simple task of hyperparameter tuning.  We realized that in the training of such high dimensional models, it is difficult to develop intuition or a mathematical explanation on how altering a non-learned parameter can affect the model's performance. In fact, we will be the first ones to admit that our model's success was at least partially due to luck.

###  Hyperparameter Selection

The outline of the structure of the U-Net presented in Figure 6 merely served as a guide as we searched for the best U-Net. We had to make several decisions educated by model performance that allowed us to find the one that ultimately did the best in prediction.

We highlight a few of the most important hyperparameters that we considered during our model selection:


\begin{enumerate}
\item \textbf{Network Depth and Width:} 

We experimented with different initial numbers of filters (width), which would be subsequently doubled at each level of the downward pass, and then expanded by the same scale during the image expansion. In addition, we tried to extend the network by adding an additional contraction layer (this is our Extended U-Net, see Fig. 7), thus creating another expansion layer (depth). We compared models with initial width of 16, 32 and 64 filters each with and without the extra contraction layer. Generally, we saw accuracy increase as we added network width, which created more feature maps. Adding the extra level only supplied marginal improvement while increasing computation time significantly. We primarily stuck with 3$\times$3 convolution layers as they allowed us to capture features with finer details. 

\item \textbf{Loss Function}

The network weights are trained according to a specified loss function that drives gradient descent. As such, choosing the loss function governing our model was an important task. 

Given that our final model would be judged based on the Dice Coefficient, it made sense to train models according to the negative soft dice coefficient, since this would optimize models to perform well at classifying according to the dice coefficient.  We also trained models using binary cross entropy.  In the end, once we introduced the pre-trained Res-Net encoder, we found that a weighted binary cross entropy loss and Jaccard Index produced our best model. Because the Jaccard Index is directly derived from the dice coefficient, we thought they would perform similarly. Since the dice coefficient generally did not have a well-behaved gradient, we were more inclined to use the Jaccard Index at the end. The Jaccard Index also is said to perform well when classes are inbalanced (as in our case) \cite{jaccard}. 

\item \textbf{Learning Rate}

For such a large data set, we were fairly set on using the Adam optimizer, but we tested varying learning rates, ranging from 1e-1 to 1e-5. With so many parameters in the model, it becomes important to choose a rate which supports sustained learning, as reflected through continued progress in lowering loss throughout the training process. In the end a learning rate of 1e-4 seemed to perform best for most of our models.

\item \textbf{Activation Function}

We only considered the ReLU and ELU activation functions and compared model performance for each. This was done to eliminate the problem of gradient saturation and encourage weight updates throughout the training process. Though the ReLU is known to suffer from the "vanishing gradient" problem, it did not seem to have a huge impact on our final results so we went with the ReLU for our final model. 

\item \textbf{Batch Size}

We tinkered with the number of images passed through each learning step in training.  Since weights are updated based on the images in a given batch, it is important to tune the optimal number of images in each batch to facilitate consistent learning. We considered batch sizes of 2, 4, 6, 8, 16 and 32 in our models. Often, batch sizes of 32, 16, and sometimes even 8 gave us memory errors, so for our final two models we used batch sizes of 4 and 8. 

\item \textbf{Dropout}

We played around with dropout layers at several points in the U-Net, but did not find that it improved our models enough to tune this parameter. We did not include any dropout layers in our final models. We did run into a slight problem with over-fitting with the ResNet encoders, as training accuracy would often surpass a dice coefficient of 0.8 with test accuracy slightly lower. Given more time, we would consider utilizing dropout in these models to introduce some regularization to the model.

\end{enumerate}

The actual act of tuning these hyper parameters was perhaps the most difficult part of our project.  For one, training a single model takes hours, so it was unfeasible to perform a grid search over the aforementioned hyper parameter space with cross-validation. 

To counter this, we had the idea to subsample the data into a representative subset using the clusters from our exploratory analysis. We thought that if we could reduce the training set while still making it representative of the whole sample, we could then cut down training time significantly and hopefully conduct a grid search. We randomly sampled 20 images total from the clusters to train and 20 to validate, with the division of "urban", "rural", and "mixed" images proportional to their representation in the training set. We would then train models using this data for one epoch, validate, and select the best one. 

This approach proved ineffective since training models for only one epoch did not provide means for comparing models with different hyperparameters.  The loss and validation accuracy after such a short period of training, regardless of the amount of data, was too low to see any marked improvement of one model over the other. 

As such, we had to make what were essentially educated guesses in our hyper parameter tuning.  Our project at this stage turned into a comparison of different U-Net architectures based on the aforementioned hyperparameters, and our best pure U-Net after post-processing (with a dice of over 0.68) was achieved by training a model for about 15 epochs with a ReLU activation, 5 contraction layers, an initial width of 64 filters (which doubled each successive layer), an 8 image batch size, and a learning rate of 1e-4 using the Adam Optimizer. 

## Methods for Model Assessment

Given a selection of hyper parameters, we would initially split off $10 \%$ of the training data as a validation set and train the model with the remaining portion. After we were confident in the performance of a model, we would reduce this number to $8 \%$ so as to allocate as much data as possible for training yet still maintaining a small set for monitoring validation metrics. We would typically allow the model to train for 5 epochs, saving the learned model weights if the model loss improved.  If validation accuracy continued to increase after the 5th epoch, this would signify a potentially good combination of hyper parameters, and we would use the saved weights to train a model for 5 more epochs, repeating the previous process. 

We were able to monitor and assess over-fitting by observing the Tensorboard after each epoch, which would report the model's
training and validation accuracy and loss. We show a plot of these metrics in Section 4.3. Furthermore, we used Kera's EarlyStopping checkpoint feature, which allows us to monitor the validation loss over time, and stopped training once this metric stops improving. We set a patience of two epochs, which allows the model to continue training for two epochs without validation loss improvement, preventing training from stopping too early. With these convenient feature, we were able to stop training whenever we felt there was over fitting, and monitor the convergence of the model as measured by the change to the loss function. If we had a good combination of hyperparameters based on these metrics, it would typically take 15-20 epochs to achieve convergence. 

### Data Augmentation

After exploring many combinations of hyper parameters when training, and finding our "best" combination, we looked for other ways to increase the power of our model. We thought that implementing data augmentation on our training data could be a method that could increase robustness.

In our training data generation function (using Keras' ImageDataGenerator), we applied a random transformation that could consist of a flip, rotation, translation, scale change, or ZCA whitening to each image and corresponding mask in the training set. We set a seed so that the random transformations would be uniform among all training instances/epochs. The hope was that perturbations of the images would increase the model's ability to pick up on minute features in the data, thus increasing accuracy.

In our final model (the U-Net-ResNet34), we observed that data augmentation improved our validation dice coefficient by approximately 0.01, which is not trivial.

### Introducing Pre-trained Encoders to the U-Net

The performance of our best model so far capped at around a 0.68 modified dice coefficient after post-processing. In order to increase this, we either had to improve the U-Net somehow or introduce new architectures. We looked up similar competitions in Kaggle (specifically the 2018 TGS Salt Identification Challenge \cite{salt} and the 2015 Ultrasound Nerve Segmentation Competition \cite{nerve}) for inspiration. Though many approaches were unrealistic for us in terms of compute time or complexity, many top performing contestants reported good results by integrating pre-trained encoders with an existing popular framework such as the U-Net or LinkNet. We decided to try some of these approaches with our current U-Net. Originally, we had very limited knowledge as to what these encoders actually did besides replace the classic down-sampling path of the U-Net with a more suitable path that allowed the new algorithm to either train faster, generalize better, or extract more relevant features. Upon further research, we understood the encoders as a method for creating "skip connections" over dense, complex regions in the architecture which had the effect of creating a smoother loss function, void of some of the local minima that a high dimensional model was bound to create. Using trial-and-error, we experimented with a few of these encoders; a brief description of each is provided in the table below. 

For all these encoders, we used weights pre-trained on the 2012 ILSVRC ImageNet dataset \cite{imagenet_cvpr09}. We used the Python package $\mathtt{segmentation-models}$, a Python library with Neural Networks specifically for image segmentation, to introduce these encoders to our U-Net framework \cite{segmod}. Though the images from ImageNet (which are usually ordinary photos and portraits) are vastly different from our satellite images, our reasoning for using these weights was that they provided the algorithm with better-than-random initialization values, so that the model will hopefully eventually converge to better minima than if random initialization was used. 

It was clear after just a few steps within each epoch which encoders would perform the best. Notably, U-Net-ResNet34 ran extremely fast and the loss decreased quickly. In fact, the entire training process took only 5-6 hours with data augmentation and 15-20 epochs total on an AWS p2.xlarge instance, which was far faster than any model we had looked at so far. LinkNet architectures seemed to converge very slowly, so we abandoned this encoder from the get go. We believe the VGG-16 and ResNeXt50 encoders did not work as well as ResNet34 because they had a high number of parameters and no skip connections, so that overfitting was much more likely.  After running a U-Net with each respective encoder until the validation loss stopped improving, we were able to see the maximum capacity of each encoder-U-Net combination. A comparison of each model is below.


Clearly, U-Net-ResNet34 performed above and beyond all other combinations of architecture-encoders. Furthermore, this model trained very fast (around 15-30 minutes per epoch on a single Tesla K80 GPU, reaching convergence around 15 epochs) compared to the other models. Because of its high performance and speed of training, we chose the U-Net-ResNet34 as the final single model from which to ensemble and create our best set of predictions. This network was remarkably sparse, due to the skip connections (Figure 8) that allows the network to pass over unnecessary layers. This simultaneously solves the problems of too many parameters as well as the vanishing gradient issue (prevalent in networks of similar depth) as a result of having too many layers. 

For future work and further improvement of the model, other architectures that we had read about in literature, such as the Stacked U-Net and Fully Convolutional Networks, may train slower but perform better than the U-Net-ResNet34.  

## Model Selection and Post-processing

After trying out multiple combinations of hyper parameters, implementing data augmentation, and introducing a pre-trained Res-Net encoder to our model, our result after training was the same regardless of technique used: a sigmoid output for each pixel as an associated probability of whether or not the pixel is a road. Several post processing decisions were made at this point impacting the predictive accuracy of the model.

### Road Inclusion Probability and Minimum Mask Size

Given a 512$\times$512 matrix of probabilities from any one of the several models we ran, we had to decide how to then convert this matrix into a proper mask that would highlight roads.  We initially labeled any pixel with output probability greater than 50$\%$ as a road.   Under this regime, we were frustrated by our models' inaccuracies, but failed to realize that these could in part be because of a low threshold value.  

Looking at the images and corresponding masks in the training set, we realized that roads comprise a generally small proportion of the image. In fact, on average only 4\% of pixels were roads (Figure 9). As a result, we hypothesized that if we increased the probability threshold when converting output to mask, i.e. required the model to be more confident when making a road classification, our accuracy would increase.  

Based on the competition metric, we also introduced the idea of a minimum mask size as another post-processing step. The Dice Coefficient greatly penalizes false positives, or pixels classified by the model as roads that are not roads, when the true mask size is very low.  Once we had thresholded the output probabilities, we examined each prediction to see the number of encoded roads in the predicted mask.  Given that there are 512$\times$512 pixels in a mask, we decided that there should be base minimum number of pixels classified as roads in a mask for us to find a sweet spot with the competition metric.

To find the optimal pixel probability and minimum mask size, we ran a simple grid search using our best model, the U-Net with a pre-trained Res-Net encoder, on training data and their true masks.  We generated predictions for the first 3000 images using minimum mask sizes from 100 to 1000, and pixel probability cutoffs ranging from 0.3 to 0.9 in increments of 0.1. For each of the predictions, we compared the modified dice coefficient. The results are seen in Figure 10, and we found that a minimum mask size of 400 and inclusion probability of 0.8 produced the best dice coefficient. We used these thresholds in predictions for single models. We applied this technique to ensembles of models as well, though because the variance of an ensemble of models is lower in general, we used a probability cutoff of 0.5 and a minimum mask size of 300 for that case specifically.

### Prediction Post-processing

To further refine the predictions, we also tried out different morphological transformations, which are essentially different types of black-and-white image post-processing methods. The different transformations we tried out on the final masks included erosion (removes boundaries of the foreground object, in our case, edges of roads), dilation (increases the pixel area occupied by roads by "enlarging" road boundaries), opening (erosion followed by dilation, useful for removing noise - i.e. small specks of random road), and closing (dilation followed by erosion, useful for removing non-road pixels in large patches of road), all from the OpenCV python package $\mathtt{cv2}$ \cite{morph}. 

Following our experimentation with these methods, we saw that both erosion and dilation with reasonable kernels separately affected the final masks too much, so we stuck with just an Opening step (Figure 11) to simply remove small pixels of road in the masks. This did not affect our final score very much, but it would be interesting to play around with image morphology more in the future, and tailor a kernel specifically for our use case. 

## Final Model

Below is a comparison of a prediction from the U-Net with a pre-trained ResNet34 encoder with predictions from the classic U-Net, our very first model. U-Net-ResNet34 is better able to capture nuances in the image data, close road segments more successfully, and ignore random noise, when compared to the U-Net. 
    
It is clear for some images, that the U-Net-ResNet34 predicts very well. This applies mostly to images with very obvious roads, of a single color, with clear edges (no shrubs or greenery obstructing the road's edges). Some cases in which our model did not do very well include narrow dirt roads, parts of images where we humans could not even confidently classify as being a road or not, and field lines in rural areas that were mistaken to be roads due to their structure. We also noticed that the algorithm had problems "closing" roads by connecting them together; this problem was more evident in areas of high classification uncertainty. 

We also include a plot of training and validation loss and dice coefficient over time (epochs) for the U-Net-ResNet34 with data augmentation. The training loss and dice coefficient continue to decrease, but validation loss and dice appear to have stagnated. It's clear that the algorithm is learning a great deal about the training set, but that learning is not translating very well past the first epoch to new images. Perhaps having learning that generalizes better to new images is not possible (or very difficult), but given more time and computing resources we would have experimented with different architectures.

As a final step to improve prediction accuracy, we ensembled two models by averaging their sigmoid outputs together. Specifically, we trained one U-Net-ResNet34 without data augmentation, saved the results, trained a second U-Net-ResNet34 (this time WITH 5000 data-augmented images added manually to the training set), and saved the second set of results. We then averaged the resulting probabilities together. Though the model with data augmentation clearly performs better, ensembling it with the model without data augmentation reduces the overall variance of the sigmoid probabilities and thus gives higher accuracy. 

## Conclusion

This project represents our gradual improvement in understanding machine learning methods both simple and complex as applied to an image segmentation problem. At the beginning of the project, when we had minimal knowledge of convolutional neural networks, we applied what we did know of machine learning (namely logistic regression and K-means clustering) to the problem and achieved nontrivial, though not very accurate, results. 

As we learned more about CNNs, we were able to implement a basic deep network called the U-Net with very promising results. We gradually improved the algorithm through multiple small steps and finally achieved an improvement of over 10\% accuracy over our original U-Net. 

Our final model utilizes a self-tuned U-Net-ResNet34, and achieved a modified dice coefficient of over 0.76 on both the public and private leader boards. We placed 8th on the public Kaggle board, and 10th on the private board. We have learned a lot through this project, mostly through both trial and error and connecting what we learned in class to new methods discovered through research on similar problems. 

Though our model exceeds the performance of a simple U-Net, there is ample room for improvement. Future work would include ensembling (averaging the sigmoid output from multiple different models, and not our "pseudo"-ensembling of two essentially identical models). We were not able to train a model with an entirely different architecture than the U-Net-ResNet34 and still produce a comparable validation accuracy. Indeed, our second-best performing model was the Extended U-Net which capped at a much lower 0.68 dice. We were worried about ensembling our U-Net-ResNet34 with a model that performed much worse, thinking it would lower accuracy overall. 

We also would have liked to experiment with additional image morphology, in particular designing kernels that could be used for our specific application. The Opening operation that we had applied was limited to single, isolated pixels and would not catch, for example, an island of two neighboring pixels. We also would have liked to spend time exploring other architectures such as the LinkNet, PSPNet, etc., in depth, and adding a decaying learning rate to allow for a coarse-to-fine search in parameter weights. Finally, though we had experimented with test time augmentation (TTA), it ultimately failed and produced worse results than originally. We coded our own TTA, which involved 3 basic transformations (left/right flip, up/down flip, and a combined flip) then averaged the results after de-transformation. Perhaps doing TTA from an existing Python package would have produced better results. 

In conclusion, considering that at the beginning of this project we had little to no knowledge of deep neural networks or how to even tackle such a complex problem, we were able to develop and tune a model that not only adequately extracted roads from satellite images, but also greatly outperformed a basic U-Net. Now, we are finally able to relate to the saying that "training neural networks is hard."
