# Detection of Unhealthy Brain Disease using MRI Images

**ABBREVIATION**

**ANN**: Artificial Neural Network

**CNN**: Convolutional Neural Network

**FNN**: Feedforward Neural Network

**GPU:** Graphics Processing Unit

**NN:** Neural Network

**MRI:**: Magnetic Resonance Image

**WT:** Whole Tumor

**TC:** Tumor Core

**ET:** Enhancing Tumor

**HGG:** High Grade Glioma

**LGG:** Low Grade Glioma

# ABSTRACT

Analysis of the 3d volume of brain MRI of the patient is a time taking task for doctors. Deep Learning algorithms help doctors for investing interesting parts timely and then finalize the results more efficiently. Brain tumor segmentation separates the cancerous part of the brain from the normal brain. Brain tumor segmentation is classified into four parts such as preprocessing, segmentation, optimization, and feature extraction. The importance of MRIbased segmentation expanded in recent years. MRI is the most reliable, safe, and has good resolution. It has no side effects, no radiation, and not harmful to other parts of the body. The tumor is segmented after MRI is processed. In this report, we proposed an automatic segmentation method based on CNN. It includes many layers for feature extraction and brain tumor segmentation. For this, the BraTS 2018 dataset consists of 210 HGG (High-Grade Glioma) images and 75 LGG (Low-Grade Glioma).

In our work, we have not only detected the brain tumor but we also segmented that where is tumor in the Human Brain. The task of detecting the position of the tumor in the body of the patient is the starting point for medical treatment. Brain tumor detection in an early stage can help to reduce the death rate in the medical field. The most common brain tumor is gliomas. It is categorized as HGG (high-grade glioma) and LGG (low-grade glioma). By the use of MRI, we get information on gliomas. For sub-regions and details of a tumor, the MRI has the following sequence such as T1-weighted image, T1-weighted with gadolinium contrast enhancement (t1gd), T2-weighted image, and fluid-attenuated inversion recovery- (FLAIR) weighted image.

### Magnetic Resonance Imaging (MRI) for Brain

Magnetic resonance imaging is most beneficial and preferable technique used in the medical field to produce images of different parts of the human body i.e., brain. It is supported the principles of nuclear magnetic resonance. It is widely used in the medical field. MRI is the most reliable, safe and has good resolution. We prefer MRI because the brain has soft organs and tissues, to get detailed information of human brain. MRI has no side effects even with repeated exposure with magnetic field and has good resolution. The brain MRIs are shown:

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211404435-3e6334fd-1bf1-4cb1-94f0-64f70b3869a9.png">
</p>

**MRI Sequences**

MRI is a sequence of events that happen in inside the MRI machine gives you images. Different magnetic resonance imaging (MRI) sequence images are used for diagnosis, including T1-weighted MRI, T2-weighted MRI, T1-weighted with gadolinium contrast enhancement (t1gd), and fluid-attenuated inversion recovery- (FLAIR) weighted MRI.

MRI images are categorized in following sequence:

1. **T1-weighted image**

In MRI, T1-weighted image provides low signal to specified lesion area(edema) of brain. Because, water and collagenous tissues have high protein. The T1-weighted images provide better anatomical details than other images. T1-weighted MRI image shows that fat is bright, water is dark and new blood vessels are also bright. It helps for observing the vascular changes and useful for disruption of blood-brain barrier (BBB) in contrast.

2. **T2-weighted image**

In MRI, T2-weighted image provides high signal to water. In T2-weighted image, CSF shows high intensity signal. T2-weighted MRI image shows that fat is dark, water is bright and new blood vessels are also dark. With the help of this sequence, we differentiate the brain lesions from normal brain than T1-weighted image but we cannot distinguish the lesion from CSF because it is also bright.

3. **FLAIR image**

It is similar to T2-weighted image but CSF is in dark area. We can improve gray-white differentiation. With the help of Fluid-attenuated inversion recovery (FLAIR), we find brain lesions (edema) that cannot be seen in T2-weighted sequence. It helps to differentiate the edema from hyper-intense area. Fluid image is mostly used in medical field. All the sequence of MRI are shown in following Figure.

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211404865-b7181d8c-b500-440f-b317-fa78d950fd28.png">
</p>


<p align=center> a) T1-w       b) T2-w     c) t1gd     d) Flair </p>

**Plane views of MRI**

Magnetic resonance imaging is comparatively defined in plane. The position of the human is described in the planes like a Cartesian coordinates system. In plane system the MRI is classified into the axial, sagittal, and coronal planes. The axial plane means top view, sagittal means side view, and coronal means front view of the body. With the help of these plane, we evaluate and inspect the disease to some extent. The basic plane view of MRI of the human body is categorized in sagittal plane, coronal plane, and transverse plan and shown in Figure.

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211403989-7eef0adb-5587-4634-b658-3e3739505a05.png">
</p>

1. **Sagittal plane**
Sagittal plane is sideway plane, from top to down, which separates left from right.
The sagittal plane gives the side view of the body. You can view either left or right side of human body.

2. **Coronal plane**
    Coronal plane gives the font view of the body plane, from front to back, which separates the anterior from the posterior.

3. **Transverse plane**
  Transverse plane, is an x-y-z plane, parallel to the ground, which separates the superior from the inferior, or put another way, the head from the feet.

The MRI head scans in axial, coronal, and sagittal planes shown in the Figure:

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211404146-4afe64d1-42cf-4ef9-9842-507c615caa97.png">
</p>

<p align="center"> Figure: Brain MRI different planes. Left to right: Transverse, Sagittal and Coronal.</p>

## BraTS dataset

We use BraTS 2018 data which consists of 210 HGG (High Grade Glioma) images and 75 LGG (Low Grade Glioma) along with survival dataset for 163 patients. We use only HGG images 180 for training and 30 for testing. MRI images are categorized in following sequence i.e., T1-weighted image, T1-weighted with gadolinium contrast enhancement (t1gd), T2weighted image, and fluid-attenuated inversion recovery- (FLAIR) weighted image. Each image is a 3D image with size (240,240,155).

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/211404267-fa92af86-61d7-4dc6-a724-9b9594edb96d.png">
</p>

# BRAIN TUMOR SEGMENTATION

**Image Segmentation**

To properly localize the image features at pixel level and partition the image into segments based on same pixel intensities assigned one class label and other pixel are assigned another class label. Pixel level segmentation helps to localize the interested part more clearly. Segmentation task is more difficult than the classification and detection. In detection, only the bounding box represents class. Bounding boxes can have different pixels other than the interested class but in case of segmentation the outline around the segmented part contains only the interested region as shown in Figure.


<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/211405660-f84d2060-43f1-4c68-bfa1-3a33087413de.png">
</p>

In this binary image segmentation Label 1 is mapped to interested region and the label 0 is as a background. This is an easy segmentation task as compared to multiclass segmentation where more classes are classified

Types of Segmentation

There are two types of segmentation which are mostly used. Segmentation of image is done by different architectures. We will discuss shortly two types of segmentation, semantic segmentation and instance segmentation.

**Semantic Segmentation**

In semantic segmentation, same type of objects assigned one class label. In this segmentation task, Higher level of image understanding is required. We need such algorithms which performs well at pixel level and can create boundary on the bases of different intensities, context, texture etc. We have done semantic segmentation. We have four different classes of tumor to segment as shown in Figure.

<p align="center">
<img width="" height="" src="https://user-images.githubusercontent.com/73955220/211405897-a3fd7593-4325-4347-bd31-8141b7161607.png">
</p>

In figure above Brain Tumor segmentation. Every region with its unique color represents the type of tumor. Labeling with numbers 1,2, and 3 is also assigned to each class of tumor. Label 4 is assigned as background with non-tumorous region.

This image helps doctors to analyze the image more deeply and focused on more dangerous parts.Same class of objects are segmented with same color. Person is segmented by the red color, car is segmented by the blue color, buildings are segmented by the yellowish color, lights are segmented by another color.

**Instance Segmentation**

Instance segmentation is same as the semantic segmentation. But in this segmentation, the architecture will find out the instance with same class labels and separate them with different colors. This will help to analyze the image more concretely and accurately. To perform such tasks in real applications, the dataset with lot of variations of same label are required. This will also lead to more training time, More, deeper network, the more time it will take to learn the features and can lead to over fitting problem. We have not used this technique because our problem is restricted to semantic segmentation. Instance segmentation can only be done through large amount of data and we have not enough data to do.

# 2D vs 3D Brain tumor segmentation

We have briefly explained 2D and 3D brain tumor segmentation and why we used 2D approach to detect brain tumors and what are the issues while using 3D segmentation.

**2D Brain tumor segmentation**

In 2D brain tumor segmentation, each slice of brain is segmented separately. This will separate out the tumorous region from the non-tumorous region. 2D segmentation takes less time for training and results are very efficient in this case but this is not effective for real time application. Because this is laborious task to analyze each and every slice of brain image separately and then analyze and combine the whole volume. To find the tumorous part of the brain becomes impossible to analyze. Any mistake can lead to the most serious issues. Doctors will use their domain knowledge to analyze the brain image. In this case, 2D image segmentation is not good for practical applications as it can lead to many errors. But 2D approach is easy to implement and one can combine different axial views to see the whole brain tumorous region.

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211406183-18140358-8226-4cb9-8407-fdc93c9fc892.png">
</p>

<p align=center>Figure: Flair modality 75th 2D slice is used here to segment</p>

In Figure above, On the very right side, there is segmented region which our architecture has done. In the middle, ground truth of the slice is given.

**3D Brain tumor segmentation**

In 3D brain tumor segmentation, not each slice is segmented separately as in case of 2D segmentation. Here the whole volume of brain is used as input to the model and model give the whole segmented volume as the output. This will help in dealing with the issues as we have faced in the above segmentation techniques. The whole volume can be observed simultaneously and correctly identified the image.

## Segmentation Architecture

There are multiple segmentation architectures which are used in deep learning. Most useful and recent architectures which works best for segmentation are described briefly. We have explained, why some architectures are superior than others and also explained that why we choose U-NET architecture for brain tumor segmentation.

**U-NET**

UNET architecture is named after its U like shape. This architecture is inspired to deal with specifically biomedical image segmentation. Because in medical field, Large amount of data is not available easily. In case of large amount of data, neural networks can be trained for months to extract the features.

Working with small dataset is not as easy as in large amount of data. After many researches, they designed an architecture which helps in working with small amount of data. U-NET Architecture is also encoding decoding path architecture like FCN. It has a contracting path which down-samples the image and extract the low- and high-level features of an image.

Contracting path consists of convolutional layers with 3\*3 filters, max-pooling layer and then non-linear activation function. Contracting path is all about to find the number of features. This grouped all features in one bottleneck. After this, expanding path with transpose convolution is used to locate the features with concatenation of same shape contracting layers with some crop. At the last layer, 1\*1 convolutions are used to segment the class.


<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211406458-fd93f59e-ad80-47f4-ab0e-958348dac356.png">
</p>

<p align=center> Figure: input image is gray-scale followed by 2 consecutive convolutions with 64 filters each. This then max-pooled to reduce the image size. Bottleneck with 1024 filters have all combined features. This is 'what' part of the image. After bottleneck, it has 'where' part of
the image which uses skip connections and concatenate to locate interested region.</p>

# Segmentation Losses

Loss function plays an important role in any machine learning model. Loss function tells how much the model is fitted to training data. We always prefer to have very lower value of loss. Zero value of loss indicates 100 % model accuracy on training data. There are different types of loss functions depend upon the model. Model parameters changes after every iteration depend upon the loss value. Segmentation losses are different than classification losses. Major semantic segmentation losses which help us in our model good or bad are given below:

**Dice Score Loss**

The Dice coefficient is widely used metric in computer vision community to calculate the similarity between two images. 𝑫𝑺𝑪 = 𝟐𝑻𝑷
<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211410866-3877c220-29c1-460a-8fab-642e845db220.png">
</p>

<p align=center>Figure: T_o_ is truth region where no tumor, T1 shows the ground truth region where tumor exists shown by blue color, Po is prediction where tumor not exists, P1 is the predicted tumor region shown by red color</p>

**Sensitivity Specificity Loss**

Similar to Dice Coefficient, Sensitivity and Specificity are widely used metrics to evaluate the segmentation predictions. In our application, dataset is very imbalanced. To overcome that issue, w is used to adjust loss to understand the model and make decisions accurately. TP is true positive and TN is true negative. The loss [40] is defined as:

**SSL = w × sensitivity + (1 − w) × specificity**

Where,

𝑻𝑷

![image](https://user-images.githubusercontent.com/73955220/211412005-cb060f7a-7a66-4896-a0a1-a92aaa2c9dc3.png)


𝑻𝑵

![image](https://user-images.githubusercontent.com/73955220/211412048-1ba944ee-bd41-42f0-bb6b-518800ee94a7.png)

## Proposed U-NET Model Architecture

We have used U-NET architecture for practical implementation. This architecture works best for the biomedical image segmentation.

# RESULTS

### Segmentation Results:

We achieved segmentation using UNET model as explained in chapter 4. We used Brats 2018 dataset consisting of 210 3D images. As we will slice, the 3D images and take images from 3 axis so the total number of 2D images come out to be 151200.The parameters and model for all them were kept the same due to its suitability with the dataset and we trained the data for a little bit more and achieved the following result:

TABLE: SEGMENTATION OF BRAIN TUMOR

| **Training and**  **Testing** | **Two Class**  **Segmentation** | **Four Class**  **Segmentation** | **Four Class Segmentation**  **Improved** |
| --- | --- | --- | --- |
| **Epochs** | 10 | 10 | 45 |
| **Training accuracy** | 0.9726 | 0.7526 | 0.7526 |
| **Training loss** | 0.0274 | 0.2474 | 0.2474 |
| **Testing accuracy** | 0.9222452600797018 | 0.6383521348237992 | 0.7266646613677342 |
| **Testing loss** | 0.07775473992029826 | 0.3616478661696116 | 0.27333533962567647 |
| **Mean Sensitivity for class 0** | 0.9987968804288612 | 0.9973130890962196 | 0.9994418608006598 |
| **Mean Specificity for class 0** | 0.8157543780263949 | 0.7399030542635091 | 0.7855767936044754 |
| **Mean Sensitivity for class 1** | 0.8157543780263949 | 0.3213817273098429 | 0.4228020048290975 |
| **Mean Specificity for class 1** | 0.9987968804288612 | 0.9994933975562182 | 0.9997933902524028 |
| **Mean Sensitivity for class 2** | NULL | 0.41349304918469726 | 0.6064759350159152 |
| **Mean Specificity for class 2** | NULL | 0.9996057752590369 | 0.9993690548406183 |
| **Mean Sensitivity for class 3** | NULL | 0.8404884744002085 | 0.826595230858187 |
| **Mean Specificity for class 3** | NULL | 0.9955309186827245 | 0.9985983158385012 |

#Segmentation Outputs of two classes:

**Graph Output:**

<p align="center">
  <img width="360" height="280" src="https://user-images.githubusercontent.com/73955220/211412615-6c6bfc78-c0f3-4ec1-935b-15537f1af555.png">
</p>

<p align=center> Figure Model accuracy vs epochs for 2 classes</p>

<p align="center">
  <img width="360" height="280" src="https://user-images.githubusercontent.com/73955220/211413049-21b4d170-7fa3-4e0e-80e0-4ad91136931d.png">
</p>

<p align=center>Figure: Model loss vs epochs for 2 classes</p>

<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211414102-a75020c5-e5a0-41b6-ae04-2b846e8163a1.png">
</p>
 
<p align="center">
  <img width="" height="" src="https://user-images.githubusercontent.com/73955220/211414296-84f506a1-9a88-40c7-a2ee-c0e924f09ff3.png">
</p> 

# Running the Code
To  run this code you will have to run the three axis files to train the model and save it on your pc or on drive if you are using google colab.
After training the three axis run the main file and give it the location of your three differnt axis models locarion.
We are training the three axis because of the three views we discussed above. And in each view we will slice the image into 2D images to give it to our model to train it.
The running of the code is very simple and the libraries used are commonly used libraries like Keras, Tensorflow, Nibabel, Panda etc.
You just have to give location of the dataset in the above files and just run the code.














