# Using ML advancement in CVM classification

## Introduction 

In this blog, I revisited an old project to assess my progress and leverage my newfound knowledge. The project involves classifying 6-class X-ray images using publicly available datasets. While I won’t use or point to the dataset, I’ll share my insights and progress.

When I first embarked on this project, my ML knowledge was limited, and I expected immediate results. However, I’ve come to realize that this project wasn’t underachieving; it was a learning experience. Now, with more resources and ideas, I’m excited to delve into what I’ve accomplished and what I can achieve in the future.

### Outline

The project should be an end-to-end model, meaning it should be trained at every step to be usable when raw input is provided. Therefore, I’ve divided the model into two parts:

1. Detection: My goal is to detect regions of interest (RoIs) in given X-ray images using standard processing techniques and cropping them.

2. Classification: I’ve designed a machine learning model with custom modes that I’ll describe in detail later.

## What is CVM classification? 

Before delving into the models and methodology, it’s crucial to provide an overview of the task and the data. CVM stage classification is a technique used to assess the bone age of a patient, which can have direct implications for treatment decisions. Each patient undergoes X-ray imaging, and dental experts evaluate the stage before choosing a treatment. Given that this is a standard procedure, automating it using a machine learning model appears promising. To achieve this, we began by constructing the dataset before proceeding with the modeling process. 

### Dataset

The images are sourced from a publicly available dataset and are all grayscale. Each image in the dataset depicts a person’s head along with the initial few cervical vertebrae. Based on the shape and size of the vertebrae, the images are categorized into six distinct development stages, representing different growth stages. Consequently, the region of interest is the vertebrae area, in which the rest can be treated as noise. Although the website presents all its images in a fixed size, the object (person’s head) within the frame does not always maintain the same scale. This issue prompted me to employ a detection model before classification. 

Here you can see an example from the dataset; a lateral cephalogram of a patient. 

To accurately capture the tasks, I should also describe the classes and their differences. Since this is a growth stage determination problem, we are not identifying a specific object in the images but rather the shape of the object. Therefore, the stages are formed based on certain characteristics of the vertebrae. There are six stages: CS1, CS2, CS3, CS4, CS5, and CS6. These stages are also ordinal classes, with CS1 being the most immature and CS6 being the most mature. Instead of simply stating the class differences, I thought it would be more beneficial to illustrate them. Here’s a model image that demonstrates the difference between the stages, taken from []:


As evident from the provided data, the distinct shapes and sizes of the three vertebrae serve as crucial distinguishing factors that separate the classes. Consequently, the model designed to address this task should be structured accordingly.

The dataset comprises 999 images, with 710 allocated for training, 96 for validation, and the remaining 123 reserved for testing. To assess the model’s performance, we evaluate its accuracy using precision, recall, and F1 score, in addition to the testing accuracy.

## Object Detection

The initial stage of the model, as previously described, involves object detection. I believed this step was crucial because the scale of the vertebrae compared to the rest of the lateral cephalogram is relatively small, and the model might be misled by noise. To identify the cervical vertebrae region, I trained a YOLOv8 model. Subsequently, the region of interest (RoI) is detected using the YOLOv8 model. Before proceeding to the second stage, the RoI is cropped and resized. Here’s an overview of the object detection phase. 

## Image Classification

The second stage of the model is more challenging than the first. The reason for this is the image quality and the similarity of the growth stages. Although the image resolution is sufficient for training, the variety in the images can cause problems.

In tasks where images are divided into ordinal classes, the quality of the context makes a significant difference. For a model to capture features effectively, everything should be standardized before training. Additionally, it should be done well to avoid any outliers confusing the model. 

### Data Preparation

#### Resizing
To achieve this, I resized the cropped region to 256x256 while preserving its aspect ratio. This involves determining the appropriate axis for resizing, calculating the ratio, resizing the image, and filling in the remaining space. An example of this resizing process is provided below.

#### Data Augmentation
Since the dataset might lack sufficient images to train a large network, I employed data augmentation techniques to mitigate overfitting. However, these techniques shouldn’t introduce errors in classification by altering the context. Consequently, I limited my use to general methods and some techniques that improve the quality of grayscale images, such as CLAHE. Below are some examples of data augmentation results.

### Model Design 
Here comes the crucial part. The design of the model is the most significant aspect of any task, in my opinion. Given its importance, it becomes challenging to achieve perfection. It is essential to comprehend the details of the task, its requirements, and its response. Considering these factors, I chose to begin with a pretrained network, such as ResNet18. This approach is straightforward and an established network that provides valuable insights into potential improvements. However, my initial experiments yielded poor performance.

As I increased the network’s capacity using EfficientNet-B0, I observed improved results, achieving approximately 45% test accuracy on six classes.

Despite further capacity increases, the performance remained stagnant. Therefore, I decided to adopt a more sophisticated and tailored solution by incorporating CBAM into the network. This modification resulted in a modest performance enhancement of around 5%. This idea prompted me to explore various mechanisms, including tailored pretraining and semi-supervised training. However, none of these approaches yielded significant improvements compared to CBAM, leading me to discard them.

One idea persisted that eluded me: mask-guided training. As you may notice, the classification task only requires the top three vertebrae. However, some images in the dataset contain more than four or five vertebrae, which I later discovered caused the model to focus on itself, leading to misclassifications. To address this issue, I created masks for 36 images from six classes and trained the model using these masks. The intention was to guide the model’s attention to specific regions. However, the performance of the model trained with these masks was even poorer than before, prompting me to abandon this approach as well.

Despite these setbacks, I believe there are still avenues for exploration. 

As I experimented with various design choices for the model, I also analyzed its performance using different metrics and figures. I also captured the model’s performance on top-2 predictions, which means if the model correctly predicts the class in its top 2 predictions. Surprisingly, the top-2 prediction accuracy for the testing set was around 80%. This indicates that the model may be overconfident in its predictions, suggesting that post-processing could be beneficial. However, my attempts to implement a refinement layer using a simple MLP or Logistic Regression that takes probability values of each class and entropy as inputs failed. Therefore, I abandoned the post-processing approach but plan to revisit it later.

Finally, I considered an ensemble model that would involve voting on the classification. I employed k-fold cross-validation to train the model on five different subsets of the training set. I then used the models trained on different folds to process the test images and generate logits. Finally, I combined the logits to make the final decision, which resulted in an enhanced performance of 60%. I have several other ideas, but I decided to complete this project for now to revisit it later. Sometimes, it’s best to leave a project aside for a while and come back to it with fresh eyes.

### Results

Here you can see some of the results I have.
