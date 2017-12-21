# skinlesionclassification
In this project, we complete an image classification task that involves three unique diagnoses of skin lesions (melanoma, nevus, and seborrheic keratosis). In this task, we distinguish between (a) melanoma and (b) nevus and seborrheic keratosis. First, we describe the results obtained from training on a pre-trained ResNet architecture without any modifications. This serves as a baseline. Then, we add certain modification to this basic architecture: normalizing color using color constancy, and data augmentation using random crop and random flips. We also try to incorporate the segmentation information in our model. To calculate the accuracy on the test set, we use two versions of ensemble learning that includes a combination of ResNet and VGGNet models.   
