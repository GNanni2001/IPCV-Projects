# Authors

Gabriele Nanni (GNanni2001)
Davide Cremonini (Cremadvd)

# Assignment 1

In the first assignment we have to recognise and locate in the scenes offered the references.
All the scenes and references are into the dataset.zip folder.

Fist of all the noisy scenes have been processed using a pipeline of filters to obtain clearer images.

For this task it has been used a traditional image processing technique based on keypoint detection.
The main functions utilised are:
```getSliceFromPoly``` Given a list of points it will return the minimal and maximal coordinates to encompass the polyline.
```detect_keypoints``` Given a detector and a list of images, returns two lists, one containing all keypoints contained in each image and the other containing all descriptors for said keypoints.
```find_matches``` given a matcher, the number of images and references, the lists of descriptors for both images and references returns an array containing, for each reference a list of matches between its descriptors and the scene images's descriptors. Each match containt the two closest matches for each descriptor.
```filter_matches``` Given an 2D array containing lists of couples of matches (m,n), filters those matches by keeping only those where the ratio of the distance is less than a threshold.
```delete_kp_in_area``` Given two lists, one of keypoints and the other of their descriptors, and a patch expressed by an origin point and height and width, deletes from the lists all keypoints contained in the patch and their respective descriptors.
```get_instances``` Detects all instances inside one image and delivers them to the ```recognize_instances``` function.
```recognize_instances``` Performs instance recognition on all images, it takes as input four lists of images already denoised, two for scenes, two for references and specific parameters that need to be passed to the *get_instance* function. This function computes keypoints and descriptors for all images and invokes the *get_instance* function on all scene images, then printing the results.

# Assignment 2

In the second assignment we need to build and train a CNN to classify images based on the product represented.

When it was asked to build a network we decided to consider a basic block constituted by:

- 2D Convolutional Layer (padding=1, stride=1)
- Bacth Norm Layer
- Leaky ReLU activation

Then we defined a modular Neural Network with the following parameters:
- The *in_channels* parameter specifies the the number of channels defined of the input which will be converted to the base of the network by the first block.
- The *out_classes* parameter defines the number of logits that the classifier has to produce it output.
- The *conv_layers* parameter, which is a list of dictionaries describing the characteristic number of units in the block, kernel size, stride and padding for each layer inside it; it is used to instantiate each block by concatenating **ConvBnRelu** units, each disctionary is used for one block. As discussed before, the first element of the block doubles the number of channels, while the following ones maintain that number of channels througout it.
- The *channel_base* parameter allows to specify the size for the first block, which will be doubled later.
- The *fc_layers* parameter specifies the number of activations present inside the fully connected layes through an array of integers.
- The *pool* specifies the type of pooling used by the network. The network is designed to perform only one type of pooling which can be selected by setting the pool parameter to "Avg" or "Max" for the desired type. All pooling will be applied with kernel size 2 and stride 2 to avoid excessive downsampling.
- The boolean parameter *pool_after_block* is used to specify if the pooling operation has to be performed after every block (True) or after each ConvBnRelu unit (False)
- The *drop* parameter specifies if dropout layers have to be inserted between fully connected ones.

We then created some models to observe the behaviour of the network with different layers.

After that we had to finetune ResNet-18