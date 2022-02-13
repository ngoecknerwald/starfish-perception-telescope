-----------------
Starfish Detector
-----------------

This repository contains our solution to the Kaggle 
`TensorFlow - Help Protect the Great Barrier Reef <https://www.kaggle.com/c/tensorflow-great-barrier-reef/>`_
competition. We have implemented a slightly modified version of the 
`Faster R-CNN <https://arxiv.org/pdf/1506.01497.pdf>`_ algorithm for object detection.

Authors
=======

Sam Guns and `Neil Goeckner-Wald <https://ngoecknerwald.github.io/>`_

Both authors contributed equally and were involved in all aspects of the project.

Guide to the Code
=================

#. ``backbone.py`` : General feature extraction backbone class and subclasses to implement different architectures.
#. ``callback.py`` : Callback routines for training, used to set the learning rate versus epoch.
#. ``classifier.py`` : Final output network, note that the training calls live in ``faster_rcnn.py``.
#. ``combined-runtime.ipynb``: Colab notebook used to run the code.
#. ``data_utils.py`` : Classes for interfacing with the dataset and matching annotations with images.
#. ``evaluation.py`` : Recall metrics used to monitor training progress.
#. ``faster_rcnn.py`` : High level driver module, instantiates and trains network subcomponents.
#. ``geometry.py`` : IoU and related methods used elsewhere.
#. ``roi_pooling.py`` : Class for the RoI pooling and IoU suppression routines attached to the output of the RPN.
#. ``rpn.py`` : Region proposal network and wrapper classes.
#. ``jointmodel.py`` : Class for jointly optimizing the RPN and backbone.

To instantiate the model and train from pretrained ``ImageNet`` weights one calls:

.. code-block:: python
    
    import faster_rcnn
    
    frcnn = faster_rcnn.FasterRCNNWrapper(
        input_shape=(720, 1280, 3),
        datapath = '<path_to_competition_dataset>',
        backbone_type = 'ResNet50', #InceptionResNetV2 and VGG16 are also supported
        backbone_weights = 'finetune',
        rpn_weights = 'train',
        classifier_weights= 'train',
    )

The model can also be built with pre-trained and weights saved using the utilities

.. code-block:: python

    # Save weights
    frcnn.backbone.save_backbone('<backbone_weights>')
    frcnn.rpnwrapper.save_rpn_state('<rpn_weights>')
    frcnn.classmodel.save_classifier_state('<class_weights>')

    ...
    
    # Rebuild the model
    frcnn = faster_rcnn.FasterRCNNWrapper(
        input_shape=(720, 1280, 3),
        datapath = '<path_to_competition_dataset>',
        backbone_type = 'ResNet50', #InceptionResNetV2 and VGG16 are also supported
        backbone_weights = '<backbone_weights>',
        rpn_weights = '<rpn_weights>',
        classifier_weights= '<class_weights>',
    )

There are a large number of configurable hyperparameters described in the various
docstrings. They have been initialized to reasonable values for the Great Barrier Reef
dataset.

The code can be fine tuned for a number of epochs by calling:

.. code-block:: python

    fine_tuning_passes = 3
    epochs_per_pass = 2

    for _ in range(fine_tuning_passes):
        frcnn.do_fine_tuning(epochs_per_pass)

resulting in substantial accuracy gains.

Results
=======

To be determined!

Future directions
=================

There are a number of ways in which this algorithm could be improved. In no particular order,
here are number of ideas that we considered but did not have time to pursue.

Training schedule improvements
------------------------------

- **Shorten the initial classifier and RPN training**: We trained both networks in isolation for 9 epochs each. We observed diminishing returns in epochs 7-9 so the final three epochs of both could be dropped in favor more joint training.

- **Change the initialization of the networks**: We found that the early training of the RPN and classifier were quite slow and required significant amounts of weight decay and a fairly aggressive gradient clip. This could be mitigated by smarter choices of initial random weights.

- **Implement label smoothing in the classifier**: The classifier is prone to overconfidence assigning classification scores of 0.0 or 1.0 to regions. This could be mitigated by label smoothing in the classifier loss function.

- **Assigning different loss penalties for false positives and false negatives**: The competition is scored with an ``F2`` metric averaged over IoU thresholds between 0.3 and 0.8 meaning that false negatives are more of a problem than false positives. This could be accounted for by assigning different loss penalties for the two types of mistakes.

- **Adding noise to the feature extraction pretraining**: We pre-trained the feature extraction backbone convolutional weights on a starfish / background thumbnail classification task. To do this we placed a global average pool and dense layer on the output of the convolutional layers that were subsequently discarded after pre-training. One possible improvement would be to place a Gaussian noise augmentation and an L2 regularization term after the global average pool to create a simpler boundary between starfish and background regions in the backbone output. This would be similar to (and indeed was inspired by) the resampling step in a variational auto-encoder and could result in a more robust final solution.

Architecture improvements
-------------------------

- **Use an upsampled VGG-16 backbone**: Our network struggled somewhat with localization, likely due to the fact that the backbone stride was on the scale of the starfish in the images themselves. One obvious remedy is to use a convolutional backbone with a smaller effective stride. This could be done by taking the penultimate layer of a pretrained ``VGG-16`` and stacking it with an upsampled version of the final convolutional layer. This has been shown to work in `An Improved Faster R-CNN for Small Object Detection <https://ieeexplore.ieee.org/document/8786135/>`_.

- **Use GIoU loss for localization**: This has been shown to improve localization in Faster R-CNN algorithms relative to the L1 bounding box loss that we used. We used this in early versions of the network but dropped it for simplicity.

- **Use a YOLO architecture**: A single stage detection network would have been simpler to implement and faster to train. 

- **Downweight correlations between RoI in the classifier**: We observed that the classifier had a tendency to over-learn the (real) correlation between input RoI due to the fact that starfish tend to cluster spatially in the training data. This can be mitigated by replacing the output dense layer with another 1x1 convolution and a (regularized) dense correction term to account for the real correlations between RoI.

- **Learn temporal correlations**: There are strong correlations between subsequent images in the training videos which could be exploited by a two-stage detection system. One simple way to do this would be to pass the RoI and pooled features as well as a smoothly varying spatial function from the last ``n ~ 4`` images to the final dense layer in the classifier. This would require another set of training epochs and a data loading interface that does not randomly reshuffle the images.

Dataset improvements
--------------------

- **Dropping background-only images**: The input dataset was quite unbalanced with many more background-only images than images containing starfish. We ended up ignoring many of these images by enforcing a balanced sample in the RPN and classifier training. This resulted in unnecessary calls to the feature extraction backbone which slowed down trainign. Simply ignoring those images alltogether could have resulted in faster training epochs.