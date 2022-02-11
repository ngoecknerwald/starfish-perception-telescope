-----------------
Starfish Detector
-----------------

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

To instantiate the model and train from pretrained ImageNet weights one calls:

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

Results
=======

To be determined!
