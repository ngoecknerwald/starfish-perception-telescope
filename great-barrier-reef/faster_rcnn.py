# This script will contain the faster R-CNN class

from rpn import RPN
from backbone import Backbone


class FasterRCNN:
    def __init__(self):

        # Instantiate components
        self.backbone = Backbone()
        self.rpn = RPN()

    def train(self, images_full, images_thumb):
        pass
