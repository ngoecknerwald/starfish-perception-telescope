# This class exists to collapse RoIs down to final proposals, then pool the features for the final output layer.


class IoU_supression:
    def __init__(self, IoU_threshold=0.7):

        '''
        Instantiate an IoU supression call. Designed
        to remove duplicate RoIs from the stack produced
        by the RPN.

        Arguments:

        IoU_threshold : float
            Threshold above which two returned regions of interest
            are deemed the same underlying object.

        '''

        self.IoU_threshold = IoU_threshold

    def call(roi_list_sort):

        '''
        Reduce an RoI list by removing any entry with an IoU greater
        than a higher ranked RoI.

        Arguments

        roi_list_sort : list
            List of RoIs sorted by likelihood of being a ground truth starfish.

        '''

        pass
