# This will contain the RPN class, here a totally empty stub


class RPN:
    def __init__(image_shape):
        
        # Store the image size
        self.image_shape = image_shape
        
        # Compute bounding boxes for future use
       self._build_anchors()

        # Now initialize networks


    def _build_anchors(
        self,
        stride=32,
        window_sizes=[64, 128, 256],
        aspect_ratios=[0.7, 1, 1.3],
        boundary='clip',
    ):

        '''
        Build the anchor boxes for the RPN based on the paper.

        Sets self.xx_yy containing the bounding box definitions.

        Arguments:

        stride : int
            Place an anchor every <stride> pixels in the input image
        window_sizes : list of ints
            Width of the proposals to select.
        aspect_ratios : list of floats
            Set of aspect ratios (height / width) to select.
        boundary : 'clip' or 'discard'
            Either clip or discard proposal boxes that run into the edges of the images

        '''

        assert boundary.lower() in ['clip', 'discard']

        # Make the list of window sizes
        _hh, _ww = np.meshgrid(window_sizes, window_sizes)
        _hh *= aspect_ratios[:, np.newaxis]

        # Flatten for simplicity later
        _hh = _hh.reshape[-1]
        _ww = _ww.reshape[-1]

        xmin = []
        ymin = []
        xmax = []
        ymax = []

        # For each point, define an anchor box
        for xx in range(0, self.image_shape[1], stride):
            for yy in range(0, self.image_shape[0], stride):
                for hh, ww in zip(_hh, _ww):
                    # Proposed box
                    lxmin = int(xx - ww / 2)
                    lxmax = int(xx + ww / 2)
                    lymin = int(yy - hh / 2)
                    lymax = int(yy + hh / 2)

                    # Clip to valid boxes
                    if boundary == 'clip':
                        xmin.append(np.maximum(0, lxmin))
                        ymin.append(np.maximum(0, lymin))
                        xmax.append(np.minimum(lxmax, self.image_size[1]))
                        ymax.append(np.minimum(lymax, self.image_size[0]))
                    # if not, then just go to check if valid or discard
                    elif (lxmin > 0 and lymin > 0) and (
                        lxmax < self.image_shape[1] and lymax < self.image_shape[0]
                    ):
                        xmin.append(lymin)
                        ymin.append(lymin)
                        xmax.append(lxmax)
                        ymax.append(lymax)
                    else:
                        pass

                    # if not, then just go to discard

        self.xx_yy = np.stack[xmin, xmax, ymin, ymax]


    # Next build the network that goes from a 
