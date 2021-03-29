# Part 1: RGB Image #
class RGBImage:
    """
    Create the class for image objects using the RGB color spaces.
    """

    def __init__(self, pixels):
        """
        Initialize a RGBImage instance.
        """
        self.pixels = pixels  # initialze the pixels list here

    def size(self):
        """
        Get the size of a image representing by a tuple of (number of rows,
        number of columns).
        """
        return (len(self.pixels[0]), len(self.pixels[0][0]))

    def get_pixels(self):
        """
        Get a copy of the pixels matrix of the image.
        """
        copy_of_pixels = list(self.pixels)
        return copy_of_pixels

    def copy(self):
        """
        Get a copy of the given RGBImage instance.
        """
        copy_of_pixels = self.get_pixels()
        return RGBImage(copy_of_pixels)

    def get_pixel(self, row, col):
        """
        Get the color of the pixel at the given position in a tuple.
        """
        assert type(row) == int and row >= 0 and type(col) == int and col >= 0
        assert row <= self.size()[0] and col <= self.size()[1]
        return (self.pixels[0][row][col], self.pixels[1][row][col], \
        self.pixels[2][row][col])

    def set_pixel(self, row, col, new_color):
        """
        Change the color of the given pixel to the given new color inplace.
        """
        assert type(row) == int and row >= 0 and type(col) == int and col >= 0
        assert row <= self.size()[0] and col <= self.size()[1]
        for i in range(3):
            if not new_color[i] == -1:
                self.pixels[i][row][col] = new_color[i]



# Part 2: Image Processing Methods #
class ImageProcessing:
    """
    Create a class for all the image processing methods.
    """

    @staticmethod
    def negate(image):
        """
        Get the negative image of the input.
        """
        new_pixels = [[[255 - image.get_pixels()[x][y][z] for z in \
        range(len(image.get_pixels()[x][y]))] for y in \
        range(len(image.get_pixels()[x]))] for x in \
        range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def grayscale(image):
        """
        Get a new image obtained by converting the input to grayscale.
        """
        new_pixels = [[[sum([image.get_pixels()[c][y][z] for c in \
        range(len(image.get_pixels()))])//3 for z in \
        range(len(image.get_pixels()[x][y]))] for y in \
        range(len(image.get_pixels()[x]))] for x in \
        range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def scale_channel(image, channel, scale):
        """
        Get a new image obtained by scaling all values in the given channal
        by the given scale. The value is capped to 255.
        """
        def scale_val(num, chan):
            '''
            An inner function that scales the value by the given scale if it
            is in the given channel.
            '''
            if chan == channel:
                num = int(num * scale)
            if num > 255:
                return 255
            return num
        new_pixels = [[[scale_val(image.get_pixels()[x][y][z], x) for z in \
        range(len(image.get_pixels()[x][y]))] for y in \
        range(len(image.get_pixels()[x]))] for x \
        in range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def clear_channel(image, channel):
        """
        Get a new image obtained by set all values in the given channel
        to 0.
        """
        def clear(num, chan):
            '''
            An inner function that set the given value to 0 if it is in the
            given channel.
            '''
            if chan == channel:
                num = 0
            return num
        new_pixels = [[[clear(image.get_pixels()[x][y][z], x) for z in \
        range(len(image.get_pixels()[x][y]))] for y in \
        range(len(image.get_pixels()[x]))] for x \
        in range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def rotate_90(image, clockwise):
        """
        Get a new image obtained by rotate input by 90 degrees. The direction
        of rotation is based on the clockwise input, which is a boolean value.
        """
        def rotate(channel, row, col, direction):
            '''
            An inner function that return the result color intensity of the
            given pixel after rotation.
            '''
            if direction:
                return image.get_pixels()[channel][image.size()[0]-col-1][row]
            return image.get_pixels()[channel][col][image.size()[1]-row-1]
        new_pixels = [[[rotate(x, y, z, clockwise) \
        for z in range(image.size()[0])] \
        for y in range(image.size()[1])] \
        for x in range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def crop(image, tl_row, tl_col, target_size):
        """
        Get a new image obtained by cut the given image from the given top-left
        corner and take the given size to the right and to the bottom of it.
        """
        actual_size = [target_size[0], target_size[1]]
        if tl_row + actual_size[0] > image.size()[0]:
            actual_size[0] = image.size()[0] - tl_row
        if tl_row + actual_size[1] > image.size()[1]:
            actual_size[1] = image.size()[1] - tl_col
        new_pixels = [[[image.get_pixels()[x][tl_row + y][tl_col + z] \
        for z in range(actual_size[1])] \
        for y in range(actual_size[0])] \
        for x in range(len(image.get_pixels()))]
        return RGBImage(new_pixels)

    @staticmethod
    def chroma_key(chroma_image, background_image, color):
        """
        Get a new image obtained by replacing all pixels in the chroma image
        with the given color to the corresponding pixel in the background
        image.
        """
        assert type(chroma_image) == RGBImage
        assert type(background_image) == RGBImage
        assert chroma_image.size() == background_image.size()
        def check_color(channel, row, col):
            '''
            An inner function that check if a pixel in the chroma image has
            the given color. If so, return the corresponding value in the
            background image. If not, return the original color.
            '''
            if all([chroma_image.get_pixels()[n][row][col] == \
            color[n] for n in range(len(color))]):
                return background_image.get_pixels()[channel][row][col]
            return chroma_image.get_pixels()[channel][row][col]
        new_pixels = [[[check_color(x, y, z) \
        for z in range(len(chroma_image.get_pixels()[x][y]))] \
        for y in range(len(chroma_image.get_pixels()[x]))] \
        for x in range(len(chroma_image.get_pixels()))]
        return RGBImage(new_pixels)


# Part 3: Image KNN Classifier #
class ImageKNNClassifier:
    """
    A class for classifiers that use K-nearest Neighbors to classify images.
    """

    def __init__(self, n_neighbors):
        """
        Initial a ImageKNNClassifier instance.
        """
        self.n_neighbors = n_neighbors
        self.data = []

    def fit(self, data):
        """
        A method that stores given data in the classifier.
        """
        assert len(data) > self.n_neighbors
        assert self.data == []
        self.data = data

    @staticmethod
    def distance(image1, image2):
        """
        A method that calculates the Euclidean distance between two images.
        """
        assert type(image1) == RGBImage and type(image2) == RGBImage
        assert image1.size() == image2.size()
        return sum([(image1.get_pixels()[x][y][z] - \
        image2.get_pixels()[x][y][z])**2 \
        for x in range(len(image1.get_pixels())) \
        for y in range(len(image1.get_pixels()[x])) \
        for z in range(len(image1.get_pixels()[x][y]))])**0.5

    @staticmethod
    def vote(candidates):
        """
        A method that returns the most popular label in the input list.
        """
        number_of_appearance = dict()
        current_most = 0
        current_most_label = ''
        for i in candidates:
            if i in number_of_appearance:
                number_of_appearance[i] += 1
            else:
                number_of_appearance[i] = 1
            if number_of_appearance[i] > current_most:
                current_most = number_of_appearance[i]
                current_most_label = i
        return current_most_label

    def predict(self, image):
        """
        A function that predicts the label of the input image by using KNN
        classification on the training data of the classifier.
        """
        assert self.data != []
        sorted_distances = [ImageKNNClassifier.distance(i[0], image) \
        for i in self.data]
        sorted_distances.sort()
        candidates = list(filter(lambda x: ImageKNNClassifier.distance(x[0], \
        image) <= sorted_distances[self.n_neighbors - 1], self.data))
        return ImageKNNClassifier.vote(candidates)
