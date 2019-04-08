""" Validate some of the more straight forward image processing functions. """
import unittest


from image_preprocessing import ImageDataGenerator


class TestImageDataGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Sony train list

        with open("dataset/Sony_train_list.txt", "r") as infile:
            cls.sony_train_list = infile.read()

        # Sony test list

        with open("dataset/Sony_test_list.txt", "r") as infile:
            cls.sony_test_list = infile.read()

        # Sony validation list

        with open("dataset/Sony_val_list.txt", "r") as infile:
            cls.sony_val_list = infile.read()

    def setUp(self):
        self.idg = ImageDataGenerator(preprocessing_function='bl',
                                      random_seed=0)

    def test_reformat_imgpath(self):
        img_path = './Sony/short/00194_06_0.04s.ARW'
        output = self.idg.reformat_imgpath(img_path)

        assert output == 'Sony/rgb/short/00194_06_0.04s.png'

    def test_parse_sony_list_train(self):
        sony_list = 'dataset/Sony_train_list.txt'
        output = self.idg.parse_sony_list(sony_list)

        assert output[0] == ('Sony/rgb/short/00124_03_0.1s.png',
                             'Sony/rgb/long/00124_00_30s.png')

    def test_parse_sony_list_test(self):
        sony_list = 'dataset/Sony_test_list.txt'

        output = self.idg.parse_sony_list(sony_list)

        assert output[0] == ('Sony/rgb/short/10045_02_0.1s.png',
                             'Sony/rgb/long/10045_00_10s.png')

    def test_parse_sony_list_validation(self):
        sony_list = 'dataset/Sony_val_list.txt'

        output = self.idg.parse_sony_list(sony_list)

        assert output[0] == ('Sony/rgb/short/20211_00_0.1s.png',
                             'Sony/rgb/long/20211_00_10s.png')
