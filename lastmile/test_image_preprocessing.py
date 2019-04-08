""" Validate some of the more straight forward image processing functions. """
import unittest


from image_preprocessing import ImageDataGenerator


class TestImageDataGenerator(self):

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

    def test_parse_sony_list_train(self):
        pass

    def test_parse_sony_list_test(self):
        pass

    def test_parse_sony_list_validation(self):
        pass

