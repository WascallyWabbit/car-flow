file_path_carvana_test = '/Users/Eric Fowler/Downloads/carvana/test/'
file_path_carvana_train = '/Users/Eric Fowler/Downloads/carvana/train/'
SAMPLE_CARVANA_FILE = file_path_carvana_train + '0cdf5b5d0ce1_01.jpg'

def get_carvana_test_tensor_list(path,numclasses=16,num=None):
    import classifier as cls
    return cls.get_tensor_list(numclasses=numclasses, path=path, num=num)