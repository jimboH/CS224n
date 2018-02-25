import tensorflow as tf

def get_initializer(matrix):
    def _initializer(shape=None, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer
