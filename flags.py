import tensorflow as tf
from main import main as m

flags = tf.app.flags

flags.DEFINE_string("device","/cpu:0","default device for ")

def main():
    config=flags.FLAGS
    m(config)
