import argparse


def argparse_gradcam():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-p", "--params_path", type=str)
    parser.add_argument("-i", "--image_dir_path", type=str)
    parser.add_argument("-w", "--weight_path", type=str)
    parser.add_argument("-n", "--output_num", type=int, default=5)
    return parser.parse_args()
