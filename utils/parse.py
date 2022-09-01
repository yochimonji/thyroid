import argparse


def argparse_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params_path", type=str, default="./config/params.example.json")
    return parser


def argparse_train():
    parser = argparse_base()
    return parser.parse_args()


def argparse_test():
    parser = argparse_base()
    parser.add_argument("-d", "--dataroot", type=str)
    parser.add_argument("-n", "--name", type=str, required=True)
    return parser.parse_args()


def argparse_gradcam():
    parser = argparse_base()
    parser.add_argument("-d", "--dir", type=str)
    parser.add_argument("-i", "--image_dir_path", type=str)
    parser.add_argument("-w", "--weight_path", type=str)
    parser.add_argument("-n", "--output_num", type=int, default=5)
    return parser.parse_args()
