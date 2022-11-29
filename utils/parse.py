import argparse


def argparse_base():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--params_path", type=str, default="./config/params.example.json")
    return parser


def argparse_train():
    parser = argparse_base()
    return parser.parse_args()


def argparse_base_test():
    parser = argparse_base()
    parser.add_argument("-d", "--dataroot", type=str)
    parser.add_argument("-n", "--name", type=str, required=True)
    return parser


def argparse_test():
    parser = argparse_base_test()
    return parser.parse_args()


def argparse_gradcam():
    parser = argparse_base_test()
    parser.add_argument("--image_dir_A", type=str, required=True)
    parser.add_argument("--image_dir_B", type=str, required=True)
    parser.add_argument("-s", "--save_name", type=str, required=True, help="<dir>/gradcam/<save_name>/に保存する")
    parser.add_argument("-w", "--weight_path", type=str)
    parser.add_argument("-n", "--output_num", type=int, default=10)
    return parser.parse_args()
