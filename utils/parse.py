import argparse
import json
import os


def argparse_base() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser


# TODO: README更新
def argparse_train() -> dict:
    parser = argparse_base()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-A", "--trainA", type=str, required=True)
    parser.add_argument("-B", "--trainB", type=str)
    parser.add_argument("--num_estimate", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--imbalance", type=str, default="undersampling", help="[undersampling | oversampling | lossweight, None]"
    )
    parser.add_argument("--img_resize", type=int, default=224)
    parser.add_argument("--mean", type=float, nargs=3, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, nargs=3, default=[0.229, 0.224, 0.225])
    parser.add_argument("--grayscale_flag", action="store_true")
    parser.add_argument("--normalize_per_img", action="store_true")
    parser.add_argument("--net_name", type=str, default="resnet18")
    parser.add_argument("--multi_net", action="store_true")
    parser.add_argument("--pretrained", action="store_false")
    parser.add_argument("--transfer_learning", action="store_true")
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--optim_name", type=str, default="Adam")
    parser.add_argument("--lr_not_pretrained", type=float, default=1e-4)
    parser.add_argument("--lr_pretrained", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    params = args_to_params(args)

    dirs = os.listdir(params["trainA"])
    params["labels"] = sorted(dirs)
    params["phase"] = "train"

    params = sort_dict(params)
    print_params(params)
    return params


def argparse_base_test() -> argparse.ArgumentParser:
    parser = argparse_base()
    parser.add_argument("-p", "--params_path", type=str, required=True)
    return parser


def argparse_test() -> dict:
    parser = argparse_base_test()
    parser.add_argument("-t", "--test_name", type=str, required=True)
    parser.add_argument("-d", "--dataroot", type=str, required=True)

    args = parser.parse_args()
    with open(args.params_path, "r") as params_file:
        params = json.load(params_file)

    params["test_name"] = args.test_name
    params["test"] = args.dataroot
    params["phase"] = "test"

    params = sort_dict(params)
    print_params(params)
    return params


def argparse_gradcam() -> dict:
    parser = argparse_base_test()

    args = parser.parse_args()
    with open(args.params_path, "r") as params_file:
        params = json.load(params_file)

    params["phase"] = "gradcam"
    print_params(params)
    return params


def args_to_params(args: argparse.Namespace) -> dict:
    params = dict()
    for arg in dir(args):
        if arg[0] != "_":
            params[arg] = getattr(args, arg)
    return params


def sort_dict(dict_items: dict) -> dict:
    sorted_items = sorted(dict_items.items())
    sorted_dict_items = {}
    for (key, value) in sorted_items:
        sorted_dict_items[key] = value
    return sorted_dict_items


def print_params(params: dict, nest: int = 0):
    print("=============== params ===============")
    for param in params:
        print("\t" * nest, param, end=":")
        if type(params[param]) == dict:
            print("{")
            print_params(params[param], nest=nest + 1)
            print("}\n")
        else:
            print("\t", params[param])
    print("======================================")
