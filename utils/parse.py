import argparse
import json
import os


def argparse_base() -> argparse.ArgumentParser:
    """すべてのargparseの元。共通で処理させたいことを記述。

    Returns:
        argparse.ArgumentParser: argparseの実体。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default="0")
    return parser


def argparse_base_train() -> argparse.ArgumentParser:
    parser = argparse_base()
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-A", "--trainA", type=str, required=True)
    parser.add_argument("-B", "--trainB", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--imbalance",
        type=str,
        choices=["undersampling", "oversampling", "inverse_class_freq"],
        help='["undersampling", "oversampling", "inverse_class_freq"] or None',
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

    return parser


def argparse_train() -> dict:
    """訓練用のオプションパラメータを定義。

    Returns:
        dict: 訓練のパラメータを辞書に格納したもの。本来ならパラメータを直接返すべき。負の遺産。
    """
    parser = argparse_base_train()
    parser.add_argument("--num_estimate", type=int, default=10)
    args = parser.parse_args()
    params = args_to_dict(args)

    # trainA直下のフォルダの名前をラベル名とする
    dirs = os.listdir(params["trainA"])
    params["labels"] = sorted(dirs)
    params["phase"] = "train"

    params = sort_dict_by_key(params)
    print_params(params)
    return params


def argparse_cv() -> dict:
    parser = argparse_base_train()
    parser.add_argument("--cv_n_split", type=int, default=3)

    args = parser.parse_args()
    params = args_to_dict(args)

    # trainA直下のフォルダの名前をラベル名とする
    dirs = os.listdir(params["trainA"])
    params["labels"] = sorted(dirs)
    params["phase"] = "train"

    params = sort_dict_by_key(params)
    print_params(params)
    return params


def argparse_base_test() -> argparse.ArgumentParser:
    """テストのオプションパラメータのベース。argparse_testとargparse_gradcamで呼び出す。

    Returns:
        argparse.ArgumentParser: argparseの実体。
    """
    parser = argparse_base()
    parser.add_argument("-p", "--params_path", type=str, required=True)
    return parser


def argparse_test() -> dict:
    """テストのオプションパラメータを定義。

    Returns:
        dict: テストのパラメータを格納した辞書。
    """
    parser = argparse_base_test()
    parser.add_argument("-t", "--test_name", type=str, required=True)
    parser.add_argument("-d", "--dataroot", type=str, required=True)

    args = parser.parse_args()
    with open(args.params_path, "r") as params_file:  # テストはparams.jsonからパラメータを取得
        params = json.load(params_file)

    params["test_name"] = args.test_name
    params["test"] = args.dataroot
    params["gpu_id"] = args.gpu_id
    params["phase"] = "test"

    params = sort_dict_by_key(params)
    print_params(params)
    return params


def argparse_gradcam() -> dict:
    """GradCAMのオプションパラメータを定義。

    Returns:
        dict: GradCAMのパラメータを格納した辞書。
    """
    parser = argparse_base_test()

    args = parser.parse_args()
    with open(args.params_path, "r") as params_file:  # GradCAMもparams.jsonからパラメータを取得
        params = json.load(params_file)

    params["gpu_id"] = args.gpu_id
    params["phase"] = "gradcam"

    params = sort_dict_by_key(params)
    print_params(params)
    return params


def args_to_dict(args: argparse.Namespace) -> dict:
    """argparseの各パラメータを辞書型に変換。

    Args:
        args (argparse.Namespace): 変換するargparseの実体。

    Returns:
        dict: 辞書型に変換されたパラメータ。
    """
    temp_dict = dict()
    for arg in dir(args):
        if arg[0] != "_":  # 文字列の先頭が"_"の属性はargparseが定義済みのアクセスが制限された属性。
            temp_dict[arg] = getattr(args, arg)
    return temp_dict


def sort_dict_by_key(dict_items: dict) -> dict:
    """辞書をキーでソートする。

    Args:
        dict_items (dict): ソートした辞書。

    Returns:
        dict: キーでソートされた辞書。
    """
    sorted_items = sorted(dict_items.items())
    sorted_dict_items = {}
    for (key, value) in sorted_items:
        sorted_dict_items[key] = value
    return sorted_dict_items


def print_params(params: dict, nest: int = 0):
    """ネストされた辞書型のパラメータを整えて表示する。

    Args:
        params (dict): 表示するパラメータ。
        nest (int, optional): ネストの深さ。再起呼び出しで自動で加算される。 Defaults to 0.
    """
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
