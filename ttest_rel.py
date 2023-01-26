import numpy as np
import pandas
import scipy

from utils.parse import argparse_ttest_rel


def main():
    args = argparse_ttest_rel()
    pathA = str(args.pathA)
    pathB = str(args.pathB)

    df_A = pandas.read_csv(pathA)
    f1_score_A = df_A[df_A.iloc[:, 0] == "F1 Score_Mean"].iloc[:, 2:].to_numpy()[0]
    print("pathA F1 Score:", f1_score_A)
    print(f"平均:{np.mean(f1_score_A).round(decimals=2)}\t標準偏差:{np.std(f1_score_A, ddof=1).round(decimals=2)}\n")

    df_B = pandas.read_csv(pathB)
    f1_score_B = df_B[df_B.iloc[:, 0] == "F1 Score_Mean"].iloc[:, 2:].to_numpy()[0]
    print("pathA F1 Score:", f1_score_B)
    print(f"平均:{np.mean(f1_score_B).round(decimals=2)}\t標準偏差:{np.std(f1_score_B, ddof=1).round(decimals=2)}\n")

    print(scipy.stats.ttest_rel(f1_score_A, f1_score_B, alternative="less"))


if __name__ == "__main__":
    main()
