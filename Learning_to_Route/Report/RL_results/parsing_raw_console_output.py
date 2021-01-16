from argparse import ArgumentParser
from sys import argv
import re


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-p", "--file_path", type=str, help="The path for console output file")
    parser.add_argument("-regex", "--regex", type=str, help="The regex to extract number", default="\-?\d+.?\d+")
    parser.add_argument("-line_rec", "--line_recognizer", type=str, help="The line recognizer",
                        default="ep_rew_mean")
    parser.add_argument("-scale", "--scale", type=float, help="The scale factor", default=1.0)
    parser.add_argument("-overwrite", "--rewrite", type=bool, help="The overwrite flag", default=False)
    options = parser.parse_args(args)
    return options


def parsing_learning_output(file_path: str, line_recognizer: str, regex: str, sacle: float, rewrite: bool):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    lines = list(filter(lambda l: re.search(line_recognizer, l) is not None, lines))
    lines = list(filter(lambda l: re.search(regex, l) is not None, lines))
    if rewrite:
        lines = list(map(lambda l: str(sacle * float(re.search(regex, l).group(0))) + "\n", lines))
        f = open(file_path, "w")
        f.writelines(lines)
        f.close()
    else:
        import numpy as np
        lines = list(map(lambda l: sacle * float(re.search(regex, l).group(0)), lines))
        print("Mean: {}".format(np.mean(lines)))
        print("STD: {}".format(np.std(lines)))
    return


if __name__ == "__main__":
    file_path = _getOptions().file_path
    scale = _getOptions().scale
    regex = _getOptions().regex
    line_recognizer = _getOptions().line_recognizer.replace("-", " ")
    rewrite = _getOptions().rewrite
    print("Parsing {}".format(file_path))
    parsing_learning_output(file_path, line_recognizer, regex, scale, rewrite)
    print("Done {}".format(file_path))
