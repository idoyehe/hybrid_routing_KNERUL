from argparse import ArgumentParser
from sys import argv


def _getOptions(args=argv[1:]):
    parser = ArgumentParser(description="Parses path for console output file")
    parser.add_argument("-p", "--file_path", type=str, help="The path for console output file")
    options = parser.parse_args(args)
    return options


def parsing_learning_output(file_path: str):
    f = open(file_path, "r")
    lines = f.readlines()
    f.close()
    lines = list(filter(lambda l: "ep_rew_mean" in str(l), lines))
    lines = list(map(lambda l: str(-1 * float(str(l).split("|")[2].split()[0])) + "\n", lines))
    f = open(file_path, "w")
    f.writelines(lines)
    f.close()


if __name__ == "__main__":
    file_path = _getOptions().file_path
    print("Parsing {}".format(file_path))
    parsing_learning_output(file_path)
    print("Done {}".format(file_path))
