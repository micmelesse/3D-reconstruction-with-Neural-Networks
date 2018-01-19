
import re
regex = "^.*=(.*)$"


def read_param(param_line):
    return re.findall(regex, param_line)[0]
