from Learning_to_Route.common.size_consts import SizeConsts


def norm_func(x, norm_val=1. * SizeConsts.ONE_Gb):
    return x / norm_val
