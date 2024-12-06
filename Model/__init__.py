from spektral.layers import ops

ops.modes.OTHER = -1
sp_autodetect_mode = ops.autodetect_mode

def new_autodetect_mode(x, a):
    try:
        return sp_autodetect_mode(x, a)
    except ValueError:
        return ops.modes.OTHER

ops.autodetect_mode = new_autodetect_mode