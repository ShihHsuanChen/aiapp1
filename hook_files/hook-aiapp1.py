""" https://pyinstaller.org/en/stable/hooks.html#PyInstaller.utils.hooks.copy_metadata """
from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata('aiapp1') # to solve packaing version missing problem
datas += [
    ('./models/mobilenetv4_conv_small.e2400_r224_in1k/', 'models/mobilenetv4_conv_small.e2400_r224_in1k')
]
