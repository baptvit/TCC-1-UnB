'''
Notebook to resize imagens in base dataset and convert all to RGB
'''

import os, sys
from PIL import Image

# diretório/<pastas das lesões>
path=['PATH DO DIRETÓRIO ONDE ESTA A BASE DE DADOS']
output_path=['PATH DO DIRETÓRIO ONDE DESEJA SALVAR AS NOVAS IMAGENS']
dirs = os.listdir(path)

def resize_convert_rename():
'''
Function to rename imagens with acceding number, convert images to rgb and rezise all imagens to 448x488 to reduce process complexity in later operations
'''
    i = 0
    for sub_dir in dirs:
        for item in os.listdir(path + sub_dir):
            if os.path.isfile(path + sub_dir + '/' + item):
                im = Image.open(path + sub_dir + '/' + item)
                f, e = os.path.splitext(path + sub_dir + '/' + item)
                imResize = im.resize((448,448), Image.ANTIALIAS)
                rgb_im = imResize.convert('RGB')
                rgb_im.save(output_path + sub_dir + '/' + str(i) + '.jpg', 'JPEG', quality=90)
                i = i + 1

resize_convert_rename()
