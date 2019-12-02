'''
Slipt dataset in a stratified way

train: 75%, test: 10%. val: 15%

seed: 12345

'''
import split_folders

# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
split_folders.ratio(['PATH DO DIRETÓRIO DE ORIGEM'], 
		    output=['PATH DE SAIDA ONDE SERÁ CRIADO /test /train /val'], 
		    seed=12345, ratio=(.75, .15, .1)) # default values
