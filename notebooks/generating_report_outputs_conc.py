# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import ast

def get_accuracies(filepath):
    my_dict = {}
    with open(filepath) as f:
        lines = f.readlines()
        
    for line in lines:
        title, data = line.split(':')
        data = data.replace('\n', '')
        
        if title=='test_accuracy':
            return ast.literal_eval(data)


# +
batchsize = 'batchsize_32'

my_accuracies = {}

for model_no in ['01', '02', '03', '05', '06']:
    no_conv_layers = {'01':3, '02':2, '03': 4, '05':5, '06': 6}
    
    model_name = 'model_' + model_no

    file_path = f'../outputs/{batchsize}/{model_name}/details.txt'
    
    
    my_accuracies[no_conv_layers[model_no]] = max(get_accuracies(file_path)[:-1])
    print()
    
my_accuracies

# -




