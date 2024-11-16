# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 13:15
# @Author  : debt


import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from models.model import Detector

# 模型实例化
model = Detector()

# Create sample inputs
a = torch.randn([2, 3, 384, 384])
b = torch.randn([2, 3, 384, 384])


# THOP requires the model to have a single input and output
# Here we create a wrapper model to make it compatible with THOP
class SingleInputModelWrapper(nn.Module):
    def __init__(self, model):
        super(SingleInputModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        # Since THOP expects single input, we concatenate input1 and input2
        # and pass it to the model, then split the output
        s, e, s1 = self.model(input, input)
        return s  # Return any one of the outputs, since they should be same


# Wrap the double input model
single_input_model = SingleInputModelWrapper(model)

# Calculate FLOPs and parameters using THOP
flops, params = profile(single_input_model, inputs=(a,))
gflops = flops / 1e9  # Convert FLOPs to GFLOPs
million_params = params / 1e6  # Convert params to million (M)
print(f"Total GFLOPs: {gflops}")
print(f"Total parameters (M): {million_params}")
