mport gradio as gr
import torch
from torchvision import transforms
import requests
from PIL import Image
from net import Net, Vgg16
import numpy as np


model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval()

# Download human-readable labels for ImageNet.
response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def preprocess_batch(batch):
    batch = batch.transpose(0, 1)
    (r, g, b) = torch.chunk(batch, 3)
    batch = torch.cat((b, g, r))
    batch = batch.transpose(0, 1)
    return batch

def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):

    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img

def tensor_load_rgbimage_2(filename, size=None, scale=None, keep_asp=False):

    img = Image.fromarray(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def evaluate(img):
    content_image = tensor_load_rgbimage_2(img, size=1024, keep_asp=True)
    #content_image = img
    content_image = content_image.unsqueeze(0)
    style = tensor_load_rgbimage("images/21styles/candy.jpg", size=512)
    style = style.unsqueeze(0)    
    style = preprocess_batch(style)
    
    style_model = Net(ngf=128)
    model_dict = torch.load("models/21styles.model")
    model_dict_clone = model_dict.copy()
    for key, value in model_dict_clone.items():
        if key.endswith(('running_mean', 'running_var')):
            del model_dict[key]
    style_model.load_state_dict(model_dict, False)

    style_v = Variable(style)

    content_image = Variable(preprocess_batch(content_image))
    style_model.setTarget(style_v)

    output = style_model(content_image)
    img = output.data[0].clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    #output = utils.color_match(output, style_v)
    return img

def predict(inp):
    return inp
inputs = gr.inputs.Image()
outputs = gr.outputs.Image()
gr.Interface(fn=evaluate, inputs=inputs, outputs=outputs).launch(share=False)
