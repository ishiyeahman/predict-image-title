import io
import PIL 
import pickle
import torch

from predict import *
from function import *


"""config"""
# modelが入っているパスを設定する
PATH = '/Users/ishiyamaryo/projects/hackathon-kumamoto/src/models/'


def torch_load(file_name):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with open(file_name, 'rb') as f:
        return torch.load(f, map_location=device) 
    
def read_models(path):
    ex = 'pt'
    filename = path + f'model.{ex}'
    model = torch_load(filename)

    filename = path + f'clip_model.{ex}'
    # clip_model = pickle.load(open(filename, 'rb'))
    clip_model = torch_load(filename)
    
    filename = path + f'preprocess.{ex}'
    # preprocess = pickle.load(open(filename, 'rb'))
    preprocess = torch_load(filename)

    filename = path + f'tokenizer.{ex}'
    # tokenizer = pickle.load(open(filename, 'rb'))
    tokenizer = torch_load(filename)
    
    
    filename = path + f'prefix_length.{ex}'
    # prefix_length = pickle.load(open(filename, 'rb'))
    prefix_length = 10

    return model , clip_model , preprocess , tokenizer, prefix_length

def get_image_title(image, model, clip_model, preprocess, tokenizer, prefix_length):
    #@title 画像から文の生成
    image = '01.jpg' #@param {type:"string"}
    image_path = 'images/'+image
    use_beam_search = False #@param {type:"boolean"}  

    image = io.imread(image_path)
    pil_image = PIL.Image.fromarray(image)
    #pil_img = Image(filename=UPLOADED_FILE)

    image = preprocess(pil_image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        # if type(model) is ClipCaptionE2E:
        #     prefix_embed = model.forward_image(image)
        # else:
        prefix = clip_model.encode_image(image).to('cuda', dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    print('\n')
    print(generated_text_prefix)

def main():
    model, clip_model, preprocess, tokenizer, prefix_length = read_models(PATH)
    
    #画像のファイル名を引数に渡す
    image_title = get_image_title('a', model, clip_model, preprocess, tokenizer, prefix_length)
    return image_title


if __name__ == "__main__":
    print(main())