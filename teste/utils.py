import time
import numpy as np
from contextlib import redirect_stdout

def print_np(var):
    print(var.shape, var.dtype, np.min(var), np.max(var))

def PIL2numpy(img_pil):
    img_np = np.array(img_pil)
    
    if len(img_np.shape)==3:
        #img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        pass

    return img_np

def get_layers(model):
    name = model.name
    file = f'./{name}'+'-layers.txt'

    with open(file, 'w') as f:
        try:
            i = 0
            while(1):
                f.write(str(model.get_layer(index=i).get_config()) + '\n')
                i+=1
        except Exception as e:
            print(e)


def get_summary(model):
    name = model.name
    file = f'./{name}'+'-summary.txt'

    with open(file, 'w') as f:
        with redirect_stdout(f):
            model.summary()

def pause_for(sec):
    print("Paused for {}s...".format(sec))
    time.sleep(sec)