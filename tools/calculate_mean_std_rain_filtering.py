import os
import numpy as np
import mmseg
import mmcv
from tqdm import tqdm

image_type_list = ['first_depth', 'first_intensity', 'first_return_type',
                   'last_depth', 'last_intensity', 'last_return_type',
                  ]

class SequentialCalculater():
    def __init__(self, file_path, image_type_list, img_shape= (40, 1800,6)):
        self.file_path = file_path
        self.image_type_list = image_type_list
        self.image_file_list = os.listdir(os.path.join(file_path, image_type_list[0]))
        self.img_shape = img_shape
        file_client_args=dict(backend='disk').copy()
        self.file_client = mmcv.FileClient(file_client_args['backend'])
    def sequential_calc(self, calc=['mean', 'var']):
        
        mu = np.zeros(self.img_shape)
        var = np.zeros(self.img_shape)
        
        for i, image_file in tqdm(enumerate(self.image_file_list)):
            n = i
            concat_image = self._load_concat_image(image_file).astype(np.float64)
            mu_n = mu
            var_n = var
            #sequential average
            mu = 1/(n+1)*(mu_n*n + concat_image)
            
            if 'var' in calc:
                var = (n*(var_n + mu_n**2) + concat_image**2)/(n+1) - mu**2
                
                var = np.sqrt(var)
        return mu, var
    
    def norm_param_as_rgb(self):
        mu, var = self.sequential_calc()
        
        mu_rgb = np.nanmean(mu,0).mean(0).reshape(2,3).mean(0)
        var_rgb = np.nanmean(var,0).mean(0).reshape(2,3).mean(0)
        
        return mu_rgb, var_rgb
        
            
            
    def _load_concat_image(self, image_file):
        for i, img_type in enumerate(image_type_list):
            filename = f'/{self.file_path}/{img_type}/{image_file}'
            img_bytes = self.file_client.get(filename)
            img = mmcv.imfrombytes(
               img_bytes, flag='unchanged', backend='cv2')
            if len(img.shape) ==2:
                img = img[...,np.newaxis]
            if i==0:
                img_concat = img 
            else:

                img_concat = np.concatenate([img_concat, img], axis=2)
        return img_concat 

sq = SequentialCalculater('/var/datasets/rain_filtering/range_images/train/', image_type_list)

train_mean,train_std= sq.norm_param_as_rgb()
print(train_mean)
print(train_std)