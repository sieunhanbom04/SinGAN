from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
from matplotlib import pyplot as plt
from skimage import io as img
import bm3d
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)

def PSNR(img1, img2, peak=1):
    '''
    Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
    '''
    import numpy as np

    x = ((np.array(img1).squeeze() - np.array(img2).squeeze()).flatten() )
    return (10*np.log10(peak**2 / np.mean(x**2)))


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help = 'input image dir', default = 'Input/Images')
    parser.add_argument('--input_name', help = 'training image name', required = True)
    parser.add_argument('--denoise_scale', help='denoising scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='harmonization')
    parser.add_argument('--noise', help='sigma of Gaussian noise', default = 5.0)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = '%s/Denoise/%s' % (opt.out, opt.input_name[:-4])
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    real = functions.read_image(opt)
    #Maybe we need to keep real to reduce error
    temp = real * 1.0
    real = functions.adjust_scales2image(real, opt)
    real = temp
    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    if (opt.denoise_scale < 1) | (opt.denoise_scale > (len(Gs)-1)):
        print("injection scale should be between 1 and %d" % (len(Gs)-1))
    else:
        clean_img = img.imread('%s/%s' % (opt.input_dir, opt.input_name)).astype(np.float32)
        noisy_img = clean_img + np.random.normal(0, float(opt.noise), clean_img.shape)
        noisy_img = functions.np2torch(noisy_img, opt)
        noisy_img = noisy_img[:,0:3,:,:]
        clean_img = clean_img[:,:,0:3] / 255.0

        N = len(reals) - 1
        n = opt.denoise_scale
        noisy_img = functions.read_image_dir('denoised.png', opt)
        in_s = imresize(noisy_img, pow(opt.scale_factor, (N - n + 1)), opt)
        in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
        in_s = imresize(in_s, 1 / opt.scale_factor, opt)
        in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]
        out = SinGAN_denoise(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
        denoise_img = functions.convert_image_np(out.detach())
        print(PSNR(denoise_img, clean_img))
        #plt.imshow(clean_img)
        plt.imshow(denoise_img)
        plt.show()
        plt.imsave('Output/denoise' + opt.input_name[:-4] + '_denoise' + opt.input_name[-4:], denoise_img, vmin =0.0, vmax = 1.0)
