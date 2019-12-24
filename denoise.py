from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import SinGAN.functions as functions
from matplotlib import pyplot as plt
from skimage import io as img
import pybm3d
from skimage.measure import compare_psnr
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
"""
def noiser_and_ref(clean_img, opt):
    noise = np.random.normal(scale = opt.noise, size = clean_img.shape).astype(np.int16)
    noisy_img = (clean_img.astype(np.int16) + noise).clip(0,255).astype(np.uint8)
    ref_img = pybm3d.bm3d.bm3d(noisy_img, opt.noise)
    ref_img_torch = functions.np2torch(ref_img.astype(np.float), opt)
    ref_img_torch = ref_img_torch[:,0:3,:,:]
    return noisy_img, ref_img, ref_img_torch
"""

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help = 'input image dir', default = 'Input/Images')
    parser.add_argument('--noise_dir', help = 'folder containing noisy image and bm3d image', default = 'Input/Denoise')
    parser.add_argument('--input_name', help = 'training image name', required = True)
    parser.add_argument('--denoise_scale', help='denoising scale', type=int, required=True)
    parser.add_argument('--mode', help='task to be done', default='harmonization')
    parser.add_argument('--noise', help='sigma of Gaussian noise', default = 50)
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
        clean_img = img.imread('%s/%s' % (opt.input_dir, opt.input_name))[:,:,0:3]
        #noisy_img, ref_img, ref_img_torch = noiser_and_ref(clean_img, opt)
        #plt.figure()
        #plt.imshow(noisy_img)
        #plt.show()
        #plt.figure()
        #plt.imshow(ref_img)
        #plt.show()
        #noisy_img = clean_img + np.random.normal(0, float(opt.noise), clean_img.shape)
        #noisy_img = functions.np2torch(noisy_img, opt)
        #noisy_img = noisy_img[:,0:3,:,:]
        #clean_img = clean_img[:,:,0:3] / 255.0
        N = len(reals) - 1
        n = opt.denoise_scale

        noisy_bm3d = []
        noisy_img = img.imread('%s/%s/%s/noisy.png' % (opt.noise_dir, opt.input_name[:-4], str(opt.noise)))[:,:,0:3]
        noisy_img_torch = functions.read_image_dir('%s/%s/%s/noisy.png' % (opt.noise_dir, opt.input_name[:-4], str(opt.noise)), opt)
        noisy_bm3d.append(noisy_img_torch)
        bm3d_img = img.imread('%s/%s/%s/denoised.png' % (opt.noise_dir, opt.input_name[:-4], str(opt.noise)))[:,:,0:3]
        bm3d_img_torch = functions.read_image_dir('%s/%s/%s/denoised.png' % (opt.noise_dir, opt.input_name[:-4], str(opt.noise)), opt)
        noisy_bm3d.append(bm3d_img_torch)
        print('BM3D result:', compare_psnr(bm3d_img, clean_img))

        for index, ref in enumerate(noisy_bm3d):
            temp = functions.convert_image_np(ref) * 255
            temp = temp.astype(np.uint8)
            plt.figure()
            plt.imshow(temp)
            print(compare_psnr(temp, clean_img))
            plt.show()
            in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
            in_s = in_s[:, :, : reals[n - 1].shape[2], : reals[n - 1].shape[3]]
            in_s = imresize(in_s, 1 / opt.scale_factor, opt)
            in_s = in_s[:, :, : reals[n].shape[2], : reals[n].shape[3]]
            out = SinGAN_denoise(Gs[n:], Zs[n:], reals, NoiseAmp[n:], opt, in_s, n=n, num_samples=1)
            denoise_img = functions.convert_image_np(out.detach()) * 255
            denoise_img = np.clip(denoise_img, 0, 255).astype(np.uint8)
            plt.figure()
            plt.imshow(denoise_img)
            plt.show()
            if index == 0:
                name = 'No preprocessing'
            else:
                name = 'bm3d preprocessing'
                plt.imsave('%s/%s/%s/SinGANdenoised.png' % (opt.noise_dir, opt.input_name[:-4], str(opt.noise)), denoise_img, vmin =0.0, vmax = 1.0)
            print(name, ':', compare_psnr(denoise_img, clean_img))
