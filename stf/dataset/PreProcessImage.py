import numpy as np
import cv2
import random
import math


class PreProcessIm(object):
    def __init__(
        self,
        crop_prob=0,
        crop_ratio=1.0,
        resize_h_w=None,
        scale=True,
        im_mean=None,
        im_std=None,
        mirror_type=None,
        batch_dims='NCHW',
        prng=np.random,
        # for random erasing
        random_erasing_prob=0,
        sl=0.02,
        sh=0.4,
        r1=0.25,
        fix_hw_rate=False,
        image_save=False,
        hsv_jitter_prob=0,
        hsv_jitter_range=[50, 20, 40],
        gaussian_blur_prob=0,
        gaussian_blur_kernel=7,
        horizontal_crop_prob=0,
            horizontal_crop_ratio=0.4):
        """
        Args:
          crop_prob: the probability of each image to go through cropping
          crop_ratio: a float. If == 1.0, no cropping.
          resize_h_w: (height, width) after resizing. If `None`, no resizing.
          scale: whether to scale the pixel value by 1/255
          im_mean: (Optionally) subtracting image mean; `None` or a tuple or list or
            numpy array with shape [3]
          im_std: (Optionally) divided by image std; `None` or a tuple or list or
            numpy array with shape [3]. Dividing is applied only when subtracting
            mean is applied.
          mirror_type: How image should be mirrored; one of
            [None, 'random', 'always']
          batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels,
            'H': im height, 'W': im width. PyTorch uses 'NCHW', while TensorFlow
            uses 'NHWC'.
          prng: can be set to a numpy.random.RandomState object, in order to have
            random seed independent from the global one
        """
        self.crop_prob = crop_prob
        self.crop_ratio = crop_ratio
        self.resize_h_w = resize_h_w
        self.scale = scale
        self.im_mean = im_mean
        self.im_std = im_std
        self.check_mirror_type(mirror_type)
        self.mirror_type = mirror_type
        self.check_batch_dims(batch_dims)
        self.batch_dims = batch_dims
        self.prng = prng
        self.random_erasing_prob = random_erasing_prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.fix_hw_rate = fix_hw_rate
        self.image_save = image_save
        self.hsv_jitter_prob = hsv_jitter_prob
        self.hsv_jitter_range = hsv_jitter_range
        self.gaussian_blur_prob = gaussian_blur_prob
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.horizontal_crop_prob = horizontal_crop_prob
        self.horizontal_crop_ratio = horizontal_crop_ratio

    def __call__(self, im):
        return self.pre_process_im(im)

    @staticmethod
    def check_mirror_type(mirror_type):
        assert mirror_type in [None, 'random', 'always']

    @staticmethod
    def check_batch_dims(batch_dims):
        # 'N': batch size, 'C': num channels, 'H': im height, 'W': im width
        # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
        assert batch_dims in ['NCHW', 'NHWC']

    def set_mirror_type(self, mirror_type):
        self.check_mirror_type(mirror_type)
        self.mirror_type = mirror_type

    @staticmethod
    def rand_crop_im(im, padding_size, prng=np.random):
        """Crop `im` to `new_size`: [new_w, new_h]."""

        new_h = im.shape[0] + padding_size[1] * 2
        new_w = im.shape[1] + padding_size[0] * 2
        # print new_h, new_w, im.shape[0], im.shape[1]

        h_start = prng.randint(0, new_h - im.shape[0] + 1)
        w_start = prng.randint(0, new_w - im.shape[1] + 1)
        # print h_start, w_start
        im_mean = np.mean(im.flatten())
        new_im = np.pad(im, ((padding_size[1], padding_size[1]), (
            padding_size[0], padding_size[0]), (0, 0)), 'constant', constant_values=im_mean)
        new_im = new_im[h_start: h_start + im.shape[0],
                        w_start: w_start + im.shape[1]]
        # print new_im.shape, im.shape
        cv2.imwrite('debug/debug%d.jpg' % random.randrange(0, 1000), new_im)
        return new_im

    # padding image to 3:1 before resizing

    def fix_rate(self, img):
        #print ("fix rate func!!!")
        h = img.shape[0]
        w = img.shape[1]
        # print "h:%d,w:%d" % (h, w)
        if float(h)/float(w) <= 3:
            out_img = np.random.uniform(0, 1, size=(3*w, w, 3))
            out_img[0:3*w, 0:w, 0] = self.im_mean[0]*255
            out_img[0:3*w, 0:w, 1] = self.im_mean[1]*255
            out_img[0:3*w, 0:w, 2] = self.im_mean[2]*255
        else:
            out_img = np.random.uniform(0, 1, size=(h, int(h/3), 3))
            out_img[0:h, 0:int(h/3), 0] = self.im_mean[0]*255
            out_img[0:h, 0:int(h/3), 1] = self.im_mean[1]*255
            out_img[0:h, 0:int(h/3), 2] = self.im_mean[2]*255
        h_o = out_img.shape[0]
        w_o = out_img.shape[1]
        delta_h = (h_o - h)/2
        delta_w = (w_o - w)/2
        # print "delta_h: %d,delta_w: %d" % (delta_h,delta_w)
        # print "h_o:%d w_o: %d" % (h_o,w_o)
        out_img[delta_h:h+delta_h, delta_w:w+delta_w, :] = img
        return out_img

    def random_erasing(self, img):
        if random.uniform(0, 1) > self.random_erasing_prob:
            return img
        for attempt in range(100):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(1.5, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            # print ("random erasing: ", img.shape)
            #rate = 1
            if w < img.shape[1] and h < img.shape[0]:
                x1 = random.randint(0, img.shape[0] - h)
                y1 = random.randint(0, img.shape[1] - w)
                if img.shape[2] == 3:
                    img[x1:x1+h, y1:y1+w, 0] = self.im_mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.im_mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.im_mean[2]
                else:
                    img[x1:x1+h, y1:y1+w, 0] = self.im_mean[0]
                return img
        return img

    def hsv_jitter(self, im, saturation_range=50, hue_range=20, value_range=40):
        im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int)
        # saturation
        if saturation_range > 0:
            offset = np.random.randint(- saturation_range, saturation_range)
            # print offset
            im_hsv[:, :, 1] = im_hsv[:, :, 1] + offset

        # hue
        if hue_range > 0:
            offset = np.random.randint(- hue_range, hue_range)
            # print offset
            im_hsv[:, :, 0] = im_hsv[:, :, 0] + offset

        # value
        if value_range > 0:
            offset = np.random.randint(- value_range - 30, value_range)
            # print offset
            im_hsv[:, :, 2] = im_hsv[:, :, 2] + offset

        im_hsv = np.clip(im_hsv, 0, 255).astype(np.uint8)
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        return im_rgb

    def pre_process_im(self, im):
        """Pre-process image.
        `im` is a numpy array with shape [H, W, 3], e.g. the result of
        matplotlib.pyplot.imread(some_im_path), or
        numpy.asarray(PIL.Image.open(some_im_path))."""
        # original image saving
        if im.shape[0] * 1.0 / im.shape[1] > 1.5:
            save_img = True
        else:
            save_img = False

        if self.image_save and save_img:
            img_root = 'debug/'
            num_ber = self.prng.randint(0, 1000)
            image_path = img_root + "06%dori.jpg" % num_ber
            cv2.imwrite(image_path, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

        # Randomly crop a sub-image.
        if ((self.crop_ratio < 1)
            and (self.crop_prob > 0)
                and (self.prng.uniform() < self.crop_prob)):
            h_ratio = self.prng.uniform(0, self.crop_ratio)
            w_ratio = self.prng.uniform(0, self.crop_ratio)
            # print 'ratio', h_ratio, w_ratio
            padding_h = int(im.shape[0] * h_ratio)
            padding_w = int(im.shape[1] * w_ratio)
            im = self.rand_crop_im(im, (padding_w, padding_h), prng=self.prng)

        # Horizontal Crop
        if ((self.horizontal_crop_ratio < 1)
            and (self.horizontal_crop_prob > 0)
            and (self.prng.uniform() < self.horizontal_crop_prob)
                and im.shape[0] * 1.0 / im.shape[1] > 1.5):
            # print self.horizontal_crop_ratio, self.horizontal_crop_prob, 'fooo'
            h_ratio = self.prng.uniform(self.horizontal_crop_ratio, 1)
            crop_h = int(im.shape[0] * h_ratio)
            im = im[0:crop_h]

        # Fix image to 3:1 ori_image_size_rate
        if (self.fix_hw_rate):
            im = self.fix_rate(im)

        # Resize.
        if (self.resize_h_w is not None) \
                and (self.resize_h_w != (im.shape[0], im.shape[1])):
            im = cv2.resize(
                im, self.resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
        # print "say yeah yeah yeah"
        # scaled by 1/255.

        # color jitering
        if ((self.hsv_jitter_prob > 0)
                and (self.prng.uniform() < self.hsv_jitter_prob)):
            im = self.hsv_jitter(
                im, self.hsv_jitter_range[0], self.hsv_jitter_range[1], self.hsv_jitter_range[2])

        # blur
        # print self.gaussian_blur_prob, self.gaussian_blur_kernel, 'fo'
        if self.gaussian_blur_prob > 0 and (self.prng.uniform() < self.gaussian_blur_prob):
            sizes = range(1, self.gaussian_blur_kernel, 2)
            kernel_size = random.sample(sizes, 1)[0]
#      print kernel_size
            im = cv2.GaussianBlur(im, (kernel_size, kernel_size), 0)

        if self.scale:
            im = im / 255.

        # May random erasing
        if (self.random_erasing_prob > 0):
            im = self.random_erasing(im)

        # Subtract mean and scaled by std
        # im -= np.array(self.im_mean) # This causes an error:
        # Cannot cast ufunc subtract output from dtype('float64') to
        # dtype('uint8') with casting rule 'same_kind'
        if self.im_mean is not None:
            im = im - np.array(self.im_mean)

        if self.im_mean is not None and self.im_std is not None:
            im = im / np.array(self.im_std).astype(float)

        # versualize the images
        if self.image_save and save_img:
            print("save_images")
            img_root = 'debug/'
            # num_ber = self.prng.randint(0, 1000)
            image_path = img_root + "06%d.jpg" % num_ber
            save_img = ((im * np.array(self.im_std).astype(float)
                         ) + np.array(self.im_mean))*255
            print save_img.shape
            cv2.imwrite(image_path, cv2.cvtColor(
                save_img.astype(np.uint8), cv2.COLOR_RGB2BGR))

        # May mirror image.
        mirrored = False
        if self.mirror_type == 'always' \
                or (self.mirror_type == 'random' and self.prng.uniform() > 0.5):
            im = im[:, ::-1, :]
            mirrored = True

        # May random erasing
        # if (self.random_erasing_prob > 0):
            # print ("random erasing!!!")
            #im = self.random_erasing(im)

        # The original image has dims 'HWC', transform it to 'CHW'.
        if self.batch_dims == 'NCHW':
            im = im.transpose(2, 0, 1)

        return im, mirrored
