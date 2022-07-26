import os
from options.test_options import TestOptions
# from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import util
from data.base_dataset import BaseDataset, get_transform
from PIL import Image

def save_images(visuals, image_dir, name, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = '%s/%s.png' % (label, name)
        os.makedirs(os.path.join(image_dir, label), exist_ok=True)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        print('done saving at %s' % (save_path))



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options

    opt.num_test = opt.num_test if opt.num_test > 0 else float("inf")
    # data processing
    transform_fn = get_transform(opt, grayscale=(opt.input_nc == 1))
    A_path = './demo_img/h2z/test_h2z_2.jpeg'
    A_img = Image.open(A_path).convert('RGB')
    A = transform_fn(A_img)
    print(A.size())

    data = {'A': A.unsqueeze(0), 'B': A.unsqueeze(0), 'A_paths': A_path}
    # testing model
    model.data_dependent_initialize(data)
    model.setup(opt)
    model.parallelize()
    if opt.eval:
        model.eval()

    

    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    img_path = model.get_image_paths()     # get image paths

    save_dir = './demo_img/h2z/'
    save_name = 'output_h2z_2'
    save_images(visuals, save_dir, save_name, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    # webpage.save()  # save the HTML
