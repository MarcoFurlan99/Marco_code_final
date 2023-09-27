import numpy as np
from .utils.functions import create_directories
from tqdm import tqdm
from .utils.perlin_noise import generate_perlin_noise_2d
from PIL import Image
from os import listdir
from random import randint
from scipy.signal import convolve2d

def generate_masks( dest_folder,
                    n_img,
                    size = (64,64),
                    resolution = (2,2)):
    
    """Generate masks.
    dest_folder:    the destination folder;
    n_img:          number of masks to be generated;
    size:           width and height of the output image;
    resolution:     is a Perlin noise parameter."""

    # check if input is correct
    assert dest_folder[-1] == '/', f"please put a slash in the folder name: '{dest_folder}' -> '{dest_folder}/'"
    size = np.array(size)
    res = np.array(resolution)
    assert np.all(size % res == 0), "size must be divisible by resolution"

    # create directories
    create_directories(dest_folder, verbose = False)
    
    for i in tqdm(range(n_img), leave = False):
        
        # this messy code is to make sure the density is homogeneous. In fact, perlin noise
        # has some fix zero-crossing points and we introduce a random shift to randomize
        # their position in the matrix.
        unit_size = size // res
        size_perlin_matrix = unit_size * (res + 1)
        perlin_matrix = generate_perlin_noise_2d(size_perlin_matrix,res + 1)
        shift = np.random.randint(unit_size) # array containing x and y shift
        perlin = perlin_matrix[shift[0]:shift[0]+size[0], shift[1]:shift[1]+size[1]]

        # create the mask
        threshold = 0.4
        mask = (abs(perlin) > threshold)
        
        # save mask in folder
        im = Image.fromarray(mask)
        im.save(dest_folder + str(i) + '.png')

def img_white_noise(mask_folder, dest_folder, parameters = (100,20,150,20), verbose = False):

    """Generate images from masks using white noise.
    mask_folder:    folder containing the masks;
    dest_folder:    destination folder;
    parameters:     noise parameters (mu1, sigma1, mu2, sigma2);
    verbose:        if True will show the progress bar."""

    # check if input is correct
    assert mask_folder[-1] == '/', f"please put a slash in the folder name: '{mask_folder}' -> '{mask_folder}/'"
    assert dest_folder[-1] == '/', f"please put a slash in the folder name: '{dest_folder}' -> '{dest_folder}/'"
    mu1, sigma1, mu2, sigma2 = parameters
    assert mu1<=mu2, "mu1 should not be bigger than mu2!"
    
    # create destination folder
    create_directories(dest_folder, verbose = False)
    
    for file in tqdm(listdir(mask_folder), disable = not verbose):
        # open mask image and convert to matrix
        mask = np.array(Image.open(mask_folder + file))
        
        # generate two random normal noise matrices
        noise_False = np.random.normal(mu1, sigma1, size = mask.shape)
        noise_True  = np.random.normal(mu2, sigma2, size = mask.shape)
        
        # given mask, replace 0s (black) with noise_False and 1s (white) with noise_True
        x = mask * noise_True + (1 - mask)*noise_False
        
        # clip to avoid conversion issues in uint8
        x = np.clip(x,0,255)

        # convert in uint8 format and then to image
        x = x.astype(np.uint8)
        im = Image.fromarray(x).convert('RGB')

        # save image
        im.save(dest_folder + file)

def img_gradient(mask_folder, dest_folder, verbose = False):

    """Generate images from masks using Perlin noise. Perlin noise parameters are set inside the function.
    mask_folder:        folder containing the masks;
    dest_folder:        destination folder;
    verbose:            if True will show the progress bar."""

    # check if input is correct
    assert mask_folder[-1] == '/', f"please put a slash in the folder name: '{mask_folder}' -> '{mask_folder}/'"
    assert dest_folder[-1] == '/', f"please put a slash in the folder name: '{dest_folder}' -> '{dest_folder}/'"

    # create directories
    create_directories(dest_folder, verbose = False)

    # perlin noise parameters
    u = 32 # unit (just for readability)
    r = 2 # resolution

    for file in tqdm(listdir(mask_folder), disable = not verbose):
        # open mask image and convert to matrix
        mask = np.array(Image.open(mask_folder + file))

        # generate two random Perlin noise matrices
        noise_True  = generate_perlin_noise_2d(((r+1)*u,(r+1)*u),(r+1,r+1))
        noise_False = generate_perlin_noise_2d(((r+1)*u,(r+1)*u),(r+1,r+1))

        # we shift so zero-crossings of perlin are in random places, and clip to avoid issues when converting to image
        x_True,  y_True  = np.random.randint(u), np.random.randint(u)
        x_False, y_False = np.random.randint(u), np.random.randint(u)
        noise_True =  128+128*np.clip(noise_True [x_True :x_True +r*u, y_True :y_True +r*u],-1,1)
        noise_False = 128+128*np.clip(noise_False[x_False:x_False+r*u, y_False:y_False+r*u],-1,1)

        # given mask, replace 0s (black) with noise_False and 1s (white) with noise_True
        x = mask * noise_True + (1 - mask)*noise_False

        # convert to image
        im = Image.fromarray(x).convert('RGB')

        # save image
        im.save(dest_folder + file)

def img_edges(mask_folder, dest_folder, verbose = False):

    """Generate images from masks convolving with an edge/corner detection filter.
    mask_folder:        folder containing the masks;
    dest_folder:        destination folder;
    verbose:            if True will show the progress bar."""

    # check if input is correct
    assert mask_folder[-1] == '/', f"please put a slash in the folder name: '{mask_folder}' -> '{mask_folder}/'"
    assert dest_folder[-1] == '/', f"please put a slash in the folder name: '{dest_folder}' -> '{dest_folder}/'"

    # create directory
    create_directories(dest_folder, verbose = False)

    # define filter
    filter = np.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
    
    for file in tqdm(listdir(mask_folder), disable = not verbose):
        # open mask image and convert to matrix
        mask = np.array(Image.open(mask_folder + file))

        # convolve with filter, shift all values to non-negative (+8) and rescale (*16)
        x = ((convolve2d(mask, filter, mode='same') + 8)*16)

        # clip to avoid conversion issues in uint8, and convert to uint8
        x = np.clip(x,0,255).astype(np.uint8)
        
        # convert to image
        im = Image.fromarray(x).convert('RGB')

        # save image
        im.save(dest_folder + file)
