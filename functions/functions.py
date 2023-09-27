from os import makedirs
from os.path import isdir
import shutil
import numpy as np
from PIL import Image
from scipy import linalg

###############
### FOLDERS ###
###############

def create_directories(folder, verbose = True):
    """Create folder and full folder path. If already existing will do nothing.
    folder:     folder path;
    verbose:    if True will print a message if folder is created successfully."""
    try:
        makedirs(folder)
        if verbose: print(f'created folder {folder}')
    except:
        None

def remove_directory(folder, ask = True, verbose = True):
    """Removes the folder 'folder' and all its contents, if such a folder exists. Use carefully.
    Returns True if folder was successfully removed (or wasn't there), False otherwise.
    ask:        if True will ask confirmation in the terminal before removing;
    verbose:    if True will print a message if folder was removed or if it was not there. """
    if isdir(folder):
        if ask:
            input_ = str(input('\nYou want to delete the folder "'+folder+'" and all its contents? [y/N] '))
        if not ask or input_ == 'y' or input_ == 'Y':
            try:
                shutil.rmtree(folder)
                removed_ = True
            except:
                assert True, "Error: it was not possible to remove the folder. This message should not appear in any situation, check carefully what went wrong."
        else:
            removed_ = False
    else:
        removed_ = True # folder was not there
    
    if removed_ and verbose:    print('\nThe folder "'+folder+'" was removed!\n')
    return removed_

###########
### IOU ###
###########

from os.path import join
from os import listdir

def IoU(mask1, mask2, value):
    """Compute pixel-wise IoU between two black and white images.
    mask1:  path to first image;
    mask2:  path to second image;
    value:  value to consider as mask. Set to either 1 or 255 for white, and to 0 for black."""
    im1 = Image.open(mask1).convert('L')
    im2 = Image.open(mask2).convert('L')
    matrix1 = np.array(im1) == value
    matrix2 = np.array(im2) == value
    intersection = np.sum(np.logical_and(matrix1,matrix2))
    union = np.sum(np.logical_or(matrix1,matrix2))
    if union == 0:
        return 0.0
    return intersection/union

def avg_IoU(folder1, folder2, value):
    """Compute average IoU between two folders of black and white images. Corresponding images should have the same file names.
    folder1:    path to first folder;
    folder2:    path to second folder;
    value:      value to consider as mask. Set to either 1 or 255 for white, and to 0 for black."""
    IoU_list = []
    for file in listdir(folder1):
        IoU_list.append(IoU(join(folder1, file),join(folder2, file), value))
    return np.mean(IoU_list)

###################
### WASSERSTEIN ###
###################

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6): # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-1): ## I changed the tolerance
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_source_normalized_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """My implementation of the source-normalized Wasserstein distance."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu2 - mu1

    sigma1_inv = linalg.inv(sigma1) # unique inverse matrix
    product = sigma1_inv @ sigma2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(product, disp=False)
    
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-1): ## I don't make any promises for the computational stability of this code
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff @ sigma1_inv @ diff + len(mu1) # len(mu1) == trace(I)
            + np.trace(product) - 2 * tr_covmean)


#############
### GRAPH ###
#############

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

def graph_2d(z,
             save = False,
             xlabel = '',
             ylabel = '',
             xyticks = None,
             title = '',
             show = False,
             vmin = 0,
             vmax = 1,
             cmap = None):
    """Just ignore this and look up matplotlib.pyplot.imshow: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
    Otherwise try:
    graph_2d(np.array([[0.5,0.7],[0.2,0.1]]), xlabel = 'x axis', ylabel = 'y axis', xyticks = ['a','b'], title = 'title', show = True)
    """
    m, n = z.shape
    if cmap:
        norm = Normalize(vmin = vmin, vmax = vmax)
        cmap = cmap
    else:
        norm = Normalize(vmin = vmin, vmax = vmax)
        cmap = LinearSegmentedColormap.from_list('my_colormap', ['red','yellow','green'])
    plt.imshow(z,interpolation='nearest', cmap = cmap, norm = norm)
    # plt.colorbar(fig, cmap=cmap)

    plt.title(title, fontsize = 20)
    # adjust ticks
    plt.xticks(ticks = range(n), labels = xyticks, rotation = 90)
    plt.yticks(ticks = range(m), labels = xyticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # add values in cells
    for (j,i),label in np.ndenumerate(np.around(z,2)):
        plt.text(i,j,label,ha='center',va='center')
    
    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()




