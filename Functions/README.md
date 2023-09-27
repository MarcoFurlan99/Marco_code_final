# Functions

This folder contains a single file containing general utility functions. Specifically:

***Folders***

- create_directories: to create directory path.[^1]

- remove_directory: to remove directory and its contents (please use carefully!).[^2]

***IoU***

-IoU: compute IoU between two images.

-avg_IoU: compute average IoU between two folders of images.

***Wasserstein/Fréchet distance***

- calculate_frechet_distance: compute Fréchet distance between two multivariate normal distributions.[^3]

- calculate_source_normalized_frechet_distance: compute source-normalized Fréchet distance between two multivariate normal distributions.[^4]

***Graph***

- graph_2d: essentially the function matplotlib.pyplot.imshow with more parameters to control it.

For more details read the functions descriptions (and try them yourself!)

[^1]: basically os.makedirs
[^2]: basically shutil.rmtree
[^3]: see [https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152)
[^4]: see pag.15 of [https://arxiv.org/abs/2006.16971](https://arxiv.org/abs/2006.16971). No promises on the computational stability of this implementation, often the imaginary component which arises inevitably from taking square root matrices is out of hand.
