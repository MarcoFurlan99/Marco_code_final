# Functions

This folder contains a single file containing general utility functions. Specifically:

***Folders***

- create_directories[^1]: to create directory path.

- remove_directory[^2]: to remove directory and its contents (please use carefully!).

***IoU***

-IoU: compute IoU between two images.

-avg_IoU: compute average IoU between two folders of images.

***Wasserstein/Fréchet distance***

- calculate_frechet_distance[^3]: compute Fréchet distance between two multivariate normal distributions.

- calculate_source_normalized_frechet_distance: compute source-normalized Fréchet distance between two multivariate normal distributions.

***Graph***

- graph_2d: essentially the function matplotlib.pyplot.imshow with more parameters to control it.

For more details read the functions descriptions (and try them yourself!)

[^1]: basically os.makedirs
[^2]: basically shutil.rmtree
[^3]: see [https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py#L152)
