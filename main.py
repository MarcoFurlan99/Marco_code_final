
#####################
### STEP 0: SETUP ###
#####################

from Functions.functions import create_directories, remove_directory

print_step = lambda i, text: print(f"\n> STEP {i} - \t{text}")

main_dir = 'data/'

remove_directory(main_dir)

n_train = 1000 # number of train images
n_test = 100 # number of test images
classes = 2

parameters_list_source = [  (96,18,138,18) ]

mu1_list = [64,96,128,160]
mu2_add_list = [24,42,73,128]
sigma1_list = [6,18,30]
sigma2_list = [6,18,30]

parameters_list_target = [(mu1,sigma1,mu2,sigma2)
                            for sigma1 in sigma1_list
                            for sigma2 in sigma2_list
                            for mu1 in mu1_list
                            for mu2 in [mu1 + diff for diff in mu2_add_list if mu1 + diff <= 192]
                            ]

n_mu1 = len(mu1_list)
n_mu2_intervals = [4,3,2,1]
assert n_mu1 == len(n_mu2_intervals), "something is wrong"
n_sigma1 = len(sigma1_list)
n_sigma2 = len(sigma2_list)

n_s = len(parameters_list_source)
n_t = len(parameters_list_target)

labels_source = [str(p) for p in parameters_list_source]
labels_target = [str(p) for p in parameters_list_target]


##############################
## STEP 1: CREATE DATASETS ###
##############################
# print_step(1,"Creating datasets")

# from Dataset_generators.dataset_perlin_noise import generate_masks, img_white_noise
# from tqdm import tqdm

# # create directory data_toydataset
# create_directories(main_dir, verbose=False)

# # save parameter list
# with open(main_dir + 'parameters_list.txt','w') as prprprprprp:
#     prprprprprp.write(f'source:\n{parameters_list_source}\n\ntarget:\n{parameters_list_target}')

# # create source datasets
# for i,parameters in enumerate(parameters_list_source):
#     masks_folder  = main_dir + f'source{i}/label/'
#     images_folder = main_dir + f'source{i}/img/'
    
#     generate_masks(masks_folder, n_train)
#     img_white_noise(masks_folder, images_folder, parameters)

# # create target datasets
# for i,parameters in enumerate(tqdm(parameters_list_target, desc = f'generating target datasets')):
#     masks_folder  = main_dir + f'target{i}/label/'
#     images_folder = main_dir + f'target{i}/img/'
    
#     generate_masks(masks_folder, n_test)
#     img_white_noise(masks_folder, images_folder, parameters)


###################################
### STEP 2: TRAIN, PREDICT, IOU ###
###################################
# print_step(2,"training, prediction and IoU")

# from UNet.train import train
# from UNet.predict import predict
# from os import listdir, rename
# from os.path import join
# from Functions.functions import avg_IoU
# from tqdm import tqdm
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap

# i = 0

# ### TRAINING ###

# plt_losses = []

# train_img =         f'./{main_dir}source{i}/img/'
# train_mask =        f'./{main_dir}source{i}/label/'
# train_model =       f'./{main_dir}source{i}/MODEL{i}.pth'

# print(f'training\t({i+1}/{n_s})')

# plt_loss= train(train_img,
#                 train_mask,
#                 train_model)
# plt_losses.append(plt_loss)

# # save training history
# with open(main_dir + "training_history.txt",'w') as training_history:
#     training_history.write(str(plt_losses))

# cmap = LinearSegmentedColormap.from_list('my_colormap', ['red', 'green'])
# colors_indexes = np.linspace(0,256,len(plt_losses), dtype = np.int0)
# for i_, plt_loss in enumerate(plt_losses):
#     plt.plot(plt_loss, color = cmap(colors_indexes[i_]))


# ### PREDICTING ###

# with tqdm(total = n_t) as pbar:
#     for j in range(n_t):
#         pred_input =        main_dir + f'target{j}/img/'
#         pred_output =       main_dir + f'target{j}/pred{i}/'
#         true_masks =        main_dir + f'target{j}/label/'

#         pred_input_files  = sorted([join(pred_input,  file) for file in listdir(pred_input)])
#         pred_output_files = sorted([join(pred_output, file) for file in listdir(pred_input)])
        
#         create_directories(pred_output, verbose = False)

#         predict(input = pred_input_files,
#                 output = pred_output_files,
#                 model = main_dir + f"source{i}/MODEL{i}.pth",
#                 classes = classes
#                 )

#         pbar.update(1)
#         pbar.set_description(f'MODEL{i}.pth is predicting target{j}')
        
#         torch.cuda.empty_cache() # empty torch cache


# ### IOU ###

# z = np.zeros((n_s,n_t))

# for j in tqdm(range(n_t)):
#     pred_input =        main_dir + f'target{j}/img/'
#     pred_output =       main_dir + f'target{j}/pred{i}/'
#     true_masks =        main_dir + f'target{j}/label/'

#     v = avg_IoU(pred_output, true_masks, 255)
#     z[(i,j)] = v
#     print(v)


# plt.title("Validation losses")
# plt.legend(labels_source)
# plt.yscale("log")
# plt.savefig(main_dir + "training_history.png")
# plt.close()

# # save IoU data
# np.savetxt(main_dir + 'graph_2d.txt',z)


#############################
### STEP 3: BN ADAPTATION ###
#############################
print_step(3, "Applying the BN adaptation, predicting and saving the IoU")

from tqdm import tqdm
from os import cpu_count, listdir
from os.path import join
from UNet.utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from Functions.functions import avg_IoU
from UNet.utils.BN_adapt import BN_adapt
from itertools import product
from UNet.predict import predict
import numpy as np
import torch

create_directories(main_dir + 'MODELS_adapted/')

ij_list = list(product(range(n_s), range(n_t))) # this is just so tqdm scrolls through all the iterations

z_adapted = np.zeros((n_s,n_t))

for i,j in tqdm(ij_list):

    original_model =    main_dir + f'source{i}/MODEL{i}.pth'
    pred_input =        main_dir + f'target{j}/img/'
    true_masks =        main_dir + f'target{j}/label/'
    adapted_model =     main_dir + f'MODELS_adapted/MODEL{i}{j}.pth'

    dataset = BasicDataset(pred_input, true_masks, scale=1.0)
    loader_args = dict(batch_size=1, num_workers=0, pin_memory=True) ## num_workers = cpu_count()
    dataset_loader = DataLoader(dataset, shuffle=True, **loader_args)
    BN_adapt(original_model, dataset_loader, 'cuda', adapted_model)
    torch.cuda.empty_cache()

    pred_output =       main_dir + f'target{j}/pred_adapted{i}/'
    pred_input_files  = sorted([join(pred_input,  file) for file in listdir(pred_input)])
    pred_output_files = sorted([join(pred_output, file) for file in listdir(pred_input)])

    # remove_directory(pred_output, ask = False, verbose = False)
    create_directories(pred_output, verbose = False)

    predict(input = pred_input_files,
            output = pred_output_files,
            model = main_dir + f"MODELS_adapted/MODEL{i}{j}.pth",
            classes = classes
            )

    v = avg_IoU(pred_output, true_masks, 255)
    z_adapted[(i,j)] = v

np.savetxt(main_dir + 'graph_2d_adapted.txt',z_adapted)

raise e


# print_step(7,"draw graph")
# #####################
# ### STEP ?: GRAPH ###
# #####################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, Normalize

# z = np.loadtxt(main_dir + 'graph_2d.txt')
# z_adapted = np.loadtxt(main_dir + 'graph_2d_adapted.txt')

# z_list = [z, z_adapted, z_adapted-z]
# z_names = ['graph_2d.png', 'graph_2d_adapted.png', 'graph_2d_diff.png']

# for i_, zz in enumerate(z_list):
#     w = np.zeros((n_mu1,max(n_mu2_intervals)))
#     w.fill(np.NaN)
#     indexes = [(i,j) for i in range(n_mu1) for j in range(n_mu2_intervals[i])]
#     plt.rcParams['figure.figsize'] = [24, 20]

#     fig, axs = plt.subplots(n_sigma1, n_sigma2)
#     if i_<2:
#         norm = Normalize(vmin = 0, vmax = 1)
#         cmap = LinearSegmentedColormap.from_list('my_colormap', ['red','yellow','green'])
#     if i_ == 2:
#         norm = Normalize(vmin = -1, vmax = 1)
#         cmap = LinearSegmentedColormap.from_list('my_colormap', ['red','gray','cyan'])

#     k = 0
#     for i_sigma1 in range(n_sigma1):
#         for i_sigma2 in range(n_sigma2):
#             for index in indexes:
#                 w[index] = zz[k]
#                 k+=1
#             # add imshow graph
#             axs[i_sigma1][i_sigma2].imshow(w,
#                                     interpolation='nearest',
#                                     cmap = cmap,
#                                     norm = norm,
#                                     )
            
#             # add sigmas on rows and columns
#             if i_sigma2 == 0:
#                 axs[i_sigma1][i_sigma2].set_ylabel(f'$\sigma_1 = {sigma1_list[i_sigma1]}$')
#             if i_sigma1 == 0:
#                 axs[i_sigma1][i_sigma2].set_title( f'$\sigma_2 = {sigma1_list[i_sigma2]}$')

#             # add numbers in cells
#             for (j,i),label in np.ndenumerate(np.around(w,2)):
#                 axs[i_sigma1][i_sigma2].text(i,j,label,ha='center',va='center')
    
#     plt.setp(axs,
#             xticks = range(max(n_mu2_intervals)),
#             yticks = range(len(mu1_list)),
#             xticklabels = [f'$+{diff}$' for diff in mu2_add_list],
#             yticklabels = [f'$\mu_1={mu1}$' for mu1 in mu1_list],
#             )
    
#     plt.suptitle(f'Source dataset: {parameters_list_source[0]}', size = 'xx-large')
#     plt.savefig(main_dir + f'{z_names[i_]}')
#     # plt.show()
#     plt.close()


# print_step(8,"sigma graph")
# ###########################
# ### STEP Q: SIGMA GRAPH ###
# ###########################
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import LinearSegmentedColormap, Normalize

# z = np.loadtxt(main_dir + 'graph_2d.txt')
# z_adapted = np.loadtxt(main_dir + 'graph_2d_adapted.txt')

# z_list = [z, z_adapted, z_adapted-z]
# z_names = ['graph_2d_sigma.png', 'graph_2d_adapted_sigma.png', 'graph_2d_diff_sigma.png']

# def reorder_by_mu(z):
#     out = np.zeros((n_sigma1, n_sigma2, n_mu1, max(n_mu2_intervals)))
#     out.fill(np.nan)
#     m = 0
#     for i in range(n_sigma1):
#         for j in range(n_sigma2):
#             for k in range(n_mu1):
#                 for l in range(n_mu2_intervals[k]):
#                     out[i,j,k,l] = z[m]
#                     m += 1
#     return out

# for i_, zz in enumerate(z_list):
#     plt.rcParams['figure.figsize'] = [24, 20]

#     fig, axs = plt.subplots(n_mu1, max(n_mu2_intervals))

#     if i_<2:
#         norm = Normalize(vmin = 0, vmax = 1)
#         cmap = LinearSegmentedColormap.from_list('my_colormap', ['red','yellow','green'])
#     if i_ == 2:
#         norm = Normalize(vmin = -1, vmax = 1)
#         cmap = LinearSegmentedColormap.from_list('my_colormap', ['red','gray','cyan'])

#     k = 0
#     z_ = reorder_by_mu(zz)

#     for i_mu1 in range(n_mu1):
#         for i_mu2s in range(max(n_mu2_intervals)):
#             w = z_[:,:,i_mu1, i_mu2s]
#             # add imshow graph
#             axs[i_mu1][i_mu2s].imshow(w,
#                                     interpolation='nearest',
#                                     cmap = cmap,
#                                     norm = norm,
#                                     )
            
#             # add sigmas on rows and columns
#             if i_mu2s == 0:
#                 axs[i_mu1][i_mu2s].set_ylabel(f'$\mu_1 = {mu1_list[i_mu1]}$')
#             if i_mu1 == 0:
#                 axs[i_mu1][i_mu2s].set_title( f'$\mu_2 = \mu_1 + {mu2_add_list[i_mu2s]}$')

#             # add numbers in cells
#             for (j,i),label in np.ndenumerate(np.around(w,2)):
#                 axs[i_mu1][i_mu2s].text(i,j,label,ha='center',va='center')
    
#     plt.setp(axs,
#             xticks = range(len(sigma2_list)),
#             yticks = range(len(sigma1_list)),
#             xticklabels = [f'${sigma2}$' for sigma2 in sigma2_list],
#             yticklabels = [f'$\sigma_1 = {sigma1}$' for sigma1 in sigma1_list],
#             )
    
#     plt.suptitle(f'Source dataset: {parameters_list_source[0]}', size = 'xx-large')
#     plt.savefig(main_dir + f'{z_names[i_]}')
#     # plt.show()
#     plt.close()



##########################################
## STEP ??: GET LATENT SPACES (SOURCE) ###
##########################################

# print_step('??', "get latent spaces (source without BN adapt)")

# from functions.functions import get_latent_space
# from UNet.unet.unet_model import UNet
# from UNet.utils.data_loading import BasicDataset
# from torch.utils.data import DataLoader
# import torch
# from itertools import product
# from tqdm import tqdm

which_ls_lst = [0]

# for which_ls in tqdm(which_ls_lst, desc = f'getting source latent spaces'):
    
#     create_directories(main_dir + f'ls_{which_ls}_source/')

#     img_folder =     main_dir + f'source0/img/'
#     masks_folder =   main_dir + f'source0/label/'
#     original_model = main_dir + f'source0/MODEL0.pth'

#     dataset = BasicDataset(img_folder, masks_folder, scale=1.0)
#     loader_args = dict(batch_size=1000, num_workers=0, pin_memory=True) # only first batch is computed! (first latent space goes on memory error with full batch size)
#     source_dataset = DataLoader(dataset, shuffle=True, **loader_args)
    
#     ls = get_latent_space(source_dataset, original_model, device = 'cuda', which_ls = which_ls)
#     print('size:',ls.shape)
#     torch.save(ls, main_dir + f'ls_{which_ls}_source/ls_source.pt')

#     del dataset
#     del source_dataset
#     del ls

###############################################
### STEP 10: GET LATENT SPACES (WITH ADAPT) ###
###############################################

# print_step(10, "get latent spaces (with BN adapt)")

# from functions.functions import get_latent_space
# from UNet.unet.unet_model import UNet
# from UNet.utils.data_loading import BasicDataset
# from torch.utils.data import DataLoader
# import torch
# from itertools import product
# from tqdm import tqdm

# lst = product(range(n_s), range(n_t))
# print(lst)
# for which_ls in which_ls_lst:

#     create_directories(main_dir + f'ls_{which_ls}_adapted/')

#     for source, target in tqdm(list(lst), desc = 'getting latent spaces - target'):
        
#         img_folder =     main_dir + f'target{target}/img/'
#         masks_folder =   main_dir + f'target{target}/label/'
#         original_model = main_dir + f'MODELS_adapted/MODEL{source}{target}.pth'

#         dataset = BasicDataset(img_folder, masks_folder, scale=1.0)
#         loader_args = dict(batch_size=500, num_workers=0, pin_memory=True) # only first batch is computed!
#         source_dataset = DataLoader(dataset, shuffle=True, **loader_args)

#         ls = get_latent_space(source_dataset, original_model, device = 'cuda', which_ls = which_ls)
#         torch.save(ls, main_dir + f'ls_{which_ls}_adapted/ls_adapted_{source}{target}.pt')

#     del dataset
#     del source_dataset
#     del ls
    
import numpy as np
from os.path import isfile

###############
### FRECHET ###
###############
print_step('?', 'frechet')



for which_ls in which_ls_lst:
    
    directory = main_dir + f'wasserstein_{which_ls}/'
    from os import listdir
    import torch
    from functions.functions import calculate_frechet_distance, calculate_source_normalized_frechet_distance
    from tqdm import tqdm

    create_directories(directory)

    ls_source = torch.load(main_dir + f'ls_{which_ls}_source/ls_source.pt')

    for file in tqdm(listdir(main_dir + f'ls_{which_ls}_adapted/'), desc = 'calculating Wasserstein'):
        if isfile(directory + 'w' + file[10:-3] + '.pt'):
            continue
        ls_adapted = torch.load(main_dir + f'ls_{which_ls}_adapted/' + file)

        mu1 = ls_source.mean(dim = 0).numpy()
        mu2 = ls_adapted.mean(dim = 0).numpy()
        
        sigma1 = torch.cov(ls_source.T).numpy()
        sigma2 = torch.cov(ls_adapted.T).numpy()

        w = calculate_frechet_distance(mu2, sigma2, mu1, sigma1)
        torch.save(w, directory + 'w' + file[10:-3] + '.pt')


#############################
### STEP ???: WASSERSWEIN ###
#############################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import torch
from os import listdir

for which_ls in which_ls_lst:
    directory = main_dir + f'wasserstein_{which_ls}/'
    save_filename = main_dir + f'wasserstein_{which_ls}.png'

    n = len(listdir(directory))
    z = np.zeros(n)
    for i in range(n):
        z[i] = torch.load(directory + f'w_0{i}.pt')
    z2 = np.loadtxt(main_dir + 'graph_2d.txt')
    
    w = np.zeros((n_mu1,max(n_mu2_intervals)))
    w.fill(np.NaN)
    indexes = [(i,j) for i in range(n_mu1) for j in range(n_mu2_intervals[i])]
    plt.rcParams['figure.figsize'] = [14, 12]

    fig, axs = plt.subplots(n_sigma1, n_sigma2)
    norm = Normalize(vmin = 0, vmax = float(np.max(z)))
    cmap = LinearSegmentedColormap.from_list('my_colormap', ['white','violet'])
    k = 0
    for i_sigma1 in range(n_sigma1):
        for i_sigma2 in range(n_sigma2):
            for index in indexes:
                w[index] = z[k]
                k+=1
            # add imshow graph
            axs[i_sigma1][i_sigma2].imshow(w,
                                    interpolation='nearest',
                                    cmap = cmap,
                                    norm = norm,
                                    )
            
            # add sigmas on rows and columns
            if i_sigma2 == 0:
                axs[i_sigma1][i_sigma2].set_ylabel(f'$\sigma_1 = {sigma1_list[i_sigma1]}$')
            if i_sigma1 == 0:
                axs[i_sigma1][i_sigma2].set_title( f'$\sigma_2 = {sigma1_list[i_sigma2]}$')

            # add numbers in cells
            for (j,i),label in np.ndenumerate(np.around(w,2)):
                axs[i_sigma1][i_sigma2].text(i,j,label,ha='center',va='center')
    
    plt.setp(axs,
            xticks = range(max(n_mu2_intervals)),
            yticks = range(len(mu1_list)),
            xticklabels = [f'$+{diff}$' for diff in mu2_add_list],
            yticklabels = [f'$\mu_1={mu1}$' for mu1 in mu1_list],
            )
    
    plt.suptitle(f'Source dataset: {parameters_list_source[0]}, ls = {which_ls}', size = 'xx-large')
    plt.savefig(save_filename)
    # plt.show()
    plt.close()

raise LOLOLOLOLOL

########################################
### STEP 11: HISTOGRAMS (WITH ADAPT) ###
########################################
print_step(11, "histograms (with BN adapt)")



import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from itertools import product

dim = 0
bins = np.linspace(a,b,(b-a)*8+1)

lst = product(range(n_s), range(n_t))

folder = f'data_toydataset/histograms_grid/'
create_directories(folder, verbose = False)

fig, axs = plt.subplots(n_s,n_t, figsize = (n_t*1.5,n_t*1.2))

plt.suptitle('WITH batch norm adaptation', size = "xx-large")

# common x and y labels
fig.text(0.5, 0.04, 'Target', ha='center')
fig.text(0.93, 0.5, 'Source', va='center', rotation='vertical')

for source, target in tqdm(list(lst), desc = f'creating histograms of dim {dim}'):
    ls_adapted = torch.load(f'data_toydataset/ls_adapted/ls_adapted_{source}{target}.pt')
    samples = ls_adapted[:,dim]

    ax = axs[source][target]
    ax.hist(samples, bins = bins, density = True, stacked = True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([a,b])
    ax.set_ylim([0, 2])

    if source == 0:
        ax.set_title(parameters_list_target[target], size = "medium")
    if target == 0:
        ax.set_ylabel(parameters_list_source[source],rotation=90, size='medium')
    if source == target:
        ax.set_facecolor((1,1,0,0.3)) # RGBA

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(folder + f'dim{dim}_adapted.png')

raise howitfeelstochew5gum

##################################
### STEP : CHANGE FOLDER NAME ###
##################################
print_step(9, "change folder name")

from os import rename
rename('data_toydataset',f'data_toydataset_{iteration}')

###############################
### STEP 10: RELEASE MEMORY ###
###############################
print_step(10, "release memory")

for var in dir():
    if not var.startswith('__') and not var in save_vars:
        del globals()[var] # delete variable
del var

import torch
torch.cuda.empty_cache() # empty torch cache

memory_usage.append((torch.cuda.memory_reserved(0),torch.cuda.memory_allocated(0)))

print(memory_usage)

# torch.cuda.mem_get_info()


t.time()
