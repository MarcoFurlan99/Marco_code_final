from os import makedirs
from os.path import isdir
import shutil

def create_directories(folder, verbose = True):
    """Will try to create the full path 'folder'. If already existing will do nothing"""
    try:
        makedirs(folder)
        if verbose: print(f'created folder {folder}')
    except:
        None

def remove_directory(folder, ask = True, verbose = True):
    """Removes the folder 'folder' and all its contents, if such a folder exists. Use carefully.
    If 'ask' = True it will ask confirmation in the terminal before removing; if 'notify' = True will print the message "The folder "'+folder+'" was removed!". """
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

print_step = lambda i, text: print(f"\n> STEP {i} - \t{text}")

