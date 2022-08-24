from utilss.libs import *

root = '../private_test/BAD/'

def remove_space():
    path_image = [name for name in os.listdir(root) if name.endswith('jpg')]
    for path in tqdm(path_image):
        new_name = ''.join(path.split())
        os.rename(root + path, root + new_name)
# remove_space()

