from .utils.functions import create_directories
from PIL import Image
from tqdm import tqdm
from math import cos, radians, sin
from PIL import ImageDraw
from random import randint
import numpy as np

def triangles_and_circles(img_folder, label_folder, n, parameters = (100,20,150,20), t_or_c = 't', size = (64,64), verbose = False):
    
    """Generate triangles and circles masks and images.
    img_folder:     folder where images will be generated;
    label_folder:   folder where masks will be generated;
    n:              number of images + masks generated;
    parameters:     noise parameters (mu1, sigma1, mu2, sigma2);
    t_or_c:         't' will set the triangles as masks, 'c' will set the circles as masks;
    size:           width and height of the output image;
    verbose:        if True will show the progress bar."""

    # check if input is correct
    assert img_folder[-1] == '/', f"please put a slash in the folder name: '{img_folder}' -> '{img_folder}/'"
    assert label_folder[-1] == '/', f"please put a slash in the folder name: '{label_folder}' -> '{label_folder}/'"
    mu1, sigma1, mu2, sigma2 = parameters
    assert mu1<=mu2, "mu1 should not be bigger than mu2!"
    
    # create folders
    create_directories(img_folder, verbose = False)
    create_directories(label_folder, verbose = False)

    W,H = size
    for i in tqdm(range(n), disable = not verbose, leave = False):
        # given a point with polar coordinates (r, theta), the function trngl returns this point in cartesian coordinates translated by (x,y)
        trngl = lambda x,y,r,theta: (x + r * cos(radians(theta)), y + r * sin(radians(theta)))
        
        # create image and mask (label) and corresponding ImageDraw classes
        img = Image.new('1',(W,H))
        label = Image.new('RGB',(W,H))
        draw_img = ImageDraw.Draw(img)
        draw_label = ImageDraw.Draw(label)
        
        for j in range(10):
            # select random center and radius
            x,y = randint(-W//4,W+W//4), randint(-H//4,H+H//4)
            r = randint(3,max(W,H)//3)

            # the first circle (triangle) does not need to check for intersection with other circles (triangles)
            if j == 0:
                # append (x,y,r) to circles list: this will be used to check for intersections when generating the next circles (triangles)
                circles = [(x,y,r)]

                # 50% of chance to create a circle
                if randint(0,1) == 0:
                    circle = [(x-r,y-r),(x+r,y+r)]
                    draw_img.ellipse(circle, fill = 'white')
                    if t_or_c == 'c':
                        draw_label.ellipse(circle, fill = 'white')
                
                # 50% chance to create a triangle
                else:
                    theta = randint(0,359)
                    triangle = [trngl(x,y,r,theta-120),trngl(x,y,r,theta),trngl(x,y,r,theta+120)]
                    draw_img.polygon(triangle,fill = 'white')
                    if t_or_c == 't':
                        draw_label.polygon(triangle, fill = 'white')
                
                # jump to j=1
                continue
            
            # this code checks for intersection with already existing circles (triangles)
            intersects = False
            for circle in circles:
                d_squared = (circle[0] - x) * (circle[0] - x) + (circle[1] - y) * (circle[1] - y)
                if d_squared < (circle[2] + r) * (circle[2] + r):
                    intersects = True
            
            # if it does not intersect and circle (triangle), will add the figure and add its coordinates
            if not intersects:
                circles.append((x,y,r))
                if randint(0,1) == 0:
                    circle = [(x-r,y-r),(x+r,y+r)]
                    draw_img.ellipse(circle, fill = 'white')
                    if t_or_c == 'c':
                        draw_label.ellipse(circle, fill = 'white')
                else:
                    theta = randint(0,359)
                    triangle = [trngl(x,y,r,theta-120),trngl(x,y,r,theta),trngl(x,y,r,theta+120)]
                    draw_img.polygon(triangle,fill = 'white')
                    if t_or_c == 't':
                        draw_label.polygon(triangle, fill = 'white')
        
        label.save(label_folder + f'{i}.png')

        tmp = np.array(img)
        noise_False = np.clip(np.random.normal(mu1, sigma1, size = (W,H)),0,255) # clip to avoid conversion issues in uint8
        noise_True  = np.clip(np.random.normal(mu2, sigma2, size = (W,H)),0,255)
        
        x = tmp * noise_True + (1 - tmp)*noise_False

        x = x.astype(np.uint8)
        img = Image.fromarray(x).convert('RGB')
        img.save(img_folder + f'{i}.png')
