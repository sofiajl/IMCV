
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line, hough_circle, hough_circle_peaks
import pygame as pg
from pygame.math import Vector2
import math



def estimate_distances(number, center, radius, image):
    '''
        Estimate distances from circumference center to image edge
        Input:
            - number, int, number of angles to test
            - center, (int, int), coordinates of circumference center
            - radius, int, radius of the circumference
            - image, np.array, image to check
        Output:
            - list(int), distances from center to edge
    '''
    
    # Number of angles to test
    angle = 360 / number
    
    # Calculate distances to the edge
    distances = []
    points = []
    for i in range(number):
        border_found = False
        
        # Check radius (within a range)
        for j in range((radius[0] *2)//3, radius[0] * 2):
            
            # Point found given a center, an angle and a radius
            vec = center + Vector2(j, 0).rotate(i * angle)
            
            if (int(vec.x) < 0) | (int(vec.y) < 0):
                break                
            
            try:
                # Check if border
                if image[int(vec.y), int(vec.x)] == 255:
                    distances += [j]
                    points += [[int(vec.x), int(vec.y)]]
                    border_found = True
                    
                    if i*angle == 0:
                        rad_right = j
                    elif i*angle == 90:
                        rad_top = j
                    elif i*angle == 180:
                        rad_left = j
                    elif i*angle == 270:
                        rad_down = j
                    
                    break
            except IndexError:
                break
        
        # In case of not border found
        if not border_found:
            # Get the radius of previous point
            try:
                vec = center + Vector2(distances[-1], 0).rotate(i * angle) 
                distances += [distances[-1]]
                points += [[int(vec.x), int(vec.y)]]

                if i*angle == 0:
                    rad_right = distances[-1]
                elif i*angle == 90:
                    rad_top = distances[-1]
                elif i*angle == 180:
                    rad_left = distances[-1]
                elif i*angle == 270:
                    rad_down = distances[-1]

                    
            except IndexError:
                # When border for the first angle is missing
                if i == 0:
                    vec = center + Vector2(int(radius), 0).rotate(i * angle) 
                    distances += [int(radius)]
                    points += [[int(vec.x), int(vec.y)]]
                    if i*angle == 0:
                        rad_right = int(radius)
                    elif i*angle == 90:
                        rad_top = int(radius)
                    elif i*angle == 180:
                        rad_left = int(radius)
                    elif i*angle == 270:
                        rad_down = int(radius)
                    
            
    return distances, points, rad_right, rad_left, rad_top, rad_down


def circ_points(number, center, radius):
    '''
        Estimate the cirfumference's points for the specific radius and center
        Input:
            - number, int, number of angles to test
            - center, (int, int), coordinates of circumference center
            - radius, int, radius of the circumference
        Output:
            - list(int), points list of the circumference
    '''
    angle = 360 / number
    point_list = []
    for i in range(number):
        vec = center + Vector2(radius, 0).rotate(i * angle)
        point_list.append([int(vec.x), int(vec.y)])
    return point_list


def struts_selection(image, bw_area_points, bw_circle_points, bw_circle_5_points, bw_circle_10_points, bw_circle_15_points, min_point_strut, max_point_strut):
    '''
        Selection of struts by analyzed different size of circumferences 
        Input:
            - image, np.array, image to check
            - bw_area_points, list(int), points list of the area
            - bw_circle_points, list(int), points list of a circumference (Hough detection)
            - bw_circle_5_points, list(int), points list of a circumference (Hough detection + 5)
            - bw_circle_10_points, list(int), points list of a circumference (Hough detection + 10)
            - bw_circle_15_points, list(int), points list of a circumference (Hough detection + 15)
            - min_point_strut, int, min threshold for determining if a candidate is consider or not
            - max_point_strut, int, max threshold for determining if a candidate is consider or not
        Output:
            - selection_1_order, array, candidates selection in 1st order
            - selection_2_order, array, candidates selection in 2nd order
            - selection_3_order, array, candidates selection in 3rd order
            - pos_struts_1, dict, circumference position (1st order) and number-name assigned of the struts selected
            - pos_struts_3, dict, circumference position (3rd order) and number-name assigned of the struts selected
    '''    
    
    # Getting points of 1st degree
    selection_1_order = np.unique(np.concatenate((np.where(np.asarray([i[1] for i in bw_circle_points])==0)[0], np.where(np.asarray([i[1] for i in bw_area_points])==0)[0])))

    # Getting points of 2n degree
    candidates_2_order = np.unique(np.concatenate((np.where(np.asarray([i[1] for i in bw_circle_5_points])==0)[0], np.where(np.asarray([i[1] for i in bw_circle_10_points])==0)[0])))


    selection_2_order = []
    size = []
    for i in selection_1_order:
        if i in candidates_2_order and i-1 in candidates_2_order:
            selection_2_order +=[ i-1]

        if i in candidates_2_order and i+1 in candidates_2_order:
            selection_2_order += [i+1]

        if i in candidates_2_order:
            selection_2_order += [i]

    selection_2_order = np.unique(np.concatenate((selection_1_order,np.unique(selection_2_order)))).astype(int)


    # Getting points of 3rd degree
    candidates_3_order = np.where(np.asarray([i[1] for i in bw_circle_15_points])==0)[0]

    selection_3_order = []
    for i in selection_2_order:
        if i in candidates_3_order:
            selection_3_order += [i]

        if i in candidates_3_order and i-1 in candidates_3_order:
            selection_3_order +=[ i-1]

        if i in candidates_3_order and i+1 in candidates_3_order:
            selection_3_order += [i+1]


    selection_3_order = np.unique(np.concatenate((selection_2_order,np.unique(selection_3_order)))).astype(int)
    
    
    # Assigning points by candidate in 3rd degree 
    pos_struts_3 = {}
    strut = 1
    for i, pos in enumerate(selection_3_order):

        if selection_3_order[i] - 1 == selection_3_order[i-1]:
            pos_struts_3[pos] = strut-1
        else:
            pos_struts_3[pos] = strut
            strut += 1

    # Number of point within 3rd degree
    unique, counts = np.unique(np.array(list(pos_struts_3.values())), return_counts=True)
    struts_3 = dict(zip(unique, counts))

    # Assigning points by candidate in 3rd degree
    pos_struts_1 = {}
    for pos in selection_1_order:
        if pos in list(pos_struts_3.keys()):        
            pos_struts_1[pos] = pos_struts_3[pos]

    # Number of point within 1st degree
    unique, counts = np.unique(np.array(list(pos_struts_1.values())), return_counts=True)
    struts_1 = dict(zip(unique, counts))
    
    
    # Filtering selected candidates
    pos_remove = []
    for pos, strut in pos_struts_3.items():
        if struts_1[strut] > max_point_strut:
            pos_remove += [pos]
        if struts_3[strut] == min_point_strut:
            pos_remove += [pos]

    for i in pos_remove:
        del pos_struts_3[i]

    for i in pos_remove:
        try:
            del pos_struts_1[i]
        except:
            pass
        
    return selection_1_order,selection_2_order,selection_3_order, pos_struts_1, pos_struts_3