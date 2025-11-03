## Libraries
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import re
import pandas as pd
import glob

from skimage import feature
from skimage import exposure
from skimage.filters import gabor

from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class Experiment:
    """
    
    input:
        - lbp_numPoints, int, param for LBP function
        - lbp_radius, int, param for LBP function
        - lbp_method, string, param for LBP function
        - lbp_eps, float, param for LBP function
        - glcm_list_distances, list, param for GLCM function
        - glcm_list_angles, list, param for GLCM function
        - glcm_gray_level, int, param for GLCM function
        - gf_gamma, float, param for Gabor function
        - gf_sigma, float, param for Gabor function
        - gf_phi, float, param for Gabor function
        - gf_theta_list, list, param for Gabor function
        - gf_lamda_list, list, param for Gabor function
        - pca_n_components, int, param for PCA function
        - resize_dim, tuple, size for reszing the images
        - rfe_algorithm, model procedure, param  
        - rfe_n_features, int,
        - algorithm, model procedure, 
    
    """
    def __init__(self,
                 lbp_numPoints, 
                 lbp_radius, 
                 lbp_method,
                 lbp_eps,
                 glcm_list_distances, 
                 glcm_list_angles, 
                 glcm_gray_level,
                 gf_gamma, 
                 gf_sigma, 
                 gf_phi, 
                 gf_theta_list, 
                 gf_lamda_list,
                 pca_n_components,
                 resize_dim,
                 rfe_algorithm,
                 rfe_n_features,
                 algorithm):
        
        self.lbp_numPoints = lbp_numPoints 
        self.lbp_radius = lbp_radius 
        self.lbp_method = lbp_method
        self.lbp_eps = lbp_eps
        self.glcm_list_distances = glcm_list_distances
        self.glcm_list_angles = glcm_list_angles
        self.glcm_gray_level = glcm_gray_level
        self.gf_gamma = gf_gamma
        self.gf_sigma = gf_sigma
        self.gf_phi = gf_phi 
        self.gf_theta_list = gf_theta_list
        self.gf_lamda_list = gf_lamda_list
        self.pca_n_components = pca_n_components
        self.resize_dim = resize_dim
        self.rfe_algorithm = rfe_algorithm
        self.rfe_n_features = rfe_n_features
        self.algorithm = algorithm
        
    
    def localBinaryPattern(self, image):
        """
        Gray Level Co-occurence Matrix extracts texture features from an image.
        
        input:
            - image, 2D array, input image
        output:
            - contrast, dissimilarity, homogeneity, energy, correlation, ASM, array, for each image
        """
        lbp = feature.local_binary_pattern(image, 
                                           self.lbp_numPoints, 
                                           self.lbp_radius, 
                                           self.lbp_method)
        hist, _ = np.histogram(lbp.ravel(), 
                               bins = np.arange(0, self.lbp_numPoints + 3), 
                               range = (0, self.lbp_numPoints + 2))

        # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + self.lbp_eps)
        
        return hist        
    
    
    def grayLevelCoMatrix(self, image):
        """
        Gray Level Co-occurence Matrix extracts texture features from an image.
        
        input:
            - image, 2D array, input image
        output:
            - contrast, dissimilarity, homogeneity, energy, correlation, ASM, array, for each image
        """
        
        # GLCM
        graycom = feature.greycomatrix(image, 
                                       distances = self.glcm_list_distances, 
                                       angles = self.glcm_list_angles, 
                                       levels = self.glcm_gray_level)

        # Find the GLCM properties
        contrast = feature.greycoprops(graycom, 'contrast')
        dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
        homogeneity = feature.greycoprops(graycom, 'homogeneity')
        energy = feature.greycoprops(graycom, 'energy')
        correlation = feature.greycoprops(graycom, 'correlation')
        ASM = feature.greycoprops(graycom, 'ASM')

        return np.concatenate((contrast[0], dissimilarity[0], homogeneity[0], energy[0], correlation[0], ASM[0]))
    
    
    def gaborFilters(self, image):
        """
        Applying Gabor filters with different combination of parameters of an image.
        
        input:
            - image, 2D array, input image
        output:
            - mean_ampl_list, array, mean amplitude of the image for each Gabor filter applied
            - local_energy, array, local energy  of the image for each Gabor filter applied
        """
        
        local_energy_list=[]
        mean_ampl_list=[]

        for theta in self.gf_theta_list:
            for lamda in self.gf_lamda_list:
                # Applying Gabor filter with the params configuration
                kernel = cv2.getGaborKernel((3,3), 
                                          self.gf_sigma, 
                                          theta, 
                                          lamda, 
                                          self.gf_gamma, 
                                          self.gf_phi, 
                                          ktype=cv2.CV_32F)
                fimage = cv2.filter2D(image, 
                                      cv2.CV_8UC3, 
                                      kernel)

                # Calculating the mean amplitude
                mean_ampl=np.sum(abs(fimage))
                mean_ampl_list.append(mean_ampl)

                # Calculating the local energy
                local_energy=np.sum(fimage**2)
                local_energy_list.append(local_energy)

        return np.asarray(mean_ampl_list), np.asarray(local_energy_list)
    

    def HOG(self, image):
        """
        Histogram Oriented Gradient
        
        input:
            - - image, 2D array, input image
        output:
            - hog_image_rescaled, 2D array, 
        """
        fd, hog_image = feature.hog(image, orientations = 8, pixels_per_cell = (4, 4), cells_per_block = (2, 2), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range = (0, 10))
        
        return hog_image_rescaled
    
    
    def getTextureFeatures(self, image_files, set_type):
        """
        Getting texture features of a list of images.
        
        input:
            - image_files, list, image path files
            - set_type, string, extracting image for train or for test
        output:
            - features, array, feature's array for each image 
            - columns, array, feature's names
        """
        
        im_flatten = []
        hog_flatten = []
        lbp_features = []
        glcm_features = []
        gabor_features = []    

        for i in range(len(image_files)):
            im = cv2.imread(image_files[i])    
            
            # Pre-processing
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_res = cv2.resize(gray, self.resize_dim, interpolation = cv2.INTER_LINEAR)
            im_res = im_res[:450, 100:1024]

            # local Binary Pattern
            hist = self.localBinaryPattern(im_res)
            lbp_features.append(hist)

            # Gray Level Co-ocurrence Matrix
            graycom = self.grayLevelCoMatrix(im_res)
            glcm_features.append(graycom)

            # Gabor filters
            mean_ampl_list, local_energy_list = self.gaborFilters(im_res)
            gabor_features.append(np.concatenate((mean_ampl_list, local_energy_list)))

            # PCA preparation
            im_flatten += [im_res.flatten()]
            
            # HOG preparation
            hog_flatten += [self.HOG(im_res).flatten()]

        # PCA features
        if set_type == 'train':
            self.pca = PCA(n_components = self.pca_n_components)
            pca_features = self.pca.fit_transform(im_flatten)
        else:
            pca_features = self.pca.transform(im_flatten)
            
        # HOG features
        if set_type == 'train':
            self.hog_pca = PCA(n_components = self.pca_n_components)
            hog_features = self.hog_pca.fit_transform(hog_flatten)
        else:
            hog_features = self.pca.transform(hog_flatten)

        # Gathering all features
        features = np.concatenate((np.asarray(lbp_features), 
                                       np.asarray(glcm_features), 
                                       np.asarray(gabor_features), 
                                       pca_features,
                                       hog_features), axis = 1)

        columns = np.concatenate((np.asarray(["lbp_" + str(i) for i in range(len(lbp_features[0]))]), 
                            np.asarray(["glcm" + str(i) for i in range(len(glcm_features[0]))]), 
                            np.asarray(["gabor_ampl" + str(i) for i in range(len(mean_ampl_list))]), 
                            np.asarray(["gabor_energy" + str(i) for i in range(len(local_energy_list))]), 
                            np.asarray(["pca" + str(i) for i in range(len(pca_features[0]))]), 
                            np.asarray(["hog" + str(i) for i in range(len(hog_features[0]))])))

        return features, columns
    
    def crossValScore(self, algorithm, X, y):
        """
        Cross validation function for training classifier using f1 score as metric.
        
        input:
            - algorithm, model procedure
            - X, 2D array, features values
            - y, array, target values
        output:
            - mean and st of f1 score for all k-fold validation sets
        """
        
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)
        f1 = []
        for train_index, test_index in skf.split(X, y):
            algorithm.fit(X[train_index, :], y[train_index])
            y_pred = algorithm.predict(X[test_index])
            f1 += [f1_score(y[test_index], y_pred)]
        return np.mean(f1), np.std(f1)

    
    def run(self, image_files, y_real, idx_train, idx_test):
        """
        Running the experiment with
        
        input:
            - image_files, list, image path files
            - y_real, array, target values
            - idx_train, array, indexes corresponding to train images
            - idx_test, array, indexes corresponding to test images
        output:
            - sel_features, array, features selected
            - cv_mean, float, f1 score mean for all k-fold validation sets
            - cv_std, float, f1 score standard desv. for all k-fold validation sets
            - score, float, f1 score from test set
            - disp, display of the confusion matrix
        """
        
        # Transform
        X_train, columns = self.getTextureFeatures(image_files[idx_train], set_type = "train")
        X_test, columns = self.getTextureFeatures(image_files[idx_test], set_type = "test")
        y_train = y_real[idx_train]
        y_test = y_real[idx_test]
        
        # Feature selection
        self.rfe = RFE(estimator = self.rfe_algorithm, 
                  n_features_to_select = self.rfe_n_features)
        self.rfe.fit(X_train, y_train)
        sel_features = self.rfe.support_
        
        # Cross validation
        cv_mean, cv_std = self.crossValScore(self.algorithm, X_train[:, self.rfe.support_], y_train)
        
        # Fit model
        self.algorithm.fit(X_train[:, self.rfe.support_], y_train)
        
        # Predict
        y_pred = self.algorithm.predict(X_test[:, self.rfe.support_])
        score = f1_score(y_test, y_pred)
        
        # Evaluate
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['normal','dme'])
        
        return sel_features, cv_mean, cv_std, score, disp