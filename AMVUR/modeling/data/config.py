"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.

Adapted from opensource project GraphCMR (https://github.com/nkolot/GraphCMR/) and Pose2Mesh (https://github.com/hongsukchoi/Pose2Mesh_RELEASE)

"""

from os.path import join
folder_path = 'AMVUR/modeling/'
MANO_FILE = folder_path + 'data/MANO_RIGHT.pkl'
MANO_sampling_matrix = folder_path + 'data/mano_downsampling.npz'

"""
We follow the hand joint definition and mesh topology from 
open source project Manopth (https://github.com/hassony2/manopth)

The hand joints used here are:
"""
J_NAME = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1',
'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4')
ROOT_INDEX = 0