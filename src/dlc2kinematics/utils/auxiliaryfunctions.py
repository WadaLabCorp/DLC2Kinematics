"""
dlc2kinematics
© M. Mathis Lab
Some functions are adopted directly from https://github.com/DeepLabCut/DeepLabCut
We thank those authors for this code, and are licensed:
Licensed under GNU Lesser General Public License v3.0
"""

import pandas as pd
import numpy as np
import scipy as sc
import skinematics
from skinematics import quat, vector
import os
from ruamel.yaml import YAML


def read_config(configname):
    if not os.path.exists(configname):
        raise FileNotFoundError(
            "Config file is not found. Please make sure that the file exists and/or "
            "there are no unnecessary spaces in the path of the config file!"
        )
    with open(configname) as file:
        yaml = YAML()
        return yaml.load(file)


"""
def _get_filenames(destfolder, output_filename, scorer):

    if destfolder == None:
        destfolder = os.getcwd()

    if output_filename == None:
        output_filename = str("joint_quaternions_" + scorer)

    return destfolder, output_filename
"""


def IntersectionofBodyPartsandOnesGivenbyUser(cfg, comparisonbodyparts):
    """FUNCTION TAKEN FROM DEEPLABCUT. Returns all body parts when comparisonbodyparts=='all', otherwise all bpts that are in the intersection of comparisonbodyparts and the actual bodyparts"""
    allbpts = cfg["bodyparts"]
    if "MULTI" in allbpts:
        allbpts = cfg["multianimalbodyparts"] + cfg["uniquebodyparts"]
    if comparisonbodyparts == "all":
        return allbpts
    else:  # take only items in list that are actually bodyparts...
        cpbpts = []
        # Ensure same order as in config.yaml
        for bp in allbpts:
            if bp in comparisonbodyparts:
                cpbpts.append(bp)
        return cpbpts


def getbodyparts(config, df):
    # bodyparts = df.columns.get_level_values("bodyparts").unique().tolist()
    cfg = read_config(config)
    bodyparts = IntersectionofBodyPartsandOnesGivenbyUser(cfg, displayedbodyparts)
    # bodyparts = df.columns.get_level_values(1)
    print(bodyparts)
    _, idx = np.unique(bodyparts, return_index=True)
    bodyparts = list(bodyparts[np.sort(idx)])
    return bodyparts


def check_2d_or_3d(df):
    if "z" in list(df.columns.get_level_values("coords")):
        flag = "3d"
        columns = ["x", "y", "z"]
    else:
        flag = "2d"
        columns = ["x", "y", "likelihood"]
    return flag, columns


def smooth(df, dataFrame, pdindex, bp, columns, filter_window, order, axis=0, deriv=0):
    """Smooths the input pandas data frame with a Savitzky-Golay filter. If it's 2D data, it also appends the likelihood from DLC"""

    scorer = df.columns.get_level_values(0)[0]
    flag, columns = check_2d_or_3d(df)

    if flag == "2d":
        smooth = sc.signal.savgol_filter(
            df[scorer][bp][columns[0:2]],
            window_length=filter_window,
            polyorder=order,
            axis=0,
            deriv=deriv,
        )
        likelihood = np.array(df[scorer][bp][columns[-1]])
        # join likelihood to the smoothed data
        smooth = [np.append(val, likelihood[idx]) for idx, val in enumerate(smooth)]
        frame = pd.DataFrame(smooth, columns=pdindex, index=range(0, df.shape[0]))
        dataFrame = pd.concat([dataFrame, frame], axis=1)
    else:
        smooth = sc.signal.savgol_filter(
            df[scorer][bp][columns],
            window_length=filter_window,
            polyorder=order,
            axis=0,
            deriv=deriv,
        )
        frame = pd.DataFrame(smooth, columns=pdindex, index=range(0, df.shape[0]))
        dataFrame = pd.concat([dataFrame, frame], axis=1)

    return dataFrame

def signed_angle(v1,v2):
    '''Angle between two vectors
    
    Parameters
    ----------
    v1 : array (N,) or (M,N)
        vector 1
    v2 : array (N,) or (M,N)
        vector 2

    Returns
    -------
    angle : double or array(M,)
        angle between v1 and v2


    .. image:: ../docs/Images/vector_angle.png
        :scale: 33%

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    >>>       [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    >>>       [0,1,0]])
    >>> skinematics.vector.angle(v1,v2)
    array([ 1.30024656,  0.96453036])
    
    Notes
    -----

    .. math::
        \\alpha =arccos(\\frac{\\vec{v_1} \\cdot \\vec{v_2}}{| \\vec{v_1} |
        \\cdot | \\vec{v_2}|})
    '''

    
    # make sure lists are handled correctly
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    if v1.ndim < v2.ndim:
        v1, v2 = v2, v1
    n1 = vector.normalize(v1)
    n2 = vector.normalize(v2)
    dot = np.dot(n1, n2)
    if v2.ndim == 1:
        sign = np.arcsin(dot)
        angle = np.arccos(dot)
        if sign < 0:
            angle = -angle
    else:
        angle = np.arccos(list(map(np.dot, n1, n2)))
    return angle

def q_signed_shortest_rotation(v1, v2):
    """Quaternion indicating the shortest rotation from one vector into another.
    You can read "qrotate" as either "quaternion rotate" or as "quick
    rotate".

    Parameters
    ----------
    v1 : ndarray (3,)
        first vector
    v2 : ndarray (3,)
        second vector

    Returns
    -------
    q : ndarray (3,)
        quaternion rotating v1 into v2


    .. image:: ../docs/Images/vector_q_shortest_rotation.png
        :scale: 33%

    Example
    -------
    >>> v1 = np.r_[1,0,0]
    >>> v2 = np.r_[1,1,0]
    >>> q = qrotate(v1, v2)
    >>> print(q)
    [ 0.          0.          0.38268343]
    """

    # calculate the direction
    n = vector.normalize(np.cross(v1, v2))

    # make sure vectors are handled correctly
    n = np.atleast_2d(n)

    # handle 0-quaternions
    nanindex = np.isnan(n[:, 0])
    n[nanindex, :] = 0

    # find the angle, and calculate the quaternion
    angle12 = signed_angle(v1, v2)
    print(angle12)
    q = (n.T * np.sin(angle12 / 2.0)).T

    # if you are working with vectors, only return a vector
    if q.shape[0] == 1:
        q = q.flatten()

    return q


def jointquat_calc(pos, use4d=False):
    """Returns the quaternion of shortest rotation at a joint given position of keypoints

    Parameters
    ----------
    pos : np.array or pandas, shape either [1, 9] or [3, 3]
        x, y, z coordinates of three relevant keypoints: First joint, second joint (around which rotation is calculated), third joint
        e.g. if we are calculating the quaternion of rotation at the elbow, this would be (1) shoulder, (2) elbow, (3) wrist

    use4d: bool
        Optional. Whether or not to return result as 4d quaternion or 3d quaternion

    Returns
    -------
    q_shortest_rotation: ndarray (3,)

    """
    # print('this place is reached')
    pos = pos.values
    # print(pos)
    pos = pos.reshape((3, 3))
    # print(pos)
    u = pos[1, :] - pos[0, :]
    v = pos[1, :] - pos[2, :]
    # print(u.shape)
    # print(v.shape)
    q_shortest_rotation = q_signed_shortest_rotation(u.astype(float), v.astype(float))
    # print(q_shortest_rotation)
    if use4d:
        q_shortest_rotation = quat.unit_q(q_shortest_rotation)

    return q_shortest_rotation


def calc_q_angle(q):
    """Return the angle associated with a quaternion of rotation

    Parameters
    ----------

    q : np.array or skinematics.quat
        The quaternion of rotation, either 3d or 4d

    Returns
    -------
    theta : float
        The shortest angle associated to the same rotation

    """
    q = quat.unit_q(q)

    cos = q[0][0]

    arccos = np.arccos(cos) * 2 * (180 / np.pi)
    arcsin = np.arcsin(cos) * 2 * (180 / np.pi)

    theta = arccos if arcsin >= 0 else -arccos

    return theta


def calc_q_axis(q):
    """Return the axis of rotation associated with a rotation quaternion
    Source of equations: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm

     Parameters
    ----------

    q : np.array or skinematics.quat
        The quaternion of rotation, either 3d or 4d

    Returns
    -------
    axis: np.array, [1, 3]

    """

    q = quat.unit_q(q)

    w = q[0][0]
    denom = np.sqrt(1 - w**2)

    return np.array([q[0][1] / denom, q[0][2] / denom, q[0][3] / denom])


def jointangle_calc(pos):
    """Return the joint angle given position of keypoints

    Parameters
    ----------
    pos : np.array or pandas, shape either [1, 9] or [3, 3]
        x, y, z coordinates of three relevant keypoints: First joint, second joint (around which rotation is calculated), third joint
        e.g. if we are calculating the quaternion of rotation at the elbow, this would be (1) shoulder, (2) elbow, (3) wrist

    Returns
    -------
    angle : float

    """
    q_shortest_rotation = jointquat_calc(pos)
    #     turns a quaternion vector into a unit quaternion
    unit = quat.unit_q(q_shortest_rotation)
    angle = calc_q_angle(unit)
    return angle


def doubleangle_calc(pos):
    """Return first two Euler angles at a joint given position of keypoints

    Parameters
    ----------
    pos : np.array or pandas, shape either [1, 9] or [3, 3]
        x, y, z coordinates of three relevant keypoints: First joint, second joint (around which rotation is calculated), third joint
        e.g. if we are calculating the quaternion of rotation at the elbow, this would be (1) shoulder, (2) elbow, (3) wrist

    Returns
    -------
    doubleangles : np.array, [1, 2]
    """

    pos = pos.values
    # print(pos)
    pos = pos.reshape((3, 3))
    # print(pos)
    u = pos[1, :] - pos[0, :]
    v = pos[1, :] - pos[2, :]

    rel = v - u

    x = rel[0]
    y = rel[1]
    z = rel[2]
    # yaw, pitch relative to the vertical plane (x-z plane)
    yaw = np.rad2deg(np.arctan2(x, z))
    padj = np.sqrt((x**2) + (z**2))
    pitch = np.arctan(padj / y) * 180.0 / np.pi

    return np.array([pitch, yaw])


def smoothen_angles(angles, jointname, filter_window=3, order=1):
    """Smoothen angle

    Parameters
    ----------
    angles : Pandas dataframe
    jointname : float
    filter_window : int
    order : int

    Returns
    -------
    angles : np.array
    """

    return sc.signal.savgol_filter(
        angles[jointname], window_length=filter_window, polyorder=order, axis=0, deriv=0
    )


def create_empty_df(df):
    """
    Creates an empty dataFrame of same shape as dataFrame (df) and with the same indices.

    """

    a = np.empty((df.shape[0], 3))
    a[:] = np.nan
    dataFrame = None
    scorer = df.columns.get_level_values(0)[0]
    flag, columns = check_2d_or_3d(df)
    bodyparts = list(df.columns.get_level_values(1))[0::3]
    for bodypart in bodyparts:
        if flag == "2d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "likelihood"]],
                names=["scorer", "bodyparts", "coords"],
            )
        elif flag == "3d":
            pdindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y", "z"]],
                names=["scorer", "bodyparts", "coords"],
            )
        frame = pd.DataFrame(a, columns=pdindex, index=range(0, df.shape[0]))
        dataFrame = pd.concat([dataFrame, frame], axis=1)
    return dataFrame


def remove_outlier_points(df, bodyparts, scorer, pcutoff, flag):
    """
    Any point above the pcutoff is assigned to nan
    """
    dataframe = df.copy(deep=True)
    if flag == "2d":
        for bp in bodyparts:
            dataframe.loc[:][scorer, bp, "x"][
                dataframe.loc[:][scorer, bp, "x"] > pcutoff
            ] = np.nan
            dataframe.loc[:][scorer, bp, "y"][
                dataframe.loc[:][scorer, bp, "y"] > pcutoff
            ] = np.nan
    else:
        for bp in bodyparts:
            dataframe.loc[:][scorer, bp, "x"][
                dataframe.loc[:][scorer, bp, "x"] > pcutoff
            ] = np.nan
            dataframe.loc[:][scorer, bp, "y"][
                dataframe.loc[:][scorer, bp, "y"] > pcutoff
            ] = np.nan
            dataframe.loc[:][scorer, bp, "z"][
                dataframe.loc[:][scorer, bp, "z"] > pcutoff
            ] = np.nan
    return dataframe


def points_above_pcutoff(df, bodyparts, scorer, pcutoff):
    """
    Bodypart with low likelihood (i.e. < pcutoff) is assigned to nan. It is used only in case of 2d data.

    """
    dataframe = df.copy(deep=True)
    for bp in bodyparts:
        dataframe.loc[:][scorer, bp, "x"][
            dataframe.loc[:][scorer, bp, "likelihood"] < pcutoff
        ] = np.nan
        dataframe.loc[:][scorer, bp, "y"][
            dataframe.loc[:][scorer, bp, "likelihood"] < pcutoff
        ] = np.nan
    return dataframe


def remove_outlier_angles_points(df, angles, pcutoff):
    """
    Any joint angle above the pcutoff is assigned to nan
    """
    dataframe = df.copy(deep=True)
    for ang in angles:
        dataframe.loc[:][ang][dataframe.loc[:][ang] > pcutoff] = np.nan
    return dataframe
