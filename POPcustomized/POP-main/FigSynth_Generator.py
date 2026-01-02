from math import sin, cos, pi
import random
import smplx
import numpy as np
import torch
import os
from os import listdir
from os.path import join, dirname, realpath, isfile, isdir
from lib.config_parser import parse_config
from lib.network import POP
from lib.utils_io import load_masks, load_barycentric_coords, load_latent_feats
from lib.utils_model import SampleSquarePoints
from lib.utils_model import gen_transf_mtx_from_vtransf
from os import path as osp
import cv2
import json
import open3d as o3d
from mesh_normalise import mesh_read, mesh_write, v_normalise


# initialisation
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
RESULT_DIR = '/dockerspace/BigBench/Workspace/UnityProjects/FlagshipBB/Assets/Resources'
# RESULT_DIR = '/dockerspace/BigBench/test_field'
STATUS_FILE = 'status.txt'
MESH_INFO_FILE = 'mesh_info.txt'
CLOTH_IMG_DIR = '/dockerspace/BigBench/Downloads/images'
CLO_IMG_list = [f for f in listdir(CLOTH_IMG_DIR) if isfile(join(CLOTH_IMG_DIR, f))]
MAX_ITERATION = 8000
SMPLX_TOOLFIT_DIR = '/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/'
MODEL_FOLDER = '/dockerspace/BigBench/Downloads/AMASS/models_smplx_v1_1/models'
MODEL_TYPE = 'smplx'
EXT = 'npz'
USE_FACE_CONTOUR = True
NUM_BETAS = 10
MASK_DIR = '/dockerspace/Workspace/POPcustomized/POP-main/assets/uv_masks'
SMPLX_POP_TOOLKIT_DIR = '/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/'
BETAS_MEAN = torch.zeros([1, NUM_BETAS], dtype=torch.float32)
T_POSE = np.zeros((1, 63), dtype=float)
torch.manual_seed(12345)
np.random.seed(12345)
bg_dir = '/dockerspace/BigBench/Downloads/images'
bg_ls = [f for f in listdir(bg_dir) if isfile(join(bg_dir, f))]
SHEET_DIR = '/dockerspace/BigBench/Downloads/images/sheet'
SHEET_TXTs = [f for f in listdir(SHEET_DIR) if isfile(join(SHEET_DIR, f))]
inflate_coef = 0.012
MAX_HUMAN_IN_PIC = 4
glasses_dir = '/dockerspace/Workspace/POPcustomized/POP-main/assets/glasses'
glasses_ls = [join(glasses_dir, m) for m in listdir(glasses_dir) if m.endswith(".obj")]
BGR_GLASSES = np.array([[20, 20, 20], [169, 169, 169], [192, 192, 192], [18, 18, 146]])

RGB_HAIR = {'MidnightBlack': [8, 8, 6], 'DarkestBrown': [59, 48, 38], 'ChestnutBrown': [106, 78, 66],
            'GoldenBrown': [167, 133, 106], 'AshBrown': [151, 121, 97], 'GoldenBlonde': [229, 200, 168],
            'MediumGray': [183, 166, 158], 'LightGray': [214, 196, 194], 'BrunetteMud': [102, 79, 60],
            'BrunetteCookie': [140, 104, 74], 'TreeBark': [51, 42, 34]}

RGB_SKIN = [[255, 220, 177], [229, 194, 152], [228, 185, 142], [217, 145, 100], [204, 132, 67], [199, 122, 88],
            [165, 57, 6], [91, 8, 8], [187, 109, 74], [190, 114, 60], [189, 151, 120], [225, 173, 164],
            [168, 112, 63],
            [51, 42, 34], [250, 187, 134], [244, 159, 104], [143, 70, 29], [252, 188, 140], [234, 154, 95]]

RGB_EXTREME_SKIN = np.array([[45, 34, 30], [255, 206, 180]])

BGR_SKIN_TONE = np.array([
    [177, 188, 230],  # pink
    [10, 19, 54],  # dark
    [34, 95, 130],  # yellow
    [166, 190, 235]  # pale
])

BGR_SHOE = np.array(
    [[30, 30, 30], [225, 120, 120], [145, 150, 220], [230, 230, 230], [134, 220, 110], [200, 200, 50], [70, 80, 90],
     [100, 130, 160], [180, 150, 110]])

# prepare the connectivity and vt for a torsoless SMPLX PCD
with open(SMPLX_POP_TOOLKIT_DIR + 'smplx_vert_segmentation.json') as file:  # indexing from 0
    ids_m = json.load(file)
# torso_m = ids_m['torso']
# torso_f = ids_f['torso']
torso_m = ids_m['hips'] + ids_m['spine'] + ids_m['spine1'] + ids_m['spine2'] + ids_m['leftShoulder'] + ids_m[
    'rightShoulder']
torso_f = torso_m

facets_m = np.load(SMPLX_POP_TOOLKIT_DIR + 'SMPLX_facets_fromFPX_male.npy')  # indexing from 1
facets_f = np.load(SMPLX_POP_TOOLKIT_DIR + 'SMPLX_facets_fromFPX_female.npy')  # indexing from 1
# Edited 5 Sept below 1 line
feet = ids_m['leftFoot'] + ids_m['rightFoot'] + ids_m['leftToeBase'] + ids_m['rightToeBase']

skin_uvs_m = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'male_skin_uv1.npy'))
pants_uvs_m = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'male_pants_uv1.npy'))
cloth_uvs_m = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'male_cloth_uv1.npy'))
hair_uvs_m = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'male_hair_uv1.npy'))
bald_uvs = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'male_bald_uv.npy'))
skin_uvs_f = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'female_skin_uv1.npy'))
pants_uvs_f = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'female_pants_uv1.npy'))
cloth_uvs_f = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'female_cloth_uv1.npy'))
hair_uvs_f = np.load(join(SMPLX_POP_TOOLKIT_DIR, 'female_hair_uv1.npy'))
feet_uvs = np.asarray(np.load(join(SMPLX_POP_TOOLKIT_DIR, 'feet_uv1.npy')) * 4096, dtype=int)
MOCAP_DIR = '/dockerspace/Downloads/CMU/'
SUB_DIR = [f for f in listdir(MOCAP_DIR) if isdir(join(MOCAP_DIR, f))]

### get uv mapping for SMPLX body mesh
def read_obj(SMPLX_POP_TOOLKIT_DIR, gender):
    vt_arr = []
    vt_arr.append('vt %f %f\n' % (0.9998, 0.9998))
    with open(SMPLX_POP_TOOLKIT_DIR + 'smplx-' + gender + '_fromFPX.obj', 'r') as FPXfile:
        for line in FPXfile:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            elif values[0] == 'vt':
                vt = list(map(float, values[1:3]))
                vt_arr.append('vt %f %f\n' % (vt[0], vt[1]))

    return vt_arr

contents_m = read_obj(SMPLX_POP_TOOLKIT_DIR, 'male')
contents_f = read_obj(SMPLX_POP_TOOLKIT_DIR, 'female')
FPX_VT = {'male': contents_m, 'female': contents_f}
for i in range(0, facets_m.shape[0]):
    if not (facets_m[i, 0] - 1 in torso_m or facets_m[i, 3] - 1 in torso_m or facets_m[i, 6] - 1 in torso_m):
        # Edited 5 Sept below 10 lines:
        if (facets_m[i, 0] - 1 in feet) or (facets_m[i, 3] - 1 in feet) or (facets_m[i, 6] - 1 in feet):
            contents_m.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                facets_m[i, 0], 1, facets_m[i, 0],
                facets_m[i, 3], 1, facets_m[i, 3],
                facets_m[i, 6], 1, facets_m[i, 6]))  # discarding the stock normals, will calculate new
        else:
            contents_m.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                facets_m[i, 0], facets_m[i, 1] + 1, facets_m[i, 0],
                facets_m[i, 3], facets_m[i, 4] + 1, facets_m[i, 3],
                facets_m[i, 6], facets_m[i, 7] + 1,
                facets_m[i, 6]))  # discarding the stock normals, will calculate new
for i in range(0, facets_f.shape[0]):
    if not (facets_f[i, 0] - 1 in torso_f or facets_f[i, 3] - 1 in torso_f or facets_f[i, 6] - 1 in torso_f):
        # Edited 5 Sept below 10 lines:
        if (facets_f[i, 0] - 1 in feet) or (facets_f[i, 3] - 1 in feet) or (facets_f[i, 6] - 1 in feet):
            contents_f.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                facets_f[i, 0], 1, facets_f[i, 0],
                facets_f[i, 3], 1, facets_f[i, 3],
                facets_f[i, 6], 1, facets_f[i, 6]))  # discarding the stock normals, will calculate new
        else:
            contents_f.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                facets_f[i, 0], facets_f[i, 1] + 1, facets_f[i, 0],
                facets_f[i, 3], facets_f[i, 4] + 1, facets_f[i, 3],
                facets_f[i, 6], facets_f[i, 7] + 1,
                facets_f[i, 6]))  # discarding the stock normals, will calculate new
CONTENTS = {'male': contents_m, 'female': contents_f}
FPX_FACETS = {'male': facets_m, 'female': facets_f}



### get the SMPLX mesh's vertex id for feet vertices
def get_feet():
    ids_m = json.load(open('/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/smplx_vert_segmentation.json', 'r'))
    feet = ids_m['leftFoot'] + ids_m['rightFoot'] + ids_m['leftToeBase'] + ids_m['rightToeBase']
    return feet

### get the pose from AMASS
def pose_from_AMASS_CMU():
    # import the pose
    MOCAP_DIR = '/dockerspace/Downloads/CMU/'
    SUB_DIR = [f for f in listdir(MOCAP_DIR) if isdir(join(MOCAP_DIR, f))]
    motion_folder = random.choice(SUB_DIR)
    motion_folder = osp.join(MOCAP_DIR, motion_folder)
    motion_seq_list = [f for f in listdir(motion_folder) if isfile(join(motion_folder, f))]
    while True:
        motion_seq = random.choice(motion_seq_list)
        if not motion_seq == 'neutral_stagei.npz':
            break
    bdata = np.load(join(motion_folder, motion_seq))
    pose = bdata['pose_body'].reshape((-1, 21, 3))
    pick = random.randint(0, pose.shape[0] - 1)  # randint is inclusive
    return pose[pick, :, :].flatten('C')

### for sheet simulation
class Sheet:
    def __init__(self, vertices, joints):
        self.vertices = vertices
        self.pvs = joints[0]
        sp3 = joints[9]
        vecY = sp3 - self.pvs
        theta = random.randint(1, 360) / 180 * pi
        phi = random.randint(0, 180) / 180 * pi
        vec1 = np.array([sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi)])
        self.normal = np.cross(vec1, vecY)
        self.vecX = np.cross(vecY, self.normal)
        self.coords_c_T = np.array([self.vecX, vecY, self.normal])

    def get_frame_sheet_T(self):
        return self.coords_c_T

    def carve(self, sparsity, write=False):
        assert 0.05 <= sparsity <= 0.5, 'sparsity must in [0.05, 0.5]'
        verts = self.vertices - self.pvs
        dots = np.dot(verts, self.normal)
        val_ids = np.where(dots > 0)
        st = o3d.geometry.PointCloud()
        sheet_pos_v = self.vertices[val_ids] + random.uniform(0.1, 0.5) * self.normal
        sheet_pos_n = np.ones_like(sheet_pos_v) * self.normal
        st.points = o3d.utility.Vector3dVector(sheet_pos_v)
        st.normals = o3d.utility.Vector3dVector(sheet_pos_n)
        st = st.voxel_down_sample(voxel_size=sparsity)  # sparsify the points
        st.remove_radius_outlier(nb_points=200, radius=5*sparsity)
        temp = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(st, depth=9, linear_fit=False)[0]
        temp = temp.filter_smooth_simple(number_of_iterations=2)
        print(np.asarray(temp.vertices).shape)
        if write:
            o3d.io.write_triangle_mesh('/dockerspace/BigBench/weirdshit/smplx_1102_t.obj', temp)
        return temp

### create the basic SMPLX point cloud
def smplx_create(gender, model_folder, model_type, num_betas, ext):
    bob = smplx.create(model_folder, model_type=model_type,
                       gender=gender, use_face_contour=USE_FACE_CONTOUR,
                       num_betas=num_betas, ext=ext, num_expression_coeffs=10)
    return bob

### generate the posmap for POP
def posmap(posmap_res, vertices, gender):
    pmap = np.ndarray((posmap_res, posmap_res, 3), dtype=float)
    vuv = np.load('/dockerspace/Workspace/POPcustomized/POP-main/POP_app/POP_SMPLX_assets/v_vt_u_v_SMPLX_fromFPX.npy')
    vuv[:, -2:] = vuv[:, -2:] * posmap_res
    mask = np.load(  # mask is a 2d array
        '/dockerspace/Workspace/POPcustomized/POP-main/assets/uv_masks/uv_mask' + str(
            posmap_res) + '_with_faceid_smplx.npy')
    vid_mask = np.load(SMPLX_POP_TOOLKIT_DIR + 'vid_mask_' + str(posmap_res) + gender + '.npy')
    for v in range(0, posmap_res):  # v vertical, u horizontal
        for u in range(0, posmap_res):
            if mask[v, u] == -1:
                pmap[v, u, :] = np.array([0, 0, 0])
            else:
                n4 = list(map(int, vid_mask[v, u, :]))
                pmap[v, u, :] = np.mean(vertices[n4, :], axis=0)
    return pmap

### create a basic SMPLX point cloud without torso vertices
def create_torsoless(verts, gender, file_name=None, anchor_id=None, deformation=True, dummy_rot=np.array([[1,0,0],[0,1,0],[0,0,1]]), dummy_trans=np.array([0,0,0])):  # have to estimate the normals
    torsoless_pcd = o3d.geometry.PointCloud()
    torsoless_pcd.points = o3d.utility.Vector3dVector(verts)
    torsoless_pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    torsoless_pcd.estimate_normals()
    torsoless_pcd.orient_normals_consistent_tangent_plane(100)
    vn_arr = np.asarray(torsoless_pcd.normals)
    verts = np.matmul(dummy_rot, verts.T).T + dummy_trans  # rotate the mesh
    vn_arr = np.matmul(dummy_rot, vn_arr.T).T
    contents = []
    for vi in range(0, verts.shape[0]):
        contents.append('v %f %f %f\n' % (verts[vi, 0], verts[vi, 1], verts[vi, 2]))
        contents.append('vn %f %f %f\n' % (vn_arr[vi, 0], vn_arr[vi, 1], vn_arr[vi, 2]))
    contents += CONTENTS[gender]
    if deformation:
        pass
    else:
        for fi in range(0, FPX_FACETS[gender].shape[0]):  # though all vertices are stored, only torsoless facets are defined
            if (FPX_FACETS[gender][fi, 0]-1 in torso_m) or (FPX_FACETS[gender][fi, 3]-1 in torso_m) or (FPX_FACETS[gender][fi, 6]-1 in torso_m):
                contents.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                    FPX_FACETS[gender][fi, 0], FPX_FACETS[gender][fi, 1] + 1, FPX_FACETS[gender][fi, 0],
                    FPX_FACETS[gender][fi, 3], FPX_FACETS[gender][fi, 4] + 1, FPX_FACETS[gender][fi, 3],
                    FPX_FACETS[gender][fi, 6], FPX_FACETS[gender][fi, 7] + 1, FPX_FACETS[gender][fi, 6]))  # discarding the stock normals, will calculate new
    with open(file_name, 'w') as file:
        file.writelines(contents)
    print('body mesh written')
    if anchor_id is not None:
        return vn_arr[anchor_id, :]
    else:
        return None

### mount the wig onto the head
def barber(hair_file, anchor_h, joints, m_id, dummy_rot=np.array([[1,0,0],[0,1,0],[0,0,1]]), dummy_trans=np.array([0,0,0])):
    v_arr = []
    vn_arr = []
    f_list = []
    with open(hair_file, 'r') as hair_mesh:
        for line in hair_mesh:
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            elif values[0] == 'v':
                v = list(map(float, values[1:]))
                v_arr.append(v)
            elif values[0] == 'vn':
                vn = list(map(float, values[1:]))
                vn_arr.append(vn)
            elif values[0] == 'f':
                f_list.append(list(map(lambda x: x.split('/'), values[1:])))
    verts = np.asarray(v_arr)

    dv = verts[:, 0] * verts[:, 0] + verts[:, 2] * verts[:, 2]
    od = np.argsort(dv)[:4]
    pivot = np.mean(verts[od, :], axis=0)
    verts = verts - pivot

    # Build a new frame within the head
    earL1, earR1, nose1 = joints[59, :], joints[58, :], joints[55, :]
    X_AXIS1 = (earL1-earR1) / np.linalg.norm(earL1-earR1)
    nose1_proj = np.dot(X_AXIS1, nose1-earR1)
    origin1 = earR1 + X_AXIS1 * nose1_proj
    Z_AXIS1 = (nose1-origin1) / np.linalg.norm(nose1-origin1)
    Y_AXIS1 = np.cross(Z_AXIS1, X_AXIS1)
    FRAME1 = np.array([X_AXIS1, Y_AXIS1, Z_AXIS1])
    FRAMEG = np.array([[1,0,0], [0,1,0], [0,0,1]])
    ROT_G_1 = np.linalg.solve(FRAMEG, FRAME1).T
    verts = np.matmul(ROT_G_1, verts.T).T
    noms = np.asarray(vn_arr)
    noms = np.matmul(ROT_G_1, noms.T).T
    verts = verts + anchor_h  # - hair_summit
    verts = np.matmul(dummy_rot, verts.T).T + dummy_trans
    noms = np.matmul(dummy_rot, noms.T).T
    lines = []
    for i in range(0, verts.shape[0]):
        lines.append('v %f %f %f\n' % (verts[i, 0], verts[i, 1], verts[i, 2]))
    for i in range(0, noms.shape[0]):
        lines.append('vn %f %f %f\n' % (noms[i, 0], noms[i, 1], noms[i, 2]))
    lines.append(' vt 0.5 0.5\n')  # replace the original vt with a constant, the hair will be colored uniformly
    for i in range(0, len(f_list)):
        if len(f_list[i]) == 3:
            lines.append('f ' + f_list[i][0][0] + '/1/' + f_list[i][0][2] + ' '
                         + f_list[i][1][0] + '/1/' + f_list[i][1][2] + ' '
                         + f_list[i][2][0] + '/1/' + f_list[i][2][2] + '\n')
        elif len(f_list[i]) == 4:
            lines.append('f ' + f_list[i][0][0] + '/1/' + f_list[i][0][2] + ' '
                         + f_list[i][1][0] + '/1/' + f_list[i][1][2] + ' '
                         + f_list[i][2][0] + '/1/' + f_list[i][2][2] + ' '
                         + f_list[i][3][0] + '/1/' + f_list[i][3][2] + '\n')
    with open(join(RESULT_DIR, 'wig' + str(m_id) + '.obj'), 'w') as file:
        file.writelines(lines)
    print('wig mesh written')
    return None

### generate a random rotation matrix
def dummy_rotation():
    new_x_axis = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    assist_vec = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    while not np.all(assist_vec - new_x_axis):
        assist_vec = np.array([random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)])
    new_z_axis = np.cross(new_x_axis, assist_vec)
    new_y_axis = np.cross(new_z_axis, new_x_axis)
    new_x_axis = new_x_axis / np.linalg.norm(new_x_axis)
    new_y_axis = new_y_axis / np.linalg.norm(new_y_axis)
    new_z_axis = new_z_axis / np.linalg.norm(new_z_axis)
    FRAME1 = np.array([new_x_axis, new_y_axis, new_z_axis])
    FRAMEG = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ROT_G_1 = np.linalg.solve(FRAMEG, FRAME1).T
    return ROT_G_1

### generate the sheet mesh
def sheet_gen(vertices, joints, bob, betas, rotter, trans):
    pose_0 = np.zeros((21, 3), dtype=float)
    output = bob(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0), body_pose=torch.tensor(pose_0, dtype=torch.float32).unsqueeze(0))
    joints_0 = output['joints'].detach().cpu().numpy().squeeze(0)
    joints_0 = joints_0 - joints_0[0]
    spine2 = joints_0[6, :]
    l_elbow = joints_0[18, :]
    l_knee = joints_0[4, :]
    r_elbow = joints_0[19, :]
    l_ankle = joints_0[7, :]
    neck = joints_0[12, :]
    bound_min = np.asarray([r_elbow[0], random.uniform(l_knee[1], l_ankle[1]), -0.5],
                           dtype=float)  # reference for mesh cropping
    bound_max = np.asarray([l_elbow[0], random.uniform(spine2[1], neck[1]), 0.5], dtype=float)
    global_frame_T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    root = joints[0, :]
    vertices = vertices - joints[0, :]
    joints = joints - joints[0]
    pelvis = joints[0, :]
    spine3 = joints[9, :]
    r_collar = joints[14, :]
    l_collar = joints[13, :]
    x = (l_collar - r_collar) / np.linalg.norm(l_collar - r_collar)
    y = (spine3 - pelvis) / np.linalg.norm(spine3 - pelvis)
    z = np.cross(x, y)
    x = np.cross(y, z)
    body_frame_T = np.asarray([x, y, z])

    duvet = Sheet(vertices, joints)
    mesh = duvet.carve(0.05)
    v_b = np.asarray(mesh.vertices)
    b2g = np.linalg.solve(body_frame_T,
                          global_frame_T).T  # rotate the vertices to make it align with the T-pose directions
    g2b = np.linalg.solve(global_frame_T, body_frame_T).T
    v_g = np.matmul(b2g, v_b.T).T  # rotated vertices
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=bound_min, max_bound=bound_max)
    mesh.vertices = o3d.utility.Vector3dVector(v_g)
    mesh = mesh.crop(box)
    v_g = np.asarray(mesh.vertices)  # note v_g has fewer points than v_b does
    triangles = np.asarray(mesh.triangles).flatten('C')
    sheet_frame_T = duvet.get_frame_sheet_T()

    v_b = np.matmul(g2b, v_g.T).T + root
    s2g = np.linalg.solve(sheet_frame_T, global_frame_T).T
    v_s = np.matmul(s2g, v_b.T).T
    uv = v_s[:, :2] - np.min(v_s[:, :2], axis=0)
    ratio = 1 / np.max(uv[:, :2], axis=0)
    uv = uv * ratio
    triangle_uvs = uv[triangles]
    v_b = np.matmul(rotter, v_b.T).T + trans
    n_b = np.asarray(mesh.vertex_normals)
    n_b = np.matmul(rotter, n_b.T).T
    mesh.vertex_normals = o3d.utility.Vector3dVector(n_b)
    mesh.vertices = o3d.utility.Vector3dVector(v_b)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    return mesh




def main():
    args = parse_config()
    flist_uv, valid_idx, uv_coord_map = load_masks(dirname(realpath(__file__)), 256, body_model='smplx')
    bary_coords = load_barycentric_coords(dirname(realpath(__file__)), 256, body_model='smplx')
    model = POP(
        input_nc=3,
        c_pose=args.c_pose,
        c_geom=args.c_geom,
        inp_posmap_size=args.inp_posmap_size,  # default = 128
        hsize=args.hsize,
        nf=args.nf,
        up_mode=args.up_mode,
        use_dropout=bool(args.use_dropout),
        pos_encoding=bool(args.pos_encoding),
        num_emb_freqs=args.num_emb_freqs,
        posemb_incl_input=bool(args.posemb_incl_input),
        uv_feat_dim=2,
        geom_layer_type=args.geom_layer_type,
        gaussian_kernel_size=args.gaussian_kernel_size,
    )
    outfits_num = 12
    geom_featmap = torch.ones(outfits_num, args.c_geom, args.inp_posmap_size, args.inp_posmap_size).normal_(mean=0.,
                                                                                                            std=0.01).cuda()
    geom_featmap.requires_grad = True
    subpixel_sampler = SampleSquarePoints(npoints=1)
    model_config = {
        'device': torch.device('cuda'),
        'flist_uv': flist_uv,
        'valid_idx': valid_idx,
        'uv_coord_map': uv_coord_map,
        'bary_coords_map': bary_coords,
        'transf_scaling': args.transf_scaling,
    }
    pretrained_resynth_pop = torch.load('POP_pretrained_ReSynthdata_12outfits_epoch00400_model.pt')
    pretrained_resynth_geom = 'POP_pretrained_ReSynthdata_12outfits_epoch00400_geom_featmap.pt'
    model.load_state_dict(pretrained_resynth_pop['model_state'])
    load_latent_feats(pretrained_resynth_geom, geom_featmap)
    model.to('cuda')
    model.eval()

    # load cloth mesh information including point id and uv mapping
    with open(SMPLX_POP_TOOLKIT_DIR + 'cloth_mesh_info.json') as file:
        cloth_info = json.load(file)

    # load wigs
    HAIR_DIR = '/dockerspace/BigBench/Workspace/Gallery/Meshes/PreparedHair'
    HAIR_FILES = [f for f in listdir(HAIR_DIR) if isfile(join(HAIR_DIR, f))]

    vv = 0
    while vv < MAX_ITERATION:
        myfile = join(RESULT_DIR, STATUS_FILE)
        fd = os.open(myfile, os.O_RDONLY)
        status_valid = False
        if os.fstat(fd).st_size:
            os.close(fd)
            with open(myfile, 'r') as mf:
                for line in mf:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    sv = line.split()
                    if not sv:
                        status_valid = False
                    else:
                        svlist = list(map(float, sv[0:]))
                        if len(svlist) > 0:
                            status_valid = True
            if not status_valid:
                pass
            elif svlist[0] == 0:  # 0 means last batch has been processed
                print('########## iteration ' + str(vv) + '##########')
                mesh_info = []  # content format: 0/1(gender) 0/1/2/3/4(mesh No.) 0/1/2/3(Scenario)
                person_count = random.randint(1, MAX_HUMAN_IN_PIC)
                for m_idx in range(0, person_count):
                    dummy_rotter = dummy_rotation()
                    dummy_transl = np.array([random.uniform(-1, 1), random.uniform(0, 0.2), random.uniform(-1, 1)])
                    # dummy_rotter = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                    # dummy_transl = np.array([0, 0, 0])
                    # unit: metre; the altitude difference between meshes shouldn't be too large
                    if random.choice([True,
                                      False]):  # randomly decide whether to initiate the POP pipeline, it's either scenario 0&1 or 2&3
                        # load the pose
                        R_POSE = pose_from_AMASS_CMU()
                        # choose gender, hairstyle, and cloth randomly
                        gender = random.choice(['male', 'female'])
                        if gender == 'male':
                            hair_style = random.choice(['bald', 'semi-bald', 'wig', 'stock'])
                        else:
                            hair_style = random.choice(['wig', 'stock'])
                        CLO = random.randint(0, 11)
                        # initialize the smplx creator
                        mannequin = smplx_create(gender, MODEL_FOLDER, MODEL_TYPE, NUM_BETAS, EXT)
                        # below is the mean/average SMPLX output
                        output_m = mannequin(betas=BETAS_MEAN,
                                             body_pose=torch.tensor(R_POSE, dtype=torch.float32).unsqueeze(0))
                        vertices_m = output_m['vertices'].detach().cpu().numpy().squeeze(0)
                        p128 = posmap(128, vertices_m, gender)
                        # below is the wanted SMPLX output
                        # betas = torch.randn([1, NUM_BETAS], dtype=torch.float32) # * 1.5  # random body shape
                        betas = (np.random.rand(NUM_BETAS) - 0.5) * 2
                        output = mannequin(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                                           body_pose=torch.tensor(R_POSE, dtype=torch.float32).unsqueeze(0))
                        vertices = output['vertices'].detach().cpu().numpy().squeeze(0)
                        pos_vtransf = output['vtransf'].detach().cpu().numpy().squeeze(0)
                        joints = output['joints'].detach().cpu().numpy().squeeze(0)
                        earL1_o, earR1_o, nose1_o = joints[59, :], joints[58, :], joints[55, :]  # for glasses mounting
                        p256 = posmap(256, vertices, gender)
                        anchor_id = None
                        # prepare the full body uv texture
                        full_map = cv2.imread(
                            '/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/smplx_texture_' + gender + '.png')
                        if gender == 'male':
                            skin_uvs = skin_uvs_m
                            pants_uvs = pants_uvs_m
                            cloth_uvs = cloth_uvs_m
                            hair_uvs = hair_uvs_m
                        else:
                            skin_uvs = skin_uvs_f
                            pants_uvs = pants_uvs_f
                            cloth_uvs = cloth_uvs_f
                            hair_uvs = hair_uvs_f
                        # select colour for skin&hair
                        pick1 = random.randint(0, BGR_SKIN_TONE.shape[0] - 1)
                        pick2 = random.randint(0, BGR_SKIN_TONE.shape[0] - 1)
                        skin_c = random.uniform(0, 1) * (
                                    BGR_SKIN_TONE[pick1, :] - BGR_SKIN_TONE[pick2, :]) + BGR_SKIN_TONE[
                                                                                         pick2, :]
                        hair_c = RGB_HAIR[random.choice(list(RGB_HAIR.keys()))]  # RGB-->BGR used by cv2
                        hair_c.reverse()
                        full_map[skin_uvs[:, 0], skin_uvs[:, 1], :] = skin_c
                        full_map[pants_uvs[:, 0], pants_uvs[:, 1], :] = skin_c
                        full_map[cloth_uvs[:, 0], cloth_uvs[:, 1], :] = skin_c
                        full_map[hair_uvs[:, 0], hair_uvs[:, 1], :] = hair_c
                        # below deals with the hair
                        if hair_style == 'wig':  # Scenario 3
                            # below is a mannequin for computing the anchor of wigs
                            output_h = mannequin(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                                                 body_pose=torch.tensor(T_POSE, dtype=torch.float32).unsqueeze(0))
                            joints_h = output_h['joints'].detach().cpu().numpy().squeeze(0)
                            vertices_h = output_h['vertices'].detach().cpu().numpy().squeeze(0)
                            anchor_id = np.argmax(vertices_h[:, 1])
                            anchor_h = vertices[anchor_id, :]
                            bias_h = create_torsoless(vertices, gender,
                                                      file_name=join(RESULT_DIR, 'body' + str(m_idx) + '.obj'),
                                                      anchor_id=anchor_id, dummy_rot=dummy_rotter,
                                                      dummy_trans=dummy_transl)
                            hair_f = join(HAIR_DIR, random.choice(HAIR_FILES))
                            barber(hair_f, anchor_h, joints, m_idx, dummy_rot=dummy_rotter, dummy_trans=dummy_transl)
                            # uv texture map for wig
                            c_map = np.ones((2, 2, 3)) * hair_c
                            cv2.imwrite(join(RESULT_DIR, 'wig_uv' + str(m_idx) + '.png'), c_map)
                            print('wig uv written')
                            scenario = 3
                            print('Scenario 3')
                        else:  # Scenario 2
                            # create a SMPLX mesh without torso
                            create_torsoless(vertices, gender, file_name=join(RESULT_DIR, 'body' + str(m_idx) + '.obj'),
                                             dummy_rot=dummy_rotter, dummy_trans=dummy_transl)
                            if hair_style == 'bald':
                                full_map[hair_uvs[:, 0], hair_uvs[:, 1], :] = skin_c
                            elif hair_style == 'semi-bald':
                                full_map[bald_uvs[:, 0], bald_uvs[:, 1], :] = skin_c
                            scenario = 2
                            print('Scenario 2')
                        bg_pic = cv2.imread(join(bg_dir, random.choice(bg_ls)))  # pattern to load onto the cloth
                        if bg_pic.shape[0] <= 1000 and bg_pic.shape[1] <= 1000:
                            bg_pic = np.tile(bg_pic, (4, 4, 1))
                        elif bg_pic.shape[0] <= 2000 and bg_pic.shape[1] <= 2000:
                            bg_pic = np.tile(bg_pic, (2, 2, 1))
                        cv2.imwrite(join(RESULT_DIR, 'cloth_uv' + str(m_idx) + '.png'), bg_pic)
                        print('cloth uv written')
                        brandom = random.randint(0, 40) - 20
                        grandom = random.randint(0, 40) - 20
                        rrandom = random.randint(0, 40) - 20
                        feet_c = random.choice(BGR_SHOE) + [brandom, grandom, rrandom]
                        full_map[:5, -5:, :] = random.choice([feet_c, skin_c, feet_c])
                        cv2.imwrite(join(RESULT_DIR, 'full_body_uv' + str(m_idx) + '.png'), full_map)
                        print('body uv written')
                        # point cloud by POP
                        index = torch.tensor([CLO]).cuda()
                        geom_featmap_batch = geom_featmap[index, ...]
                        inp_posmap0, query_posmap0 = [], []
                        query_posmap0.append(torch.tensor(p256).float().permute([2, 0, 1]))
                        query_posmap = query_posmap0[0].to('cuda', non_blocking=True)
                        inp_posmap0.append(torch.tensor(p128).float().permute([2, 0, 1]))  # posed, mean shape
                        inp_posmap = inp_posmap0[0].to('cuda', non_blocking=True)
                        query_posmap = query_posmap.expand(1, -1, -1, -1)
                        inp_posmap = inp_posmap.expand(1, -1, -1, -1)
                        bs, _, H, W = query_posmap.size()
                        uv_coord_map_batch = uv_coord_map.expand(bs, -1, -1).contiguous()
                        vtransf = torch.tensor(pos_vtransf).float()
                        if vtransf.shape[-1] == 4:
                            vtransf = vtransf[:, :3, :3]
                        vtransf = vtransf.expand(1, -1, -1, -1).contiguous().to('cuda', non_blocking=True)
                        transf_mtx_map = gen_transf_mtx_from_vtransf(vtransf, bary_coords, flist_uv,
                                                                     scaling=args.transf_scaling)
                        pq_samples = subpixel_sampler.sample_regular_points()
                        pq_repeated = pq_samples.expand(bs, H * W, -1, -1)
                        N_subsample = 1
                        bp_locations = query_posmap.expand(N_subsample, -1, -1, -1, -1).permute(
                            [1, 2, 3, 4, 0])  # bs, C, H, W, N_sample
                        transf_mtx_map = transf_mtx_map.expand(N_subsample, -1, -1, -1, -1, -1).permute(
                            [1, 2, 3, 0, 4, 5])  # [bs, H, W, N_subsample, 3, 3]
                        with torch.no_grad():
                            pred_res, pred_normals = model(inp_posmap,
                                                           # mean shape, posed body positional maps as input to the network
                                                           geom_featmap=geom_featmap_batch,
                                                           uv_loc=uv_coord_map_batch,
                                                           pq_coords=pq_repeated  # always 0 in POP
                                                           )

                            # local coords --> global coords
                            pred_res = pred_res.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
                            pred_normals = pred_normals.permute([0, 2, 3, 4, 1]).unsqueeze(-1)
                            pred_res = torch.matmul(transf_mtx_map, pred_res).squeeze(-1)
                            pred_normals = torch.matmul(transf_mtx_map, pred_normals).squeeze(-1)
                            pred_normals = torch.nn.functional.normalize(pred_normals, dim=-1)
                            # residual to abosolute locations in space
                            full_pred = pred_res.permute([0, 4, 1, 2, 3]).contiguous() + bp_locations
                            displacement = pred_res.permute([0, 4, 1, 2, 3]).contiguous()
                            # take the selected points and reshape to [N_valid_points, 3]
                            full_pred = full_pred.permute([0, 2, 3, 4, 1]).reshape(bs, -1, N_subsample, 3)[:, valid_idx,
                                        ...]
                            full_pred = full_pred.reshape(bs, -1, 3).contiguous()
                            pred_normals = pred_normals.reshape(bs, -1, N_subsample, 3)[:, valid_idx, ...]
                            pred_normals = pred_normals.reshape(bs, -1, 3).contiguous()
                            displacement = displacement.permute([0, 2, 3, 4, 1]).reshape(bs, -1, N_subsample, 3)[:,
                                           valid_idx, ...]
                            displacement = displacement.reshape(bs, -1, 3).contiguous()
                        full_pred = full_pred.detach().cpu().numpy().squeeze(0)
                        pred_normals = pred_normals.detach().cpu().numpy().squeeze(0)
                        displacement = displacement.detach().cpu().numpy().squeeze(0)
                        clo_idx = cloth_info[gender]['ids'][CLO]
                        clo_u = cloth_info[gender]['u_coords'][CLO]
                        clo_v = cloth_info[gender]['v_coords'][CLO]
                        pcd_c = np.concatenate((np.array([clo_u]).transpose(), np.array([clo_v]).transpose(),
                                                np.zeros([len(clo_u), 1], dtype=float)), axis=1)
                        pcd_v = full_pred[clo_idx, :]
                        pcd_n = pred_normals[clo_idx, :]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pcd_v)
                        pcd.normals = o3d.utility.Vector3dVector(pcd_n)
                        pcd.colors = o3d.utility.Vector3dVector(pcd_c)
                        pcd.remove_radius_outlier(nb_points=50, radius=0.05)
                        poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7, width=0,
                                                                                                 scale=1.1,
                                                                                                 linear_fit=False)[0]
                        mesh_out = poisson_mesh.filter_smooth_simple(number_of_iterations=3)
                        vets = np.array(mesh_out.vertices)
                        noms = np.array(mesh_out.vertex_normals)
                        vets = vets + inflate_coef * noms  # inflates the mesh
                        colo = np.array(mesh_out.vertex_colors)
                        facets = np.asarray(mesh_out.triangles)
                        facets = facets + np.ones_like(facets)
                        vets = np.matmul(dummy_rotter, vets.T).T + dummy_transl
                        noms = np.matmul(dummy_rotter, noms.T).T
                        # writing up the cloth mesh
                        lines = []
                        for i in range(0, vets.shape[0]):
                            lines.append('v %f %f %f\n' % (vets[i, 0], vets[i, 1], vets[i, 2]))
                            lines.append('vt %f %f\n' % (colo[i, 0], colo[i, 1]))
                            lines.append('vn %f %f %f\n' % (noms[i, 0], noms[i, 1], noms[i, 2]))
                        for i in range(0, facets.shape[0]):
                            lines.append('f %d/%d/%d %d/%d/%d %d/%d/%d\n' % (
                                facets[i, 0], facets[i, 0], facets[i, 0],
                                facets[i, 1], facets[i, 1], facets[i, 1],
                                facets[i, 2], facets[i, 2], facets[i, 2]))
                        with open(join(RESULT_DIR, 'cloth' + str(m_idx) + '.obj'), 'w') as file:
                            file.writelines(lines)
                        print('cloth mesh written')
                    else:
                        # load the pose
                        R_POSE = pose_from_AMASS_CMU()
                        # choose gender, hairstyle, and cloth randomly
                        gender = random.choice(['male', 'female'])
                        if gender == 'male':
                            hair_style = random.choice(['bald', 'semi-bald', 'wig', 'stock'])
                        else:
                            hair_style = random.choice(['wig', 'stock'])

                        mannequin = smplx_create(gender, MODEL_FOLDER, MODEL_TYPE, NUM_BETAS, EXT)
                        betas = (np.random.rand(NUM_BETAS) - 0.5) * 2
                        output = mannequin(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                                           body_pose=torch.tensor(R_POSE, dtype=torch.float32).unsqueeze(0))
                        vertices = output['vertices'].detach().cpu().numpy().squeeze(0)
                        joints = output['joints'].detach().cpu().numpy().squeeze(0)
                        earL1_o, earR1_o, nose1_o = joints[59, :], joints[58, :], joints[55, :]  # for glasses mounting
                        anchor_id = None
                        # prepare the full body uv texture
                        full_map = cv2.imread(
                            '/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/smplx_texture_' + gender + '.png')
                        if gender == 'male':
                            skin_uvs = skin_uvs_m
                            pants_uvs = pants_uvs_m
                            cloth_uvs = cloth_uvs_m
                            hair_uvs = hair_uvs_m
                        else:
                            skin_uvs = skin_uvs_f
                            pants_uvs = pants_uvs_f
                            cloth_uvs = cloth_uvs_f
                            hair_uvs = hair_uvs_f
                        pick1 = random.randint(0, BGR_SKIN_TONE.shape[0] - 1)
                        pick2 = random.randint(0, BGR_SKIN_TONE.shape[0] - 1)
                        skin_c = random.uniform(0, 1) * (
                                    BGR_SKIN_TONE[pick1, :] - BGR_SKIN_TONE[pick2, :]) + BGR_SKIN_TONE[
                                                                                         pick2, :]
                        hair_c = RGB_HAIR[random.choice(list(RGB_HAIR.keys()))]  # RGB-->BGR used by cv2
                        hair_c.reverse()
                        full_map[skin_uvs[:, 0], skin_uvs[:, 1], :] = skin_c
                        full_map[hair_uvs[:, 0], hair_uvs[:, 1], :] = hair_c
                        # Colouring the cloth and pants
                        # since not using POP, a SMPLX-UV outfit is needed, ergo it is inappropriate put skin colour on the cloth/pants
                        mask = np.zeros((4096, 4096), dtype=np.uint8)
                        # mask out the cloth first
                        mask[cloth_uvs[:, 0], cloth_uvs[:, 1]] = 255
                        full_map = np.asarray(full_map, dtype=np.uint8)
                        foreground = cv2.bitwise_and(full_map, full_map, mask=mask)
                        bg_pic = cv2.imread(join(bg_dir, random.choice(bg_ls)))  # pattern to load
                        if bg_pic.shape[0] <= 1000 and bg_pic.shape[1] <= 1000:
                            bg_pic = np.tile(bg_pic, (4, 4, 1))
                        elif bg_pic.shape[0] <= 2000 and bg_pic.shape[1] <= 2000:
                            bg_pic = np.tile(bg_pic, (2, 2, 1))
                        bg_pic = cv2.resize(bg_pic, (4096, 4096), interpolation=cv2.INTER_AREA)
                        full_map = np.where(foreground == 0, full_map, bg_pic)
                        # mask out the cloth
                        mask = np.zeros((4096, 4096), dtype=np.uint8)
                        mask[pants_uvs[:, 0], pants_uvs[:, 1]] = 255
                        foreground = cv2.bitwise_and(full_map, full_map, mask=mask)
                        bg_pic = cv2.imread(join(bg_dir, random.choice(bg_ls)))  # pattern to load
                        if bg_pic.shape[0] <= 1000 and bg_pic.shape[1] <= 1000:
                            bg_pic = np.tile(bg_pic, (4, 4, 1))
                        elif bg_pic.shape[0] <= 2000 and bg_pic.shape[1] <= 2000:
                            bg_pic = np.tile(bg_pic, (2, 2, 1))
                        bg_pic = cv2.resize(bg_pic, (4096, 4096), interpolation=cv2.INTER_AREA)
                        full_map = np.where(foreground == 0, full_map, bg_pic)
                        # below deals with the hair
                        if hair_style == 'wig':  # Scenario 1
                            # below is a mannequin for computing the anchor of wigs
                            output_h = mannequin(betas=torch.tensor(betas, dtype=torch.float32).unsqueeze(0),
                                                 body_pose=torch.tensor(T_POSE, dtype=torch.float32).unsqueeze(0))
                            joints_h = output_h['joints'].detach().cpu().numpy().squeeze(0)
                            vertices_h = output_h['vertices'].detach().cpu().numpy().squeeze(0)
                            anchor_id = np.argmax(vertices_h[:, 1])
                            anchor_h = vertices[anchor_id, :]
                            bias_h = create_torsoless(vertices, gender,
                                                      file_name=join(RESULT_DIR, 'body' + str(m_idx) + '.obj'),
                                                      deformation=False, anchor_id=anchor_id, dummy_rot=dummy_rotter,
                                                      dummy_trans=dummy_transl)
                            hair_f = join(HAIR_DIR, random.choice(HAIR_FILES))
                            barber(hair_f, anchor_h, joints, m_idx, dummy_rot=dummy_rotter, dummy_trans=dummy_transl)
                            # uv texture map for wig
                            c_map = np.ones((2, 2, 3)) * hair_c
                            cv2.imwrite(join(RESULT_DIR, 'wig_uv' + str(m_idx) + '.png'), c_map)
                            print('wig uv written')
                            scenario = 1
                            print('Scenario 1')
                        else:  # Scenario 0
                            # create a SMPLX mesh without torso
                            create_torsoless(vertices, gender, file_name=join(RESULT_DIR, 'body' + str(m_idx) + '.obj'),
                                             deformation=False, dummy_rot=dummy_rotter, dummy_trans=dummy_transl)
                            if hair_style == 'bald':
                                full_map[hair_uvs[:, 0], hair_uvs[:, 1], :] = skin_c
                            elif hair_style == 'semi-bald':
                                full_map[bald_uvs[:, 0], bald_uvs[:, 1], :] = skin_c
                            scenario = 0
                            print('Scenario 0')
                        brandom = random.randint(0, 40) - 20
                        grandom = random.randint(0, 40) - 20
                        rrandom = random.randint(0, 40) - 20
                        feet_c = random.choice(BGR_SHOE) + [brandom, grandom, rrandom]
                        full_map[:5, -5:, :] = random.choice([feet_c, skin_c, feet_c])
                        cv2.imwrite(join(RESULT_DIR, 'full_body_uv' + str(m_idx) + '.png'), full_map)
                        print('body uv written')
                    # generate sheet
                    # if random.choice([True, True, False]):
                    if random.choice([True]):
                        sheet_en = 1
                        sheet_mesh = sheet_gen(vertices=vertices, joints=joints, bob=mannequin, betas=betas, rotter=dummy_rotter, trans=dummy_transl)
                        o3d.io.write_triangle_mesh(join(RESULT_DIR, 'sheet' + str(m_idx) + '.obj'), sheet_mesh)
                        sheet_pic = cv2.imread(join(SHEET_DIR, random.choice(SHEET_TXTs)))  # pattern to load
                        if sheet_pic.shape[0] <= 300 and sheet_pic.shape[1] <= 300:
                            sheet_pic = np.tile(sheet_pic, (4, 4, 1))
                        elif sheet_pic.shape[0] <= 500 and bg_pic.shape[1] <= 500:
                            sheet_pic = np.tile(sheet_pic, (2, 2, 1))
                        cv2.imwrite(join(RESULT_DIR, 'sheet_uv' + str(m_idx) + '.png'), sheet_pic)
                        print('sheet mesh and uv written')
                    else:
                        sheet_en = 0
                    # get the joints position of current mesh
                    joints_op = joints[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                        20, 21, 21, 21, 55, 57, 56, 58, 59], :]
                    # 29 joints same as the number used in SMPL-POP,
                    # use 3 joint21 because I am too lazy to find the palms,
                    #  they won't be used anyway
                    joints_op = np.matmul(dummy_rotter,
                                          joints_op.T).T + dummy_transl  # update the joints position accordingly
                    with open(join(RESULT_DIR, 'Jpo' + str(m_idx) + '.txt'), 'w') as file:
                        for i in range(0, joints_op.shape[0]):
                            file.write('%f %f %f\n' % (
                                -joints_op[i, 0], joints_op[i, 1],
                                joints_op[i, 2]))  # -joints_op[i, 0] is VERY IMPORTANT!!!
                    print('jpo written')

                    # fit the glasses
                    if random.choice([True, True, False]):
                        glasses_en = 1
                        X_AXIS1 = (earL1_o - earR1_o) / np.linalg.norm(earL1_o - earR1_o)
                        nose1_proj = np.dot(X_AXIS1, nose1_o - earR1_o)
                        origin1 = earR1_o + X_AXIS1 * nose1_proj
                        Z_AXIS1 = (nose1_o - origin1) / np.linalg.norm(nose1_o - origin1)
                        Y_AXIS1 = np.cross(Z_AXIS1, X_AXIS1)
                        FRAME1 = np.array([X_AXIS1, Y_AXIS1, Z_AXIS1])
                        FRAMEG = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                        ROT_G_1 = np.linalg.solve(FRAMEG, FRAME1).T
                        verts, vt_s, noms, vf_s = mesh_read(random.choice(glasses_ls))
                        verts, _ = v_normalise(v_arr=verts, ear_l=earL1_o, ear_r=earR1_o)
                        # print(ROT_G_1.shape)
                        # print(verts.shape)
                        verts = np.matmul(ROT_G_1, verts.T).T
                        noms = np.asarray(noms)
                        # print(noms.shape)
                        noms = np.matmul(ROT_G_1, noms.T).T
                        verts = verts + nose1_o + [0, 0.01, -0.02]  # glasses on nose
                        verts = np.matmul(dummy_rotter, verts.T).T + dummy_transl
                        noms = np.matmul(dummy_rotter, noms.T).T
                        mesh_write(join(RESULT_DIR, 'glasses' + str(m_idx) + '.obj'), v_arr=verts, vt_s=vt_s,
                                   vn_arr=noms, f_s=vf_s)
                        brandom = random.randint(0, 16) - 8
                        grandom = random.randint(0, 16) - 8
                        rrandom = random.randint(0, 16) - 8
                        glasses_uv = np.ones((5, 5, 3)) * (random.choice(BGR_GLASSES) + [brandom, grandom, rrandom])
                        cv2.imwrite(join(RESULT_DIR, 'glasses_uv' + str(m_idx) + '.png'), glasses_uv)
                    else:
                        glasses_en = 0

                    mesh_info.append('%d %d %d %d %d\n' % (gender == 'female', m_idx, scenario, glasses_en, sheet_en))
                with open(join(RESULT_DIR, MESH_INFO_FILE), 'w') as file:
                    file.writelines(mesh_info)
                with open(join(RESULT_DIR, STATUS_FILE), 'w') as file:
                    file.write('1\n')
                    file.write(str(person_count) + '\n')
                print('Batch Over')
                vv += 1

if __name__ == '__main__':
    main()