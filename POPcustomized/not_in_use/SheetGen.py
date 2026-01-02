import smplx
import numpy as np
import torch
from os import listdir
from os.path import join, dirname, realpath, isfile, isdir
from os import path as osp
import open3d as o3d
import random
from math import sin, cos, pi
import json


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


def get_feet():
    ids_m = json.load(open('/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/smplx_vert_segmentation.json', 'r'))
    feet = ids_m['leftFoot'] + ids_m['rightFoot'] + ids_m['leftToeBase'] + ids_m['rightToeBase']
    return feet


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

    def carve(self, sparsity, write=True):
        assert 0.05 <= sparsity <= 0.5, 'sparsity must in [0.05, 0.5]'
        verts = self.vertices - self.pvs
        dots = np.dot(verts, self.normal)
        val_ids = np.where(dots > 0)
        st = o3d.geometry.PointCloud()
        sheet_pos_v = self.vertices[val_ids] + random.uniform(0.1, 0.5) * self.normal
        sheet_neg_v = self.vertices[val_ids]
        sheet_pos_n = np.ones_like(sheet_pos_v) * self.normal
        sheet_neg_n = - np.ones_like(sheet_neg_v) * self.normal
        sheet_v = np.concatenate((sheet_pos_v, sheet_neg_v), axis=0)
        sheet_n = np.concatenate((sheet_pos_n, sheet_neg_n), axis=0)
        st.points = o3d.utility.Vector3dVector(sheet_pos_v)
        st.normals = o3d.utility.Vector3dVector(sheet_pos_n)
        # st.estimate_normals()
        # st.orient_normals_consistent_tangent_plane(100)
        st = st.voxel_down_sample(voxel_size=sparsity)  # sparsify the points
        st.remove_radius_outlier(nb_points=200, radius=5*sparsity)
        temp = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(st, depth=9, linear_fit=False)[0]
        temp = temp.filter_smooth_simple(number_of_iterations=2)
        print(np.asarray(temp.vertices).shape)
        if write:
            o3d.io.write_triangle_mesh('/dockerspace/BigBench/weirdshit/smplx_1102_t.obj', temp)
        return temp


def main():
    SMPLX_POP_TOOLKIT_DIR = '/dockerspace/Workspace/POPcustomized/POP-main/SMPLX_POP_ToolKit/'
    facets_m = np.load('/dockerspace/Workspace/POPcustomized/POP-main/assets/smplx_faces.npy')
    print(facets_m.shape)
    model_folder = '/dockerspace/BigBench/Downloads/AMASS/models_smplx_v1_1/models'
    model_type = 'smplx'
    gender = 'male'
    USE_FACE_CONTOUR = True
    num_betas = 10
    ext = 'npz'
    BETAS_MEAN = torch.zeros([1, 10], dtype=torch.float32)
    bob = smplx.create(model_folder, model_type=model_type,
                           gender=gender, use_face_contour=USE_FACE_CONTOUR,
                           num_betas=num_betas, ext=ext, num_expression_coeffs=10)
    pose = np.zeros((21, 3), dtype=float)
    output = bob(betas=BETAS_MEAN, body_pose=torch.tensor(pose, dtype=torch.float32).unsqueeze(0))
    joints = output['joints'].detach().cpu().numpy().squeeze(0)
    joints = joints - joints[0]
    spine2 = joints[6, :]
    l_collar = joints[13, :]
    l_elbow = joints[18, :]
    l_knee = joints[4, :]
    r_elbow = joints[19, :]
    l_ankle = joints[7, :]
    neck = joints[12, :]
    bound_min = np.asarray([r_elbow[0], random.uniform(l_knee[1], l_ankle[1]), -0.5], dtype=float)  # reference for mesh cropping
    bound_max = np.asarray([l_elbow[0], random.uniform(spine2[1], neck[1]), 0.5], dtype=float)
    global_frame_T = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    pose = pose_from_AMASS_CMU()
    output = bob(betas=BETAS_MEAN, body_pose=torch.tensor(pose, dtype=torch.float32).unsqueeze(0))
    joints = output['joints'].detach().cpu().numpy().squeeze(0)
    vertices = output['vertices'].detach().cpu().numpy().squeeze(0) - joints[0, :]
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
    b2g = np.linalg.solve(body_frame_T, global_frame_T).T  # rotate the vertices to make it align with the T-pose directions
    g2b = np.linalg.solve(global_frame_T, body_frame_T).T
    v_g = np.matmul(b2g, v_b.T).T  # rotated vertices
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=bound_min, max_bound=bound_max)
    mesh.vertices = o3d.utility.Vector3dVector(v_g)
    mesh = mesh.crop(box)
    v_g = np.asarray(mesh.vertices)   # note v_g has fewer points than v_b does
    triangles = np.asarray(mesh.triangles).flatten('C')
    sheet_frame_T = duvet.get_frame_sheet_T()
    # proj_along_sheet_x = np.dot(v_b, sheet_frame_T[0, :])
    # proj_along_sheet_y = np.dot(v_b, sheet_frame_T[1, :])
    # print(proj_along_sheet_x.shape)
    # # #
    # uv = v_g[:, :2] - np.min(v_g[:, :2], axis=0)
    # ratio = 1 / np.max(uv[:, :2], axis=0)
    # uv = uv * ratio
    # # #
    # b2s = np.linalg.solve(body_frame_T, sheet_frame_T).T
    # s2g = np.linalg.solve(sheet_frame_T, global_frame_T).T
    # v_s = np.matmul(s2g, v_b.T).T
    # uv = v_s[:, :2] - np.min(v_s[:, :2], axis=0)
    # ratio = 1 / np.max(uv[:, :2], axis=0)
    # uv = uv * ratio

    # ratio_x = 1 / (np.max(proj_along_sheet_x) - np.min(proj_along_sheet_x))
    # ratio_y = 1 / (np.max(proj_along_sheet_y) - np.min(proj_along_sheet_y))
    # proj_along_sheet_x = (proj_along_sheet_x - np.min(proj_along_sheet_x)) * ratio_x
    # proj_along_sheet_y = (proj_along_sheet_y - np.min(proj_along_sheet_y)) * ratio_y
    # uv = np.array([proj_along_sheet_x, proj_along_sheet_y]).T

    v_b = np.matmul(g2b, v_g.T).T
    s2g = np.linalg.solve(sheet_frame_T, global_frame_T).T
    v_s = np.matmul(s2g, v_b.T).T
    uv = v_s[:, :2] - np.min(v_s[:, :2], axis=0)
    ratio = 1 / np.max(uv[:, :2], axis=0)
    uv = uv * ratio
    triangle_uvs = uv[triangles]
    mesh.vertices = o3d.utility.Vector3dVector(v_b)
    mesh.triangle_uvs = o3d.utility.Vector2dVector(triangle_uvs)
    o3d.io.write_triangle_mesh('/dockerspace/BigBench/weirdshit/smplx_1102.obj', mesh)



    with open('/dockerspace/BigBench/weirdshit/smplx_1102_body.ply', 'w') as file:
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n' % vertices.shape[0])
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        # file.write('property float nx\n')
        # file.write('property float ny\n')
        # file.write('property float nz\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('property uchar alpha\n')
        file.write('element face %d\n' % facets_m.shape[0])
        file.write('property list uchar int vertex_indices\n')
        file.write('end_header\n')
        for v in (vertices + pelvis):
            file.write(
                '%f %f %f %d %d %d %d\n' % (
                    v[0], v[1], v[2], 255, 255, 255, 255))
        for f in facets_m:
            file.write(
                '3 %d %d %d\n' % (
                    f[0], f[1], f[2]))

if __name__ == '__main__':
    main()