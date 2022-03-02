import sys
import os
path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(path))
sys.path.append(os.path.dirname(os.path.dirname(path)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(path))))
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import glob
from scipy.io import loadmat,savemat

from preprocess_img import Preprocess, get_5_face_landmarks
from load_data import *
from reconstruct_mesh import Reconstruction 
from reconstruct_mesh import *

import mesh
import deformation

def read_file_from_dir(root_dir):
    filelist = []
    for _root, _dirs, _files in os.walk(root_dir):
        for _file in _files:
            filepath = os.path.join(_root, _file)
            filelist.append(filepath)
    filelist = sorted(filelist)
    return filelist


class Deep3DFaceReconstruction():
    def __init__(self, model_root='/mnt/mfs/chenjianyi/project/Deepfake/Deep3DFaceReconstruction'):
        # read BFM face model
        # transfer original BFM model to our model
        if not os.path.isfile(os.path.join(model_root, 'BFM/BFM_model_front.mat')):
            transferBFM09(model_root)
    
        # read BFM model
        self.facemodel = BFM(model_root)
    
        # read standard landmarks for preprocessing images
        self.lm3D = load_lm3d(model_root)
    
        self.images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
        self.graph_def = self.load_graph(os.path.join(model_root, 'network/FaceReconModel.pb'))

    def load_graph(self, graph_filename):
        with tf.gfile.GFile(graph_filename,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def predict(self, img_path):
        params = {}
        img = Image.open(img_path)
        lm = get_5_face_landmarks(np.array(img))
        kpt_idx_3d = self.facemodel.keypoints
        params['lm'] = lm
        params['kpt_idx_3d'] = kpt_idx_3d
        input_img,lm_new,transform_params = Preprocess(img, lm, self.lm3D)
        params['transform_params'] = transform_params
        params['input_img'] = input_img
        with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
            images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
            tf.import_graph_def(self.graph_def, name='resnet', input_map={'input_imgs:0': images})
            coeff = graph.get_tensor_by_name('resnet/coeff:0')
            with tf.Session() as sess:
                coef = sess.run(coeff, feed_dict={images: input_img})
                id_coeff,ex_coeff,tex_coeff,angles,gamma,translation = Split_coeff(coef)
                rotation = Compute_rotation_matrix(angles)
                params['rotation'] = rotation
                params['translation'] = translation
                params['coef'] = coef
                # reconstruct 3D face with output coefficients and face model
                face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(coef, self.facemodel)
                Vertices = np.squeeze(face_shape, (0))
                Vertices = Render_prolection(Vertices.copy(), params, shift_to_origin=False)
                colors = mesh.transform.get_colors_from_image(input_img[0], Vertices.T)
                params['colors'] = colors
                # reshape outputs
                input_img = np.squeeze(input_img)
                shape = np.squeeze(face_shape, (0))
                color = np.squeeze(face_color, (0))
                landmarks_2d = np.squeeze(landmarks_2d, (0))
                print(img_path, shape.shape, input_img.shape, color.shape, face_projection.shape)
                #cv2.imwrite('out/%s' % img_path.split('/')[-1], input_img)
        return shape, color, params


    def Reenactment(self, targets, sources, targetA, sourceA, out_dir, h=480, w=640):
        # input & output
        sources_list = read_file_from_dir(sources)
        targets_list = read_file_from_dir(targets)
        n = min(len(sources_list), len(targets_list))
        nVertices = int(self.facemodel.meanshape.shape[1] / 3)
        SBs = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')
        TBs = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')
        source_name = sources_list[0].split('/')[-2]
        target_name = targets_list[0].split('/')[-2]
        ST_dir = os.path.join(out_dir, '%s_%s' % (source_name, target_name))
        TS_dir = os.path.join(out_dir, '%s_%s' % (target_name, source_name))
        if not os.path.exists(ST_dir):
            os.makedirs(ST_dir)
        if not os.path.exists(TS_dir):
            os.makedirs(TS_dir)
    
        # predict
        BG_Ss = []
        BG_Ts = []
        SB_colors = []
        TB_colors = []
        SB_params = []
        TB_params = []
        #n = 10
        for i in range(n):
            source = sources_list[i]
            target = targets_list[i]
            BG_S = cv2.imread(source)
            BG_T = cv2.imread(target)
            BG_Ss.append(BG_S)
            BG_Ts.append(BG_T)
            Svertices, Scolors, Sparams = self.predict(source)
            Tvertices, Tcolors, Tparams = self.predict(target)
            # vertices
            SBs[i] = Svertices
            TBs[i] = Tvertices
            # colors
            SB_colors.append(Scolors)
            TB_colors.append(Tcolors)
            # params
            SB_params.append(Sparams)
            TB_params.append(Tparams)
            
        SAvertices, _, _ = self.predict(sourceA)
        TAvertices, _, _  = self.predict(targetA)
        SAvertices = SAvertices.astype(dtype=np.float32, order='C')
        TAvertices = TAvertices.astype(dtype=np.float32, order='C')
        SBs = SBs.astype(np.float32)
        TBs = TBs.astype(np.float32)
    
        STresults = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')
        TSresults = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')
    
        # swap
        triangles = self.facemodel.tri - 1 # - 1 start from 0
        triangles = triangles.astype(dtype=np.int32, order='C')
        nTriangles = triangles.shape[0]
        deformation.deformation.deformation_core(SAvertices, TAvertices, triangles, \
            nVertices, nTriangles, SBs, n, TSresults)
        for i in range(n):
            # deformation transfer results
            BG_T = BG_Ts[i]
            BG_S = BG_Ss[i]
            Tcolors = TB_colors[i]
            Tparams = TB_params[i]
            Vertices = TSresults[i]  # (N, 3)
            Vertices = Render_prolection(Vertices, Tparams)
            TSimage1_1 = mesh.render.render_colors(Vertices, triangles, Tcolors[:, ::-1], h, w, BG=BG_T)
            TSimage1_2 = mesh.render.render_colors(Vertices, triangles, Tparams['colors'], h, w, BG=BG_T)
            TSpath = os.path.join(TS_dir, "%04d.jpg" % i)
            # save obj
            #path = TSpath.replace('jpg', 'obj')
            #mesh.interact.write_obj_with_colors(path, Vertices, triangles, Tcolors)

            # directly exchage expression coeff results
            BG_T = BG_Ts[i]
            Tcolors = TB_colors[i]
            Sparams = SB_params[i]
            Tparams = TB_params[i]
            TS_coef = Tparams['coef'].copy()
            TS_coef[:, 80: 144] = Sparams['coef'][:, 80: 144]
            face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(TS_coef, self.facemodel)
            Vertices = np.squeeze(face_shape, (0))
            Vertices = Render_prolection(Vertices.copy(), Tparams)
            TSimage2_1 = mesh.render.render_colors(Vertices, triangles, Tcolors[:, ::-1], h, w, BG=BG_T)
            TSimage2_2 = mesh.render.render_colors(Vertices, triangles, Tparams['colors'], h, w, BG=BG_T)
            # save obj
            #path = TSpath.replace('jpg', 'obj')
            #mesh.interact.write_obj_with_colors(path, Vertices, triangles, Tcolors)

            # save image
            BGimage = np.concatenate([BG_T, BG_S], 1)
            TSimage1 = np.concatenate([TSimage2_1, TSimage1_1], 1)
            TSimage2 = np.concatenate([TSimage2_2, TSimage1_2], 1)
            image = np.concatenate([BGimage, TSimage1, TSimage2], 0)
            cv2.imwrite(TSpath, image.astype('uint8'))

    def Swap(self, targets, sources, targetA, sourceA, out_dir, h=480, w=640):
        # input & output
        sources_list = read_file_from_dir(sources)
        targets_list = read_file_from_dir(targets)
        n = min(len(sources_list), len(targets_list))
        nVertices = int(self.facemodel.meanshape.shape[1] / 3)
        SBs = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')
        TBs = np.zeros((n, nVertices, 3), dtype=np.float32, order='C')

        # predict
        BG_Ss = []
        BG_Ts = []
        SB_colors = []
        TB_colors = []
        SB_params = []
        TB_params = []
        #n = 10
        for i in range(n):
            source = sources_list[i]
            target = targets_list[i]
            BG_S = cv2.imread(source)
            BG_T = cv2.imread(target)
            BG_Ss.append(BG_S)
            BG_Ts.append(BG_T)
            Svertices, Scolors, Sparams = self.predict(source)
            Tvertices, Tcolors, Tparams = self.predict(target)
            # vertices
            SBs[i] = Svertices
            TBs[i] = Tvertices
            # colors
            SB_colors.append(Scolors)
            TB_colors.append(Tcolors)
            # params
            SB_params.append(Sparams)
            TB_params.append(Tparams)
        SAvertices, SAcolors, SAparams = self.predict(sourceA)
        TAvertices, TAcolors, TAparams  = self.predict(targetA)
        triangles = self.facemodel.tri - 1 # - 1 start from 0
        triangles = triangles.astype(dtype=np.int32, order='C')

        # swap
        for i in range(n):
            Sparams = SB_params[i]
            Tparams = TB_params[i]
            Scolors = SB_colors[i]
            Tcolors = TB_colors[i]
            BG_S = BG_Ss[i]
            BG_T = BG_Ts[i]
            STcoef = Sparams['coef'].copy()
            TScoef = Tparams['coef'].copy()
            STcoef[:, :80] = TAparams['coef'][:, :80]  # swap identity
            #STcoef[:, 227:254] = TAparams['coef'][:, 227:254]  # swap lighting coeff
            TScoef[:, :80] = SAparams['coef'][:, :80]  # swap identity
            #TScoef[:, 227:254] = SAparams['coef'][:, 227:254]  # swap lighting coeff

            face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(STcoef, self.facemodel)
            Vertices = np.squeeze(face_shape, (0))
            Vertices = Render_prolection(Vertices.copy(), Tparams)
            STimage1 = mesh.render.render_colors(Vertices, triangles, Scolors[:, ::-1], h, w, BG=BG_T)  # swap face, swap texture(from synthetise)
            STimage2 = mesh.render.render_colors(Vertices, triangles, Sparams['colors'], h, w, BG=BG_T) # swap face, swap texture(from image)
            #STimage3 = mesh.render.render_colors(Vertices, triangles, Tcolors[:, ::-1], h, w, BG=BG_T)  # swap face, not swap texture
            #STimage4 = mesh.render.render_colors(Vertices, triangles, Tparams['colors'], h, w, BG=BG_T) # swap face, not swap texture

            face_shape,face_texture,face_color,tri,face_projection,z_buffer,landmarks_2d = Reconstruction(TScoef, self.facemodel)
            Vertices = np.squeeze(face_shape, (0))
            Vertices = Render_prolection(Vertices.copy(), Sparams)
            TSimage1 = mesh.render.render_colors(Vertices, triangles, Tcolors[:, ::-1], h, w, BG=BG_S) # swap face, swap texture(from synthetise)
            TSimage2 = mesh.render.render_colors(Vertices, triangles, Tparams['colors'], h, w, BG=BG_S) # swap face, swap texture(from image)
            #TSimage3 = mesh.render.render_colors(Vertices, triangles, Scolors[:, ::-1], h, w, BG=BG_S) # swap face, not swap texture
            #TSimage4 = mesh.render.render_colors(Vertices, triangles, Sparams['colors'], h, w, BG=BG_S) # swap face, not swap texture

            # save image
            black_BG = np.zeros((h, w, 3), dtype=np.uint8)
            path = os.path.join(out_dir, "%04d.jpg" % i)
            BGimage = np.concatenate([BG_T, BG_S], 1)
            STimage = np.concatenate([STimage1, STimage2], 1)
            TSimage = np.concatenate([TSimage1, TSimage2], 1)
            image = np.concatenate([BGimage, STimage, TSimage], 0)
            cv2.imwrite(path, image.astype('uint8'))
 
if __name__ == '__main__':
    sourceA = "/mnt/mfs/chenjianyi/records/Deepfake/face2face/frames/000/000_0001.jpg"
    targetA = "/mnt/mfs/chenjianyi/records/Deepfake/face2face/frames/003/003_0091.jpg"
    sources = "/mnt/mfs/chenjianyi/records/Deepfake/face2face/frames/000/"
    targets = "/mnt/mfs/chenjianyi/records/Deepfake/face2face/frames/003/"
    out_dir = 'out'
    demo = Deep3DFaceReconstruction()
    #demo.Reenactment(targets, sources, targetA, sourceA, out_dir)
    demo.Reenactment(sources, targets, sourceA, targetA, out_dir)

    out_dir = 'swap'
    #demo.Swap(sources, targets, sourceA, targetA, out_dir)
    #demo.Swap(targets, sources, targetA, sourceA, out_dir)
