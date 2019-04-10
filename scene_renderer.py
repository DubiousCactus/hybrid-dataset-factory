#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Theo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the GPLv3 license.

"""
SceneRenderer

Generates an image by projecting a 3D mesh over a 2D transparent background.
"""


import numpy as np
import moderngl
import random
import yaml
import os

from pyrr import Matrix33, Matrix44, Quaternion, Vector3, Vector4, vector
from moderngl.ext.obj import Obj
from PIL import Image


class SceneRenderer:
    def __init__(self, meshes_dir: str, width: int, height: int,
                 world_boundaries, camera_parameters, render_perspective=False,
                 seed=None, oos_percentage=0.05):
        if seed:
            random.seed(seed)
        else:
            random.seed()
        self.render_perspective = render_perspective
        self.meshes, self.textures = self.load_meshes_and_textures(meshes_dir)
        self.width = width
        self.height = height
        self.boundaries = self.compute_boundaries(world_boundaries)
        self.out_of_screen_margin = oos_percentage
        with open(camera_parameters, 'r') as cam_file:
            try:
                self.camera_parameters = yaml.safe_load(cam_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)
        self.setup_opengl()

    def load_meshes_and_textures(self, path):
        meshes, textures = {}, []
        mesh_attributes = {}

        try:
            with open(os.path.join(path, "config.yaml"), "r") as config:
                try:
                    mesh_attributes = yaml.safe_load(config)
                except yaml.YAMLError as exc:
                    raise Exception(exc)
        except EnvironmentError as e:
            print("[!] Could not load mesh attributes file:\
                  {}".format(os.path.join(path, "config.yaml")))
            raise EnvironmentError(e)

        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                if file.endswith('.obj'):
                    file_name = os.path.split(file)[-1]
                    if file_name not in mesh_attributes:
                        print("[!] Could not find mesh attributes for\
                              {}".format(file_name))
                        continue
                    try:
                        obj_file = Obj.open(os.path.join(path, file_name))
                    except Exception as e:
                        raise Exception(e)

                    meshes[file_name] = {
                        'obj': obj_file,
                        'center': Vector3(mesh_attributes[file_name])
                    }
                elif file.endswith('.png') or file.endswith('.jpg'):
                    textures.append(os.path.join(path, file))
        return meshes, textures

    def compute_boundaries(self, world_boundaries):
        return  {
            'x': world_boundaries['x'] / 2,
            'y': world_boundaries['y'] / 2
        }

    def set_drone_pose(self, drone_pose):
        self.drone_pose = drone_pose
        self.gate_poses = []

    def setup_opengl(self):
        self.context = moderngl.create_standalone_context()
        camera_intrinsics = [ self.camera_parameters['camera_matrix']['data'][0:3],
            self.camera_parameters['camera_matrix']['data'][3:6],
            self.camera_parameters['camera_matrix']['data'][6::]
        ]
        fx, fy = camera_intrinsics[0][0], camera_intrinsics[1][1]
        cx, cy = camera_intrinsics[0][2], camera_intrinsics[1][2]
        x0, y0 = 0, 0 # Camera image origin
        zfar, znear = 100.0, 0.1 # distances to the clipping plane
        # Works by following: https://blog.noctua-software.com/opencv-opengl-projection-matrix.html
        # Doesn't work by following: http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
        self.projection = Matrix44([
            [fx/cx, 0, 0, 0],
            [0, fy/cy, 0, 0],
            [0, 0, (-zfar - znear)/(zfar - znear), -1],
            [0, 0, (-2.0*zfar*znear)/(zfar - znear), 0] # TODO: EXPLAIN WHY IT WORKS WHEN I FLIP [2][2] AND [3][2]
        ])

    def destroy(self):
        self.context.release()

    def render_gate(self, view, min_dist):
        ''' Randomly move the gate around, while keeping it inside the boundaries '''
        gate_translation = None
        too_close = True
        # Prevent gates from spawning too close to each other
        while too_close:
            too_close = False
            gate_translation = Vector3([
                random.uniform(-self.boundaries['x'], self.boundaries['x']),
                random.uniform(-self.boundaries['y'], self.boundaries['y']),
                0
            ])
            for gate_pose in self.gate_poses:
                if np.linalg.norm(gate_pose - gate_translation) <= float(min_dist):
                    too_close = True
                    break

        self.gate_poses.append(gate_translation)

        ''' Randomly rotate the gate horizontally, around the Z-axis '''
        gate_rotation = Quaternion.from_z_rotation(random.random() * np.pi)

        # Trace a line along the gate's x axis. If the camera is behind
        # that line, rotate the gate by Pi (this is done by simply computing
        # the cross product between the line crossing the gate in the x-axis,
        # and the camera position: the resulting vector's z component is either
        # negative or positive depending on the camera's position relative to
        # that line).
        leftmost_point = Matrix44.from_translation(gate_translation)\
            * Vector4.from_vector3((Matrix33(gate_rotation)\
                                    * Vector3([-20, 0, 0])), w=1)
        rightmost_point = Matrix44.from_translation(gate_translation)\
            * Vector4.from_vector3((Matrix33(gate_rotation)\
                                    * Vector3([20, 0, 0])), w=1)
        leftmost_point = Vector3(leftmost_point.xyz)
        rightmost_point = Vector3(rightmost_point.xyz)
        cross_product = (rightmost_point -
                         leftmost_point).cross(self.drone_pose.translation -
                                               leftmost_point)
        if cross_product.z < 0:
            gate_rotation = Quaternion.from_z_rotation(np.pi) * gate_rotation

        model = Matrix44.from_translation(gate_translation) * gate_rotation
        # With respect to the camera, for the annotation
        gate_orientation = Matrix33(self.drone_pose.orientation) * Matrix33(gate_rotation)
        gate_orientation = Quaternion.from_matrix(gate_orientation)
        # Model View Projection matrix
        mvp = self.projection * view * model

        # Shader program
        vertex_shader_source = open('data/shader.vert').read()
        fragment_shader_source = open('data/shader.frag').read()
        prog = self.context.program(vertex_shader=vertex_shader_source,
                                         fragment_shader=fragment_shader_source)
        prog['Light'].value = (0.0, 0.0, 4.0) # TODO
        prog['Color'].value = (1.0, 1.0, 1.0, 0.25) # TODO
        prog['Mvp'].write(mvp.astype('f4').tobytes())

        # Vertex Buffer and Vertex Array
        mesh = self.meshes[random.choice(list(self.meshes.keys()))]
        vbo = self.context.buffer(mesh['obj'].pack())
        vao = self.context.simple_vertex_array(prog, vbo, *['in_vert', 'in_text', 'in_norm'])

        return mesh, vao, model, gate_translation, gate_orientation

    '''
        Converting the gate normal's world coordinates to image coordinates
    '''
    # TODO: Refactor the next two functions
    def compute_gate_normal(self, mesh, view, model):
        gate_normal = model * (mesh['center'] + Vector3([0, 0.5, 0]))
        clip_space_gate_normal = self.projection * (view *
                                                    Vector4.from_vector3(gate_normal,
                                                                         w=1.0))
        if clip_space_gate_normal.w != 0:
            normalized_device_coordinate_space_gate_normal\
                = Vector3(clip_space_gate_normal.xyz) / clip_space_gate_normal.w
        else: # Clipped
            normalized_device_coordinate_space_gate_normal = clip_space_gate_normal.xyz

        viewOffset = 0
        image_frame_gate_normal =\
            ((np.array(normalized_device_coordinate_space_gate_normal.xy) + 1.0) /
             2.0) * np.array([self.width, self.height]) + viewOffset

        # Translate from bottom-left to top-left
        image_frame_gate_normal[1] = self.height - image_frame_gate_normal[1]

        return image_frame_gate_normal

    '''
        Converting the gate center's world coordinates to image coordinates
    '''
    def compute_gate_center(self, mesh, view, model, gate_dist):
        # Return if the camera is within 50cm of the gate, because it's not
        # visible
        mesh_center = model * mesh['center']
        if np.linalg.norm(mesh_center - self.drone_pose.translation) <= 0.3:
            return [-1, -1]

        clip_space_gate_center = self.projection * (view *
                                                    Vector4.from_vector3(mesh_center,
                                                                         w=1.0))
        if clip_space_gate_center.w != 0:
            normalized_device_coordinate_space_gate_center\
                = Vector3(clip_space_gate_center.xyz) / clip_space_gate_center.w
        else: # Clipped
            normalized_device_coordinate_space_gate_center = clip_space_gate_center.xyz

        # Behind the camera
        if normalized_device_coordinate_space_gate_center.z >= 1:
            return [-1, -1]

        viewOffset = 0
        image_frame_gate_center =\
            ((np.array(normalized_device_coordinate_space_gate_center.xy) + 1.0) /
             2.0) * np.array([self.width, self.height]) + viewOffset

        # Translate from bottom-left to top-left
        image_frame_gate_center[1] = self.height - image_frame_gate_center[1]

        # Move the gate center back to the image frame if it's slightly
        # outside of the image frame (the gate frame is still visible and we can
        # guess where to steer)
        for i in range(2):
            if image_frame_gate_center[i] <= 0:
                offset = (-image_frame_gate_center[i]) / (self.width if i == 0
                                                          else self.height)
                # TODO: Make this dirty hack cleaner
                if gate_dist < 3 and  offset <= 0.3:
                    image_frame_gate_center[i] = 1
                elif gate_dist <= 7 and offset <= self.out_of_screen_margin:
                    image_frame_gate_center[i] = 1

        return image_frame_gate_center

    '''
        Project the perspective as a grid (might need some tuning for
        non-square environments)
    '''
    def render_perspective_grid(self, view):
        vertex_shader_source = open('data/shader.vert').read()
        fragment_shader_source = open('data/shader.frag').read()
        grid_prog = self.context.program(vertex_shader=vertex_shader_source,
                             fragment_shader=fragment_shader_source)

        grid = []
        x_length = int(self.boundaries['x'])
        for i in range(x_length * 2 + 1):
            grid.append([i - x_length, -x_length, 0.0,
                         i - x_length, x_length, 0.0])
            grid.append([-x_length, i - x_length, 0.0,
                         x_length, i - x_length, 0.0])

        grid = np.array(grid)

        vp = self.projection * view
        grid_prog['Light'].value = (0.0, 0.0, 6.0) # TODO
        grid_prog['Color'].value = (1.0, 1.0, 1.0, 0.25) # TODO
        grid_prog['Mvp'].write(vp.astype('f4').tobytes())

        vbo = self.context.buffer(grid.astype('f4').tobytes())
        vao = self.context.simple_vertex_array(grid_prog, vbo, 'in_vert')

        return vao

    '''
        Returns the Euclidean distance of the gate to the camera
    '''
    def compute_camera_proximity(self, mesh, view, model):
        dist = np.linalg.norm((model * mesh['center']) - self.drone_pose.translation)
        coords = self.compute_gate_center(mesh, view, model, dist)
        if coords[0] < 0 or coords[0] > self.width or coords[1] < 0 or coords[1] > self.height:
            return coords, 1000
        else:
            return coords, dist

    def generate(self, min_dist=2.0, max_gates=6):
        # Camera view matrix
        view = Matrix44.look_at(
            # eye: position of the camera in world coordinates
            self.drone_pose.translation,
            # target: position in world coordinates that the camera is looking at
            self.drone_pose.translation + (self.drone_pose.orientation *
                                           Vector3([1.0, 0.0, 0.0])),
            # up: up vector of the camera.
            self.drone_pose.orientation * Vector3([0.0, 0.0, 1.0])
        )

        # Framebuffers
        # Use 4 samples for MSAA anti-aliasing
        msaa_render_buffer = self.context.renderbuffer((self.width,
                                                        self.height),
                                                       samples=8)
        msaa_depth_render_buffer = self.context.depth_renderbuffer((self.width,
                                                                    self.height),
                                                                   samples=8)
        fbo1 = self.context.framebuffer(
            msaa_render_buffer,
            depth_attachment=msaa_depth_render_buffer)

        # Downsample to the final framebuffer
        render_buffer = self.context.renderbuffer((self.width, self.height))
        depth_render_buffer = self.context.depth_renderbuffer((self.width, self.height))
        fbo2 = self.context.framebuffer(render_buffer, depth_render_buffer)

        # Texturing
        texture_image = Image.open(random.choice(self.textures))
        texture = self.context.texture(texture_image.size, 3, texture_image.tobytes())
        texture.build_mipmaps()
        texture.use()


        # Rendering
        fbo1.use()
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.clear(0, 0, 0, 0)

        gate_center = None
        gate_rotation = None
        gate_normal = None
        gate_distance = None
        min_prox = None
        # Render at least one gate
        for i in range(random.randint(1, max_gates)):
            mesh, vao, model, translation, rotation = self.render_gate(view, min_dist)
            center, proximity = self.compute_camera_proximity(mesh, view, model)
            # Pick the target gate: the closest to the camera
            if min_prox is None or proximity < min_prox:
                min_prox = proximity
                gate_center = center
                gate_rotation = rotation
                gate_distance = np.linalg.norm(self.drone_pose.translation -
                                               translation)
                gate_normal = self.compute_gate_normal(mesh, view, model)
            # Rendering
            vao.render()
            vao.release()

        if self.render_perspective:
            vao_grid = self.render_perspective_grid(view)
            vao_grid.render(moderngl.LINES, 65 * 4)
            vao_grid.release()

        self.context.copy_framebuffer(fbo2, fbo1)

        # Loading the image using Pillow
        img = Image.frombytes('RGBA', fbo2.size, fbo2.read(components=4,
                                                         alignment=1), 'raw', 'RGBA', 0, -1)

        annotations = {
            'gate_center_img_frame': gate_center,
            'gate_rotation': gate_rotation,
            'gate_distance': gate_distance,
            'gate_normal': gate_normal,
            'drone_pose': self.drone_pose.translation,
            'drone_orientation':self.drone_pose.orientation
        }

        '''
        A soon-to-be-fixed bug in ModernGL forces me to release the render
        buffers manually
        '''
        msaa_render_buffer.release()
        msaa_depth_render_buffer.release()
        render_buffer.release()
        depth_render_buffer.release()
        fbo1.release()
        fbo2.release()
        texture.release()

        return (img, annotations)

