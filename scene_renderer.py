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
import copy
import yaml
import os

from pyrr import Matrix33, Matrix44, Quaternion, Vector3, Vector4
from ModernGL.ext.obj import Obj
from math import atan2, cos, sin
from PIL import Image


class SceneRenderer:
    def __init__(self, meshes_dir: str, width: int, height: int,
                 world_boundaries, camera_parameters, render_perspective=False,
                 seed=None):
        if seed:
            random.seed(seed)
        else:
            random.seed()
        self.render_perspective = render_perspective
        self.width = width
        self.height = height
        self.boundaries = self.compute_boundaries(world_boundaries)
        with open(camera_parameters, 'r') as cam_file:
            try:
                self.camera_parameters = yaml.safe_load(cam_file)
            except yaml.YAMLError as exc:
                raise Exception(exc)
        self.setup_opengl()
        self.meshes = self.load_meshes_and_textures(meshes_dir)

    def load_meshes_and_textures(self, path):
        meshes = {}
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
                if file.endswith('_frame.obj'):
                    file_name = os.path.split(file)[-1]
                    try:
                        obj_file = Obj.open(os.path.join(path, file_name))
                        contour_obj_front_file = Obj.open(
                            os.path.join(path,
                                         mesh_attributes[file_name]['contour_front']))
                        contour_obj_back_file = Obj.open(
                            os.path.join(path,
                                         mesh_attributes[file_name]['contour_back']))
                        contour_png = Image.open(
                            os.path.join(
                                path, mesh_attributes[file_name]['texture'])
                        ).transpose(Image.FLIP_LEFT_RIGHT).transpose(
                            Image.FLIP_TOP_BOTTOM).convert('RGB')
                        contour_texture = self.context.texture(
                            contour_png.size, 3, contour_png.tobytes())
                        contour_texture.build_mipmaps()
                    except Exception as e:
                        raise Exception(e)

                    meshes[file_name] = {
                        'center': Vector3(mesh_attributes[file_name]['center']),
                        'width': mesh_attributes[file_name]['width'],
                        'height': mesh_attributes[file_name]['height'],
                        'contour_obj_front': contour_obj_front_file,
                        'contour_obj_back': contour_obj_back_file,
                        'contour_texture': contour_texture,
                        'obj': obj_file
                    }

        if len(meshes.items()) is 0:
            raise Exception("Meshes not loaded!")

        return meshes

    def compute_boundaries(self, world_boundaries):
        return {
            'x': world_boundaries['x'] / 2,
            'y': world_boundaries['y'] / 2
        }

    def set_drone_pose(self, drone_pose):
        self.drone_pose = drone_pose
        self.gate_poses = []

    def setup_opengl(self):
        self.context = moderngl.create_standalone_context()
        camera_intrinsics = [
            self.camera_parameters['camera_matrix']['data'][0:3],
            self.camera_parameters['camera_matrix']['data'][3:6],
            self.camera_parameters['camera_matrix']['data'][6::]
        ]
        fx, fy = camera_intrinsics[0][0], camera_intrinsics[1][1]
        cx, cy = camera_intrinsics[0][2], camera_intrinsics[1][2]
        zfar, znear = 100.0, 0.5  # distances to the clipping plane
        self.projection = Matrix44([
            [fx/cx, 0, 0, 0],
            [0, fy/cy, 0, 0],
            [0, 0, (-zfar - znear)/(zfar - znear), -1],
            [0, 0, (-2.0*zfar*znear)/(zfar - znear), 0]
        ])

    def destroy(self):
        self.context.release()

    def render_gate(self, view, min_dist):
        '''
            Randomly move the gate around, while keeping it inside the
            boundaries
        '''
        gate_translation = None
        in_fov = False
        max_tries = 100
        mesh = self.meshes[random.choice(list(self.meshes.keys()))]

        while not in_fov:
            if max_tries == 0:
                return (None, None, None, None)

            gate_translation = Vector3([
                random.uniform(-self.boundaries['x'], self.boundaries['x']),
                random.uniform(-self.boundaries['y'], self.boundaries['y']),
                0
            ])
            gate_center = copy.deepcopy(gate_translation)
            gate_center.z = mesh['center'][2]
            gate_top = copy.deepcopy(gate_translation)
            gate_top.z = mesh['center'][2] + (mesh['height']/2)
            gate_left = copy.deepcopy(gate_translation)
            gate_left.z = mesh['center'][2]
            gate_left.x -= mesh['width']/2
            gate_right = copy.deepcopy(gate_translation)
            gate_right.z = mesh['center'][2]
            gate_right.x += mesh['width']/2
            points = [gate_center, gate_left, gate_right, gate_top]
            in_fov = True
            for p in points:
                clip_space_vector = self.projection * (view
                                           * Vector4.from_vector3(p, w=1.0))
                if clip_space_vector.w != 0:
                    nds_vector = Vector3(clip_space_vector.xyz) / clip_space_vector.w
                else:  # Clipped
                    nds_vector = clip_space_vector.xyz

                if nds_vector.z >= 1:
                    in_fov = False
                    continue

                if not (nds_vector.x >= -1 and nds_vector.x <= 1
                        and nds_vector.y >= -1 and nds_vector.y <= 1):
                    in_fov = False
                    break
            max_tries -= 1

        self.gate_poses.append(gate_translation)

        facing = False
        max_tries = 100
        while not facing:
            ''' Randomly rotate the gate horizontally, around the Z-axis '''
            if max_tries == 0:
                return (None, None, None, None)
            gate_rotation = Quaternion.from_z_rotation(random.random() * np.pi)
            model = Matrix44.from_translation(gate_translation) * gate_rotation
            leftmost_point = model * Vector3(
                [mesh['center'][0] - 20, mesh['center'][1], 0])
            rightmost_point = model * Vector3(
                [mesh['center'][0] + 20, mesh['center'][1], 0])
            cross_product = (rightmost_point -
                             leftmost_point).cross(self.drone_pose.translation -
                                                   leftmost_point)
            facing = True if cross_product.z >= 0 else False
            # With respect to the camera, for the annotation
            camera_yaw = self.euler_yaw(self.drone_pose.orientation)
            gate_yaw = self.euler_yaw(model.quaternion)
            gate_orientation = camera_yaw - gate_yaw
            if gate_orientation < 0:
                gate_orientation += 180
            if gate_orientation > 160 or gate_orientation < 20:
                facing = False
            max_tries -= 1

        # Model View Projection matrix
        mvp = self.projection * view * model

        # Shader program
        vertex_shader_source = open('data/shader.vert').read()
        fragment_shader_source = open('data/shader.frag').read()
        prog = self.context.program(vertex_shader=vertex_shader_source,
                                    fragment_shader=fragment_shader_source)

        prog['Light1'].value = (
            random.uniform(-self.boundaries['x'], self.boundaries['x']),
            random.uniform(-self.boundaries['y'], self.boundaries['y']),
            random.uniform(5, 7))
        # prog['Light2'].value = (
            # random.uniform(-self.boundaries['x'], self.boundaries['x']),
            # random.uniform(-self.boundaries['y'], self.boundaries['y']),
            # random.uniform(4, 7))
        # prog['Light3'].value = (
            # random.uniform(-self.boundaries['x'], self.boundaries['x']),
            # random.uniform(-self.boundaries['y'], self.boundaries['y']),
                # random.uniform(4, 7))
            # prog['Light4'].value = (
            # random.uniform(-self.boundaries['x'], self.boundaries['x']),
            # random.uniform(-self.boundaries['y'], self.boundaries['y']),
            # random.uniform(4, 7))
        prog['MVP'].write(mvp.astype('f4').tobytes())

        frame_vbo = self.context.buffer(
            mesh['obj'].pack('vx vy vz nx ny nz tx ty'))
        frame_vao = self.context.simple_vertex_array(
            prog, frame_vbo, 'in_vert', 'in_norm', 'in_text')
        contour_front_vbo = self.context.buffer(
            mesh['contour_obj_front'].pack('vx vy vz nx ny nz tx ty'))
        contour_front_vao = self.context.simple_vertex_array(
            prog, contour_front_vbo, 'in_vert', 'in_norm', 'in_text')
        contour_back_vbo = self.context.buffer(
            mesh['contour_obj_back'].pack('vx vy vz nx ny nz tx ty'))
        contour_back_vao = self.context.simple_vertex_array(
            prog, contour_back_vbo, 'in_vert', 'in_norm', 'in_text')

        prog['viewPos'].value = (
            self.drone_pose.translation.x,
            self.drone_pose.translation.y,
            self.drone_pose.translation.z
        )
        prog['Color'].value = (random.uniform(0, 0.7),
                               random.uniform(0, 0.7),
                               random.uniform(0, 0.7))
        prog['UseTexture'].value = False
        frame_vao.render()
        prog['Color'].value = (0.8, 0.8, 0.8)
        contour_back_vao.render()
        mesh['contour_texture'].use()
        prog['UseTexture'].value = True
        contour_front_vao.render()

        return mesh, model, gate_translation, gate_orientation

    def project_to_img_frame(self, vector, viewMatrix):
        clip_space_vector = self.projection * (
            viewMatrix * Vector4.from_vector3(vector, w=1.0))
        if clip_space_vector.w != 0:
            nds_vector = Vector3(clip_space_vector.xyz) / clip_space_vector.w
        else:  # Clipped
            nds_vector = clip_space_vector.xyz

        if nds_vector.z >= 1:
            return [-1, -1]

        viewOffset = 0
        image_frame_vector =\
            ((np.array(nds_vector.xy) + 1.0) /
             2.0) * np.array([self.width, self.height]) + viewOffset

        # Translate from bottom-left to top-left
        image_frame_vector[1] = self.height - image_frame_vector[1]

        return image_frame_vector

    def euler_yaw(self, q):
        siny_cosp = 2.0 * ((q.w * q.z) + (q.x * q.y))
        cosy_cosp = 1.0 - (2.0 * ((q.y * q.y) + (q.z * q.z)))

        return atan2(siny_cosp, cosy_cosp) * (180.0/np.pi)

    '''
        Converting the gate normal's world coordinates to image coordinates
    '''
    def compute_gate_normal(self, mesh, view, model):
        gate_normal = model * (mesh['center'] + Vector3([0, 0.5, 0]))
        return self.project_to_img_frame(gate_normal, view)

    '''
        Converting the gate center's world coordinates to image coordinates
    '''
    def compute_gate_center(self, mesh, view, model):
        # Return if the camera is within 0.6cm of the gate, because it's not
        # visible
        mesh_center = model * mesh['center']
        if np.linalg.norm(mesh_center - self.drone_pose.translation) <= 0.6:
            return [-1, -1]

        return self.project_to_img_frame(mesh_center, view)


    '''
        Project the perspective as a grid (might need some tuning for
        non-square environments)
    '''
    def render_perspective_grid(self, view):
        vertex_shader_source = open('data/shader.vert').read()
        fragment_shader_source = open('data/shader.frag').read()
        grid_prog = self.context.program(
            vertex_shader=vertex_shader_source,
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
        grid_prog['Light1'].value = (0.0, 0.0, 3.0)
        # grid_prog['Light2'].value = (3.0, 0.0, 3.0)
        # grid_prog['Light3'].value = (-3.0, 0.0, 3.0)
        # grid_prog['Light4'].value = (0.0, 6.0, 3.0)
        grid_prog['UseTexture'].value = False;
        grid_prog['Color'].value = (0.0, 1.0, 0.0)
        grid_prog['MVP'].write(vp.astype('f4').tobytes())

        vbo = self.context.buffer(grid.astype('f4').tobytes())
        vao = self.context.simple_vertex_array(grid_prog, vbo, 'in_vert')

        vao.render(moderngl.LINES, 65 * 4)
        vao.release()

    '''
        Returns the Euclidean distance of the gate to the camera
    '''
    def compute_camera_proximity(self, model, mesh):
        return np.linalg.norm((model * mesh['center']) - self.drone_pose.translation)

    '''
        Computes the bounding box min/max coordinates (diagonal corners) in the
        image frame, and clips them to the image borders if one of them is
        outside.  Otherwise, the gate is not visible.
    '''
    def compute_bbox_coords(self, model, mesh, view):
        center = mesh['center']
        world_corners = {
            'top_left': model * Vector3([
                center[0] - mesh['width']/2,
                center[1],
                center[2] + mesh['height']/2
            ]),
            'top_right': model * Vector3([
                center[0] + mesh['width']/2,
                center[1],
                center[2] + mesh['height']/2
            ]),
            'bottom_right': model * Vector3([
                center[0] + mesh['width']/2,
                center[1],
                center[2] - mesh['height']/2
            ]),
            'bottom_left': model * Vector3([
                center[0] - mesh['width']/2,
                center[1],
                center[2] - mesh['height']/2
            ])
        }
        hidden_corners = 0
        left = right = top = bottom = None

        for key, value in world_corners.items():
            img_coords = self.project_to_img_frame(value, view)
            if (img_coords[0] < 10 or img_coords[0] > (self.width-10)
                    or img_coords[1] < 10 or img_coords[1] > (self.height-10)):
                hidden_corners += 1
            if left is None or (img_coords[0] < left['x']):
                left = {'x': img_coords[0], 'y': img_coords[1]}
            if top is None or (img_coords[1] < top['y']):
                top = {'x': img_coords[0], 'y': img_coords[1]}
            if bottom is None or (img_coords[1] > bottom['y']):
                bottom = {'x': img_coords[0], 'y': img_coords[1]}
            if right is None or (img_coords[0] > right['x']):
                right = {'x': img_coords[0], 'y': img_coords[1]}

        image_corners = {
            'min': [int(left['x']), int(top['y'])],
            'max': [int(right['x']), int(bottom['y'])]
        }

        if hidden_corners > 3:
            return {}
        elif hidden_corners > 0:
            for key, img_coords in image_corners.items():
                for i in range(0,2):
                    if img_coords[i] < 0:
                        img_coords[i] = 0
                    elif img_coords[i] > (self.width if i == 0 else self.height):
                        img_coords[i] = self.width if i == 0 else self.height
                image_corners[key] = img_coords

        return image_corners


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

        # Rendering
        fbo1.use()
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.clear(0, 0, 0, 0)

        bounding_boxes = []
        closest_gate, second_closest = None, None
        n = 0
        # Render one gate
        mesh, model, translation, rotation = self.render_gate(view, min_dist)
        if mesh == None:
            return (None, None)

        proximity = self.compute_camera_proximity(model, mesh)
        coords = self.compute_bbox_coords(model, mesh, view)

        if coords != {}:
            gate_distance = proximity
            gate_normal = self.compute_gate_normal(mesh, view, model)
            gate_center = self.compute_gate_center(mesh, view, model)

            bounding_boxes.append({
                'class_id': 1,
                'min': [coords['min'][0], coords['min'][1]],
                'max': [coords['max'][0], coords['max'][1]],
                'normal': {'origin': gate_center, 'end': gate_normal},
                'distance': gate_distance,
                'rotation': rotation
            })
            n += 1

        if self.render_perspective:
            self.render_perspective_grid(view)

        self.context.copy_framebuffer(fbo2, fbo1)

        # Loading the image using Pillow
        img = Image.frombytes(
            'RGBA', fbo2.size, fbo2.read(components=4, alignment=1), 'raw',
            'RGBA', 0, -1)

        annotations = {
            'bboxes': bounding_boxes,
            'drone_pose': self.drone_pose.translation,
            'drone_orientation': self.drone_pose.orientation
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

        return (img, annotations)
