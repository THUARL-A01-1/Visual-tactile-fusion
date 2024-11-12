'''
Defines the insertion environment for the tactile sensing task.
'''
import os
import cv2 
import time
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
import cv2

from pathlib import Path


def get_objects_list():
    ''' Get the list of objects '''
    objects_list = []
    # get all xml files in the complex_objects folder
    dir_1 = "tactile_envs/assets/insertion/complex_objects/continue_1d"
    dir_2 = "tactile_envs/assets/insertion/complex_objects/discrete_2d"
    xml_files_1 = [f for f in os.listdir(dir_1) if f.endswith('.xml')]
    xml_files_2 = [f for f in os.listdir(dir_2) if f.endswith('.xml')]
    xml_files = xml_files_1 + xml_files_2
    
    # remove the .xml suffix, and add to objects_list
    for xml_file in xml_files:
        obj_name = xml_file[:-4]  # remove the .xml suffix
        if obj_name not in objects_list:
            objects_list.append(obj_name)
    return objects_list

def convert_observation_to_space(observation):
    ''' Convert an observation dictionary to a gym space '''
    
    space = spaces.Dict(spaces={})
    for key in observation.keys():
        if key == 'image':
            space.spaces[key] = spaces.Box(low = 0, high = 1, shape = observation[key].shape, dtype = np.float64)
        elif key == 'tactile' or key == 'state':
            space.spaces[key] = spaces.Box(low = -float('inf'), high = float('inf'), shape = observation[key].shape, dtype = np.float64)
        
    return space




class RegraspEnv(gym.Env):

    def __init__(self, no_rotation=True, 
        no_gripping=True, state_type='vision_and_touch', camera_idx=0, symlog_tactile=True, 
        env_id = -1, im_size=128, tactile_shape=(20,20), skip_frame=5, max_delta=None, multiccd=False):

        """
        'no_rotation': if True, the robot will not be able to rotate its wrist
        'no_gripping': if True, the robot will keep the gripper opening at a fixed value
        'state_type': choose from 'privileged', 'vision', 'touch', 'vision_and_touch'
        'camera_idx': index of the camera to use
        'symlog_tactile': if True, the tactile values will be squashed using the symlog function
        'env_id': environment id
        'im_size': side of the square image
        'tactile_shape': shape of the tactile sensor (rows, cols)
        'skip_frame': number of frames to skip between actions
        'max_delta': maximum change allowed in the x, y, z position
        'multiccd': if True, the multiccd flag will be enabled (makes tactile sensing more accurate but slower) # multiple-contact collision detection
        """

        super(RegraspEnv, self).__init__()

        self.id = env_id

        self.skip_frame = skip_frame # number of frames to skip between actions
        
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')

        self.model_path = os.path.join(asset_folder, 'insertion/scene.xml')
        self.current_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'assets/insertion')
        with open(self.model_path,"r") as f:
            self.xml_content = f.read()
            self.update_include_path()
            self.xml_content_reference = self.xml_content

        self.multiccd = multiccd # if True, the multiccd flag will be enabled

        # adjustable parameter
        # the larger the value, the greater the gripping force
        self.fixed_gripping = 160  # fixed gripper value

        self.max_delta = max_delta

        self.symlog_tactile = symlog_tactile # used to squash tactile values and avoid large spikes

        self.tactile_rows = tactile_shape[0]
        self.tactile_cols = tactile_shape[1]
        self.tactile_comps = 3 # tactile components (x, y, z)

        self.im_size = im_size

        self.state_type = state_type

        # get the list of objects, and set the object to be grasped
        self.objects = get_objects_list()
        self.object = 'obj_1' # also can be set like this: self.object = self.objects[12]
        if self.object not in self.objects:
            print("Valid objects list: ", self.objects)
            raise ValueError("Invalid object")


        if self.state_type == 'privileged':
            self.curr_obs = {'state': np.zeros(40)}
        elif self.state_type == 'vision':
            self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3))}
        elif self.state_type == 'touch':
            self.curr_obs = {'tactile': np.zeros((2 * self.tactile_comps, self.tactile_rows, self.tactile_cols))} # 2 * 3 * 20 * 20, comps = components
        elif self.state_type == 'vision_and_touch':
            self.curr_obs = {'image': np.zeros((self.im_size, self.im_size, 3)), 'tactile': np.zeros((2 * self.tactile_comps, self.tactile_rows, self.tactile_cols))}
        else:
            raise ValueError("Invalid state type")
        
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD

        self.init_z = self.mj_data.qpos[-5] # initial z position of the object

        self.adaptive_gripping = not no_gripping
        self.with_rotation = not no_rotation

        self.camera_idx = camera_idx        
        
        obs_tmp = self._get_obs() # get observation space
        self.observation_space = convert_observation_to_space(obs_tmp)
        
        self.ndof_u = 5 # x, y, z, yaw, gripper
        if no_rotation:
            self.ndof_u -= 1
        if no_gripping:
            self.ndof_u -= 1

        
        
        # action space lower and upper bounds
        self.action_space = spaces.Box(low = np.full(self.ndof_u, -1.), high = np.full(self.ndof_u, 1.), dtype = np.float32) # x, y, z, yaw, gripper
        self.action_scale = np.array([[-0.2,0.2],[-0.2,0.2],[-0.12,0.3],[-np.pi,np.pi],[0,255]])
        # self.action_scale = np.array([[-2,2],[-2,2],[-2,2],[-np.pi,np.pi],[0,255]])

        self.action_mask = np.ones(5, dtype=bool) 
        if no_rotation:
            self.action_mask[3] = False
        if no_gripping:
            self.action_mask[4] = False
        self.action_scale = self.action_scale[self.action_mask]
        
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        
        # 关于抓取动作过程
        self.grasp_phase = 0  # 当前抓取阶段
        self.grasp_sequence = [
            'approach',  # 接近物体
            'grasp',     # 闭合夹爪
            'lift',      # 提起物体
            'hold'     # 保持
            #'place',     # 放下
            #'release'    # 释放夹爪
        ]
        # 定义关键高度参数
        self.cruise_height = -0.03    # 巡航高度
        self.gripping_height = -0.13  # 抓取高度
        self.steps_per_phase = 60  # 每个阶段的步数

        # print information
        print("ndof_u: ", self.ndof_u) # number of degrees of freedom
        print("object: ", self.object) # object name
        print("state_type: ", self.state_type) # state type
        # print("objects list: ", self.objects) # objects list

    def update_include_path(self):
        ''' Update the path of the included files in the XML '''
        file_idx = self.xml_content.find('<include file="', 0)
        while file_idx != -1:
            file_start_idx = file_idx + len('<include file="')
            self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]

            file_idx = self.xml_content.find('<include file="', file_start_idx + len(self.current_dir))

        file_idx = self.xml_content.find('meshdir="', 0)
        file_start_idx = file_idx + len('meshdir="')
        self.xml_content = self.xml_content[:file_start_idx] + self.current_dir + '/' + self.xml_content[file_start_idx:]

    def edit_xml(self):
        ''' Edit the XML file to change the object and its position'''
        # self.xml_content

        object = self.object
        if object not in self.objects:
            raise ValueError("Invalid object")
        
        # find the scene.xml file
        scene_xml_path = "tactile_envs/assets/insertion/scene.xml"
        if not os.path.exists(scene_xml_path):
            raise FileNotFoundError(f"cannot find scene.xml file: {scene_xml_path}")

        # read the scene.xml file content
        with open(scene_xml_path, 'r') as f:
            scene_content = f.read()
        
        # find the line starting with <include file="complex_objects
        include_line = None
        for line in scene_content.split('\n'):
            if line.strip().startswith('<include file="complex_objects'):
                include_line = line.strip() 
                # print("successfully find include_line: ", include_line) 
                break
        
        # object name
        if object.startswith("obj"):
            include_str = '<include file="complex_objects/continue_1d/' + object + '.xml"/>'
            scene_content = scene_content.replace(include_line, include_str)            
        else:
            include_str = '<include file="complex_objects/discrete_2d/' + object + '.xml"/>'
            scene_content = scene_content.replace(include_line, include_str)

        # write the modified scene.xml file
        with open(scene_xml_path, 'w') as f:
            f.write(scene_content)
        # print("object modified --> ", object)
        
        #TODO: object position modification
        offset_x = 0
        offset_y = 0

        if self.with_rotation:
            # set random object angle
            # offset_yaw = 2*np.pi*np.random.rand()-np.pi
            offset_yaw = 0
        else:
            offset_yaw = 0.

        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_yaw = offset_yaw
        self.target_quat = np.array([np.cos(offset_yaw/2), 0, 0, np.sin(offset_yaw/2)])

    def update_tactile_feedback(self, show_full=False):
        ''' Update the tactile feedback and display the image '''
        if self.state_type == 'vision_and_touch' or self.state_type == 'touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] 
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]]
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0) # 6 * 20 * 20
            
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            
            if self.state_type == 'vision_and_touch':
                img = self.render()
                self.curr_obs = {'image': img, 'tactile': tactiles}
            else:
                self.curr_obs = {'tactile': tactiles}

        if show_full:
            self.renderer.update_scene(self.mj_data, camera=0)
            img = cv2.cvtColor(self.renderer.render(), cv2.COLOR_BGR2RGB)
            cv2.imshow('simulation img', img)
            cv2.waitKey(1)

        img_tactile1 = self.show_tactile(self.curr_obs['tactile'][:3], name='tactile1') 
        img_tactile2 = self.show_tactile(self.curr_obs['tactile'][3:], name='tactile2')

    def show_tactile(self, tactile, size=(480,480), max_shear=0.05, max_pressure=0.1, name='tactile'): 
        # Note: default params work well for 16x16 or 32x32 tactile sensors, adjust for other sizes
        ''' Visualize tactile sensor data'''
        nx = tactile.shape[2]
        ny = tactile.shape[1]

        loc_x = np.linspace(0,size[1],nx)
        loc_y = np.linspace(size[0],0,ny)

        img = np.zeros((size[0],size[1],3))

        for i in range(0,len(loc_x),1):
            for j in range(0,len(loc_y),1):
                
                dir_x = np.clip(tactile[0,j,i]/max_shear,-1,1) * 20
                dir_y = np.clip(tactile[1,j,i]/max_shear,-1,1) * 20

                color = np.clip(tactile[2,j,i]/max_pressure,0,1)
                r = color
                g = 1-color

                cv2.arrowedLine(img, (int(loc_x[i]),int(loc_y[j])), (int(loc_x[i]+dir_x),int(loc_y[j]-dir_y)), (0,g,r), 4, tipLength=0.5)

        cv2.imshow(name, img)

        return img

    def generate_initial_pose(self, show_full=True):
        """Only initialize the environment"""
        mujoco.mj_resetData(self.sim, self.mj_data)
        
        # set the initial position
        initial_action = np.zeros(5)
        initial_action[2] = self.cruise_height  # start from the cruise height
        initial_action[3] = np.pi/2
        initial_action[4] = 0  # gripper open
        
        self.mj_data.ctrl[:] = initial_action
        mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
        self.update_tactile_feedback(show_full)
        
        self.prev_action_xyz = initial_action[:3]

    def _get_obs(self):
        ''' Get the current observation '''
        return self.curr_obs
    
    def get_proprio(self):
        ''' Get the proprioceptive state of the robot '''
        left_finger = self.mj_data.site("finger_left").xpos
        right_finger = self.mj_data.site("finger_right").xpos
        distance = np.linalg.norm(left_finger - right_finger)
        robot_state = np.concatenate([self.mj_data.qpos.copy()[:4], [distance]])
        return robot_state
    
    def seed(self, seed):
        ''' Seed the environment '''
        np.random.seed(seed) # set the seed for numpy
    
    def reset(self, seed=None, options=None):
        ''' Reset the environment '''
        if seed is not None:
            self.seed(seed)
            
        # set the random offset
        # this randomness is used to change the target position every episode
        self.random_offset_x = np.random.uniform(-0.01, 0.01)
        self.random_offset_y = np.random.uniform(-0.01, 0.01)

        # reload the environment
        self.edit_xml()
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.sim)
        
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
            
        # reload the renderer
        del self.renderer
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        
        # initialize the position
        self.generate_initial_pose()
        self.grasp_phase = 0
        
        # get the initial observation
        obs = self._get_obs()
        info = {'id': np.array([self.id])}
        
        return obs, info

    def render(self, highres = False):
        ''' Render the current scene '''
        
        if highres:
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=1000, width=1000)
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render()/255
            del self.renderer
            self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        else:
            self.renderer.update_scene(self.mj_data, camera=self.camera_idx)
            img = self.renderer.render()/255
        
        return img

    def step(self, u):
        ''' Take a step in the environment '''
        # get the full action for the current phase
        action = self.get_phase_action(self.grasp_phase)
        
        # execute multiple simulation steps to ensure the action is completed
        for _ in range(self.steps_per_phase):
            self.mj_data.ctrl[:] = action
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            # update the tactile feedback
            self.update_tactile_feedback(True)
        
        # get the observation and compute the reward
        obs = self._get_obs()
        reward = self.compute_reward()
        
        # update the grasping phase
        self.grasp_phase = (self.grasp_phase + 1) % len(self.grasp_sequence)
        
        # check if the sequence is completed
        done = (self.grasp_phase == 0)
        
        info = {
            'phase': self.grasp_sequence[self.grasp_phase],
            'phase_index': self.grasp_phase,
            'target_position': [self.offset_x, self.offset_y],
            'current_height': self.mj_data.ctrl[2]
        }
        
        return obs, reward, done, False, info
    
    def compute_reward(self):
        ''' Compute the reward for the current step '''
        # here according to the original reward definition
        # can be overloaded by subclasses
        pos = self.mj_data.qpos[-7:-4]
        quat = self.mj_data.qpos[-4:]

        delta_x = pos[0] - self.offset_x
        delta_y = pos[1] - self.offset_y
        delta_z = pos[2] - self.init_z
        delta_quat = np.linalg.norm(quat - self.target_quat)

        reward = -np.log(100 * np.sqrt(delta_x**2 + delta_y**2 + delta_z**2 + int(self.with_rotation) * delta_quat**2) + 1)

        return reward

    def get_phase_action(self, phase):
        ''' 
        Get the action for the current phase
        different phases have different actions
        '''
        full_action = np.zeros(5)  # [x, y, z, yaw, gripper]

        target_x = self.offset_x + self.random_offset_x
        target_y = self.offset_y + 0.015 # manually compensate (left-back in the figure)
        print("target_pos --> [", target_x, target_y, "]")
        
        if self.grasp_sequence[phase] == 'approach':
            # move to the object above
            full_action[:3] = [target_x, target_y, self.cruise_height]
            full_action[3] = np.pi/2  # fixed angle
            full_action[4] = 0        # gripper open
            full_action[:3] = [target_x, target_y, self.gripping_height] # move to the gripping height
            
        elif self.grasp_sequence[phase] == 'grasp':
            full_action[:3] = [target_x, target_y, self.gripping_height] # move to the gripping height
            full_action[3] = np.pi/2 # fixed angle
            full_action[4] = self.fixed_gripping  # use the fixed gripping force
            
        elif self.grasp_sequence[phase] == 'lift':
            # lift the object
            full_action[4] = self.fixed_gripping
            full_action[:3] = [target_x, target_y, self.cruise_height] # move to the cruise height
            full_action[3] = np.pi/2
            
        elif self.grasp_sequence[phase] == 'hold':
            # hold the object
            full_action[:3] = [target_x, target_y, self.cruise_height]
            full_action[3] = np.pi/2
            full_action[4] = self.fixed_gripping
               
        return full_action
    



    # old functions, for reference
    # to be deleted in the future
    def old_edit_xml(self):
        ''' Edit the XML file to change the object and holder '''
        
        # holders = self.holders
        objects = self.objects

        self.xml_content = self.xml_content_reference # reset the XML content

        def edit_attribute(attribute, offset_x, offset_y, offset_yaw, object):
            ''' Edit the position and orientation of the object or holder '''
            box_idx = self.xml_content.find('<body name="' + attribute + '"')
            if box_idx == -1:
                 print("ERROR: Could not find joint name: " + attribute)
                 return False
            
            pos_key = 'pos="'
            pos_idx = box_idx + self.xml_content[box_idx:].find(pos_key)
            pos_start_idx = pos_idx + len(pos_key)
            pos_end_idx = pos_start_idx + self.xml_content[pos_start_idx:].find('"')

            pos = self.xml_content[pos_start_idx:pos_end_idx].split(" ")
            correction_rot = np.array([float(pos[0]), float(pos[1])])
            rotMatrix = np.array([[np.cos(offset_yaw), -np.sin(offset_yaw)], 
                         [np.sin(offset_yaw),  np.cos(offset_yaw)]])
            correction_rot = rotMatrix.dot(correction_rot)
            
            new_pos = [str(offset_x + correction_rot[0]), str(offset_y + correction_rot[1]), str(float(pos[2]))]
            new_pos_str = " ".join(new_pos)

            if attribute == 'object':
                print(f"New object position: x={new_pos[0]}, y={new_pos[1]}, z={new_pos[2]}")

            
            self.xml_content = self.xml_content[:pos_start_idx] + new_pos_str + self.xml_content[pos_end_idx:]

            euler_key = 'axisangle="'
            euler_idx = box_idx + self.xml_content[box_idx:].find(euler_key)
            euler_start_idx = euler_idx + len(euler_key)
            euler_end_idx = euler_start_idx + self.xml_content[euler_start_idx:].find('"')

            euler = self.xml_content[euler_start_idx:euler_end_idx].split(" ")
            new_euler = [str(float(euler[0])), str(float(euler[1])), str(float(euler[2])), str(float(euler[3]) + offset_yaw)]
            new_euler_str = " ".join(new_euler)
            
            self.xml_content = self.xml_content[:euler_start_idx] + new_euler_str + self.xml_content[euler_end_idx:]
            
            if attribute == 'object': # change the mesh of the object
                for key in ['peg_visual', 'peg_collision']:
                    key_idx = euler_end_idx + self.xml_content[euler_end_idx:].find('name="' + key + '"')
                    key_end_idx = key_idx + len('name="' + key + '"')
            
                    mesh_idx = key_end_idx + self.xml_content[key_end_idx:].find('mesh="')
                    mesh_start_idx = mesh_idx + len('mesh="')
                    mesh_end_idx = mesh_start_idx + self.xml_content[mesh_start_idx:].find('"')

                    self.xml_content = self.xml_content[:mesh_start_idx] + object + self.xml_content[mesh_end_idx:]

                    
            else:
                for i in range(1,5):
                    break
                #     for key in ['wall{}_visual'.format(i), 'wall{}_collision'.format(i)]:
                #         key_idx = euler_end_idx + self.xml_content[euler_end_idx:].find('name="' + key + '"')
                #         key_end_idx = key_idx + len('name="' + key + '"')
                    
                #         mesh_idx = key_end_idx + self.xml_content[key_end_idx:].find('mesh="')
                #         mesh_start_idx = mesh_idx + len('mesh="')
                #         mesh_end_idx = mesh_start_idx + self.xml_content[mesh_start_idx:].find('"')

                #         self.xml_content = self.xml_content[:mesh_start_idx] + object + '_wall' + str(i) + self.xml_content[mesh_end_idx:]
                
            return True
            
        
        offset_x = 0
        offset_y = 0

        if self.with_rotation:
            # 可以设置随机的物体角度
            # offset_yaw = 2*np.pi*np.random.rand()-np.pi
            offset_yaw = 0
        else:
            offset_yaw = 0.

        # object = np.random.choice(objects)
        object = objects[1]

        edit_attribute("object", offset_x, offset_y, offset_yaw, object)
        edit_attribute("walls", offset_x, offset_y, offset_yaw, object)

        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_yaw = offset_yaw
        self.target_quat = np.array([np.cos(offset_yaw/2), 0, 0, np.sin(offset_yaw/2)])

    def old_step(self, u):
        ''' Take a step in the environment '''
        action = u
        action = np.clip(u, -1., 1.)

        # # 根据当前阶段执行相应的动作
        # action = self.convert_action_to_sequence(action, self.grasp_phase)

        # Unnormalize action
        action_unnorm = (action + 1) / 2 * (self.action_scale[:, 1] - self.action_scale[:, 0]) + self.action_scale[:, 0] # 将归一化的动作转换为实际的控制量

        if self.max_delta is not None:
            action_unnorm = np.clip(action_unnorm[:3], self.prev_action_xyz - self.max_delta, self.prev_action_xyz + self.max_delta)

        self.prev_action_xyz = action_unnorm

        # 控制抓取力和旋转角度
        if self.with_rotation:
            self.mj_data.ctrl[3] = -action_unnorm[3]
        else:
            self.mj_data.ctrl[3] = 0
        if not self.adaptive_gripping:
            self.mj_data.ctrl[-1] = self.fixed_gripping  # 保持固定抓取力
        else:
            self.mj_data.ctrl[-1] = action_unnorm[-1]  # 控制抓取力

        # 控制机械臂的位置
        self.mj_data.ctrl[:3] = action_unnorm[:3]

        # 进行仿真步骤，更新状态
        mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)

        # 实时更新触觉反馈
        if self.state_type == 'vision_and_touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]]  # 转换坐标轴顺序
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]]
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            img = self.render()
            self.curr_obs = {'image': img, 'tactile': tactiles}

        elif self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}

        elif self.state_type == 'touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]]
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]]
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles)) # 对触觉数据进行对称对数变换
            self.curr_obs = {'tactile': tactiles}

        elif self.state_type == 'privileged':
            self.curr_obs = {'state': np.concatenate((self.mj_data.qpos.copy(), self.mj_data.qvel.copy(), [self.offset_x, self.offset_y, self.offset_yaw]))}

        # 计算奖励
        pos = self.mj_data.qpos[-7:-4]
        quat = self.mj_data.qpos[-4:]

        delta_x = pos[0] - self.offset_x
        delta_y = pos[1] - self.offset_y
        delta_z = pos[2] - self.init_z
        delta_quat = np.linalg.norm(quat - self.target_quat)

        reward = -np.log(100 * np.sqrt(delta_x**2 + delta_y**2 + delta_z**2 + int(self.with_rotation) * delta_quat**2) + 1)

        # 判断任务是否完成
        done = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2) < 4e-3
        if done:
            reward = 1000

        info = {'is_success': done}

        # 返回观测结果、奖励、完成状态等
        obs = self._get_obs()

        print(">>>>>>>>>>reward: ", reward)

        # 更新抓取阶段 # new 
        self.grasp_phase = (self.grasp_phase + 1) % len(self.grasp_sequence)
        
        # 如果完成了整个抓取序列，设置done为True # new
        if self.grasp_phase == 0:  # 回到初始阶段
            done = True

        return obs, reward, done, False, info

    def old_reset(self, seed=None, options=None):
        ''' 
        Reset the environment
        func: reset the environment and return the observation and info
        '''

        if seed is not None:
            np.random.seed(seed)
        
        # Reload XML (and update robot)
        self.edit_xml()
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        
        self.mj_data = mujoco.MjData(self.sim)
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD 
        
        del self.renderer
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)

        self.generate_initial_pose() # 生成初始位姿
        self.grasp_phase = 0 # new # 初始化抓取阶段

        if self.state_type == 'vision_and_touch': 
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] # zxy -> xyz
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]] # zxy -> xyz
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile: # 对触觉数据进行对称对数变换
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            img = self.render()
            self.curr_obs = {'image': img, 'tactile': tactiles}
        elif self.state_type == 'vision':
            img = self.render()
            self.curr_obs = {'image': img}
        elif self.state_type == 'touch':
            tactiles_right = self.mj_data.sensor('touch_right').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_right = tactiles_right[[1, 2, 0]] # zxy -> xyz
            tactiles_left = self.mj_data.sensor('touch_left').data.reshape((3, self.tactile_rows, self.tactile_cols))
            tactiles_left = tactiles_left[[1, 2, 0]] # zxy -> xyz
            tactiles = np.concatenate((tactiles_right, tactiles_left), axis=0)
            if self.symlog_tactile:
                tactiles = np.sign(tactiles) * np.log(1 + np.abs(tactiles))
            self.curr_obs = {'tactile': tactiles}
        elif self.state_type == 'privileged':
            self.curr_obs = {'state': np.concatenate((self.mj_data.qpos.copy(), self.mj_data.qvel.copy(), [self.offset_x,self.offset_y,self.offset_yaw]))}
        
        info = {'id': np.array([self.id])}

        
        return self._get_obs(), info

    def old_generate_initial_pose(self, show_full=True):
        ''' Generate the initial pose & movements of the robot '''
        
        # # 原始参数
        cruise_height = -0.03
        gripping_height = -0.13 #-0.11

        # 自定义参数
        # cruise_height = -0.005 # 巡航高度
        # gripping_height = 0 # 抓取高度
        
        mujoco.mj_resetData(self.sim, self.mj_data)

        # Randomize the initial position of the robot
        rand_x = np.random.rand()*0.2 - 0.1 
        rand_y = np.random.rand()*0.2 - 0.1
        if self.with_rotation:
            rand_yaw = np.random.rand()*2*np.pi - np.pi
        else:
            rand_yaw = 0

        '''debug，调整位置，random的动作的最终位置'''
        # rand_x = 0
        # rand_y = 0
        rand_yaw = np.pi/2

        displacement_x = 0

        steps_per_phase = 60 
        ##########################################
        # new movements below
        for _ in range(1):  # 抓取、放开、旋转
            # displacement_x -= 0.005
            # displacement_x = 0
            # for i in range(steps_per_phase):  # 旋转角度
            #     self.mj_data.ctrl[3] = rand_yaw
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            #     # 实时更新触觉反馈
            #     self.update_tactile_feedback(show_full)

            # for i in range(steps_per_phase):  # 移动到物体附近
            #     self.mj_data.ctrl[:3] = [self.offset_x + displacement_x, self.offset_y, gripping_height]
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            #     # 实时更新触觉反馈
            #     self.update_tactile_feedback(show_full)
            
            # for i in range(steps_per_phase):  # 抓住物体
            #     self.mj_data.ctrl[-1] = self.fixed_gripping
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            #     # 实时更新触觉反馈
            #     self.update_tactile_feedback(show_full)

            # for i in range(steps_per_phase): # lift object
            #     self.mj_data.ctrl[:3] = [self.offset_x+ displacement_x, self.offset_y, cruise_height]
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            #     self.update_tactile_feedback(show_full)

            # for i in range(steps_per_phase): # lay down object
            #     self.mj_data.ctrl[:3] = [self.offset_x+ displacement_x, self.offset_y, gripping_height]
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame+1)
            #     self.update_tactile_feedback(show_full)

            for i in range(steps_per_phase):  # 放开物体
                self.mj_data.ctrl[-1] = 0  # 放开
                mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
                # 实时更新触觉反馈
                self.update_tactile_feedback(show_full)
            
            for i in range(steps_per_phase):  # 旋转角度
                self.mj_data.ctrl[3] = rand_yaw
                mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
                # 实时更新触觉反馈
                self.update_tactile_feedback(show_full)

            for i in range(steps_per_phase):  # 移动到物体附近
                self.mj_data.ctrl[:3] = [self.offset_x + displacement_x, self.offset_y, gripping_height]
                mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
                # 实时更新触觉反馈
                self.update_tactile_feedback(show_full)

            # for i in range(steps_per_phase):  # 旋转角度
            #     self.mj_data.ctrl[3] = 0
            #     mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            #     # 实时更新触觉反馈
            #     self.update_tactile_feedback(show_full)


        ##########################################


        self.prev_action_xyz = np.array([rand_x, rand_y, cruise_height])

        pos = self.mj_data.qpos[-7:-4]
        
        if pos[2] < (cruise_height - gripping_height)/2:
            print('Failed to grasp')
    

