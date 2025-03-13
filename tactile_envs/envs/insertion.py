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
from enum import Enum

from pathlib import Path

# external functions
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


class GraspPhase(Enum):
    """
    define the four phases of grasping
    """
    APPROACH = 0
    GRASP = 1
    LIFT = 2
    HOLD = 3

class RegraspEnv(gym.Env):

    def __init__(self, no_rotation=True, 
        no_gripping=True, state_type='vision_and_touch', camera_idx=0, symlog_tactile=True, 
        env_id = -1, im_size=128, tactile_shape=(20,20), skip_frame=5, max_delta=None, multiccd=False,
        render_mode=None):

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
        'render_mode': the mode for rendering the environment
        """

        # render mode   
        self.render_mode = render_mode

        # validate input parameters
        valid_state_types = ['privileged', 'vision', 'touch', 'vision_and_touch']
        if state_type not in valid_state_types:
            raise ValueError(f"state_type must be one of {valid_state_types}")
        
        if not isinstance(tactile_shape, tuple) or len(tactile_shape) != 2:
            raise ValueError("tactile_shape must be a tuple of length 2")
        
        if skip_frame < 1:
            raise ValueError("skip_frame must be positive")

        super(RegraspEnv, self).__init__()

        self.id: int = env_id
        self.skip_frame: int = skip_frame
        self.multiccd: bool = multiccd
        self.fixed_gripping: float = 160.0
        
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets')

        self.model_path = os.path.join(asset_folder, 'insertion/scene.xml')
        self.current_dir = os.path.join(Path(__file__).parent.parent.absolute(), 'assets/insertion')
        with open(self.model_path,"r") as f:
            self.xml_content = f.read()
            self.update_include_path()
            self.xml_content_reference = self.xml_content

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
        
        # initialize the renderer based on the render mode
        if self.render_mode == 'human':
            self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        
        self.CRUISE_HEIGHT: float = -0.03    # cruise height
        self.GRIPPING_HEIGHT: float = -0.13  # gripping height
        self.STEPS_PER_PHASE: int = 60      # steps per phase
        
        # initialize the step counter and phase
        self.current_step = 0
        self.max_steps = 300  # can be adjusted as needed
        
        # define the number of steps per phase
        self.phase_steps = {
            GraspPhase.APPROACH: 1,  # approach phase
            GraspPhase.GRASP: 1,    # grasp phase
            GraspPhase.LIFT: 1,     # lift phase
            GraspPhase.HOLD: 1      # hold phase
        }
        
        # initialize the current phase
        self.current_phase = GraspPhase.APPROACH
        
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
        """
        Update tactile feedback and display image.
        
        Args:
            show_full (bool): whether to display the full simulation image
        """
        try:
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
        except Exception as e:
            print(f"Error updating tactile feedback: {e}")
            raise

    def show_tactile(self, tactile, size=(480,480), max_shear=0.05, max_pressure=0.1, name='tactile'): 
        # Note: default params work well for 16x16 or 32x32 tactile sensors, adjust for other sizes
        ''' Visualize tactile sensor data'''
        nx = tactile.shape[2]
        ny = tactile.shape[1]

        loc_x = np.linspace(0,size[1],nx) # 生成x坐标
        loc_y = np.linspace(size[0],0,ny) # 生成y坐标

        img = np.zeros((size[0],size[1],3))

        for i in range(0,len(loc_x),1):
            for j in range(0,len(loc_y),1):
                
                dir_x = np.clip(tactile[0,j,i]/max_shear,-1,1) * 20 # 计算x方向的力
                dir_y = np.clip(tactile[1,j,i]/max_shear,-1,1) * 20 # 计算y方向的力

                color = np.clip(tactile[2,j,i]/max_pressure,0,1)
                r = color
                g = 1-color

                cv2.arrowedLine(img, (int(loc_x[i]),int(loc_y[j])), (int(loc_x[i]+dir_x),int(loc_y[j]-dir_y)), (0,g,r), 4, tipLength=0.5)

        cv2.imshow(name, img)

        return img

    def generate_initial_pose(self, show_full=True):
        """Initialize the environment"""
        mujoco.mj_resetData(self.sim, self.mj_data)
        
        # set the initial position
        initial_action = np.zeros(5)
        initial_action[2] = self.CRUISE_HEIGHT  # start from the cruise height
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
        self.random_offset_x = np.random.uniform(-0.08, 0.08)
        self.random_offset_y = np.random.uniform(-0.01, 0.01)

        # reload the environment
        self.edit_xml()
        self.sim = mujoco.MjModel.from_xml_string(self.xml_content)
        self.mj_data = mujoco.MjData(self.sim)
        
        if self.multiccd:
            self.sim.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
            
        # reload the renderer
        if hasattr(self, 'renderer'):
            del self.renderer
        self.renderer = mujoco.Renderer(self.sim, height=self.im_size, width=self.im_size)
        
        # initialize the position
        self.generate_initial_pose()
        self.current_phase = GraspPhase.APPROACH
        
        # reset the step counter and phase
        self.current_step = 0
        self.current_phase = GraspPhase.APPROACH
        
        # get the initial observation
        obs = self._get_obs()
        info = {'id': np.array([self.id])}
        
        return obs, info

    def render(self):
        """Render the current environment state"""
        if self.render_mode == 'human':
            self.renderer.render()
            return True
        return False


    def step(self, action):
        # 此处的 action 是此前的 phaseaction
        """Take a step in the environment"""
        # execute multiple simulation steps to ensure the action is completed
        # STEPS_PER_PHASE = 60
        for _ in range(self.STEPS_PER_PHASE):
            self.mj_data.ctrl[:] = action
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            # update the tactile feedback
            self.update_tactile_feedback(True)

        # get the observation and compute the reward
        obs = self._get_obs()
        
        # check if the grasping is successful
        if self.current_phase == GraspPhase.HOLD:
            reward = self._compute_reward(obs)
            is_success = self._check_success(obs)
        else:
            reward = 0.0
            is_success = False
        
        # check if the sequence is completed
        done = (self.current_phase == GraspPhase.HOLD)
        
        info = {
            'phase': self.current_phase.name,
            'phase_index': self.current_phase.value,
            'target_position': [self.offset_x, self.offset_y],
            'current_height': self.mj_data.ctrl[2],
            'is_success': is_success  # add the information of whether the grasping is successful
        }
        # print("info --> [", info['phase'], info['phase_index'], 
        #       info['target_position'], info['current_height'], 
        #       "success:", info['is_success'], "]")
        # print("it is the end of the step")

        return obs, reward, done, False, info

    def old_step(self, action):
        ''' Take a step in the environment '''
        # update the phase
        self._update_phase()
        
        # execute the action
        action = self.get_phase_action(action)
        
        # execute multiple simulation steps to ensure the action is completed
        for _ in range(self.STEPS_PER_PHASE):
            self.mj_data.ctrl[:] = action
            mujoco.mj_step(self.sim, self.mj_data, self.skip_frame + 1)
            # update the tactile feedback
            self.update_tactile_feedback(True)
        
        # get the observation and compute the reward
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        
        # check if the grasping is successful
        is_success = self._check_success(obs)
        
        # check if the sequence is completed
        done = (self.current_phase == GraspPhase.HOLD)
        
        info = {
            'phase': self.current_phase.name,
            'phase_index': self.current_phase.value,
            'target_position': [self.offset_x, self.offset_y],
            'current_height': self.mj_data.ctrl[2],
            'is_success': is_success  # add the information of whether the grasping is successful
        }
        # print("info --> [", info['phase'], info['phase_index'], 
        #       info['target_position'], info['current_height'], 
        #       "success:", info['is_success'], "]")
        # print("it is the end of the step")
        
        return obs, reward, done, False, info
    
    def _compute_reward(self, obs):
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

    def get_phase_action(self, target_action_xy):
        """
        Get the action for the current grasping phase.
        
        Args:
            action (np.ndarray): the action to be executed (1D array)
            
        Returns:
            np.ndarray: an array of 5 elements representing the action [x, y, z, yaw, gripper]
            
        Note:
            different phases correspond to different predefined actions:
            - approach: move to the object above
            - grasp: close the gripper
            - lift: lift the object
            - hold: hold the object
        """
        full_action = np.zeros(5)  # [x, y, z, yaw, gripper]

        # y_offset = 0 #0.015

        target_x = target_action_xy[0]
        target_y = target_action_xy[1]
        # print("target_pos (x,y) --> [", target_x, target_y, "]")
        
        if self.current_phase == GraspPhase.APPROACH:
            # move to the object above
            full_action[:3] = [target_x, target_y, self.CRUISE_HEIGHT]
            full_action[3] = np.pi/2  # fixed angle
            full_action[4] = 0        # gripper open
            full_action[:3] = [target_x, target_y, self.GRIPPING_HEIGHT] # move to the gripping height
            
        elif self.current_phase == GraspPhase.GRASP:
            full_action[:3] = [target_x, target_y, self.GRIPPING_HEIGHT] # move to the gripping height
            full_action[3] = np.pi/2 # fixed angle
            full_action[4] = self.fixed_gripping  # use the fixed gripping force
            
        elif self.current_phase == GraspPhase.LIFT:
            # lift the object
            full_action[4] = self.fixed_gripping
            full_action[:3] = [target_x, target_y, self.CRUISE_HEIGHT] # move to the cruise height
            full_action[3] = np.pi/2
            
        elif self.current_phase == GraspPhase.HOLD:
            # hold the object
            full_action[:3] = [target_x, target_y, self.CRUISE_HEIGHT]
            full_action[3] = np.pi/2
            full_action[4] = self.fixed_gripping
               
        return full_action
    
    def _update_phase(self):
        """Update the grasping phase"""
        # calculate the total number of steps for each phase
        total_steps = sum(self.phase_steps.values())
        steps_passed = self.current_step % total_steps
        
        # determine the current phase
        accumulated_steps = 0
        for phase in GraspPhase:
            accumulated_steps += self.phase_steps[phase]
            if steps_passed < accumulated_steps:
                self.current_phase = phase
                break

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'renderer'):
            del self.renderer
        super().close()

    def compute_force_percentage(self, obs):
        """Compute the percentage of the vertical force to the object's weight"""
        if self.current_phase != GraspPhase.HOLD:
            return 0.0

        tactile_data = obs['tactile']
        
        # 获取物体实际质量
        object_mass = 0.5  # 获取物体实际质量
        # print(f"\n物体信息:")
        # print(f"物体质量: {object_mass:.4f} kg")

        # tactile_data shape is (6,20,20), the first 3 channels are the right tactile sensors, 
        # the last 3 channels are the left tactile sensors
        # y direction tactile sum
        factor = 4.9050/12.30 # 合力与实际重力对齐
        y_sum_right = np.sum(tactile_data[1,:,:]) # right sensor y direction sum
        y_sum_left = np.sum(tactile_data[4,:,:])  # left sensor y direction sum
        total_y_sum = -(y_sum_right + y_sum_left) * factor
        
        # print("Y direction tactile sum: ", -total_y_sum, "N (Gravity direction)")

        # 计算理论重力
        gravity = 9.81  # m/s^2
        gravity_force = object_mass * gravity

        # 打印详细分析
        # print("\n传感器数据分析:")
        # print(f"理论重力: {gravity_force:.4f} N")
        
        # 计算力的比值
        percentage_force = total_y_sum / gravity_force
        if self.current_phase == GraspPhase.HOLD:
            print(f"力的比值: {percentage_force:.4f}")
        
        return percentage_force

    def _check_success(self, obs):
        """Check if the grasping is successful
        
        Success condition: in the HOLD phase, the vertical force is greater than or equal to 90% of the object's weight
        
        Returns:
            bool: whether the grasping is successful
        """
        percentage_force = self.compute_force_percentage(obs)
        threshold = 0.99 # 0.97
        is_success = percentage_force >= threshold
        
        return is_success