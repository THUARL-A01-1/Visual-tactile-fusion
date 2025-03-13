'''
RLGraspEnv1D defines the 1D grasping environment
'''
import gymnasium as gym
import numpy as np
from tactile_envs.envs.insertion import RegraspEnv, GraspPhase
import mujoco
import time
import os

def get_sorted_obj_list():
    """获取排序后的obj物体列表"""
    objects_list = []
    # 获取complex_objects/continue_1d目录下的所有xml文件
    dir_path = "tactile_envs/assets/insertion/complex_objects/continue_1d"
    xml_files = [f for f in os.listdir(dir_path) if f.endswith('.xml')]
    
    # 只保留形如obj_X的物体
    obj_files = [f for f in xml_files if f.startswith('obj_')]
    
    # 提取编号并排序
    obj_dict = {}
    for obj_file in obj_files:
        obj_name = obj_file[:-4]  # 移除.xml后缀
        obj_num = int(obj_name.split('_')[1])  # 获取数字部分
        obj_dict[obj_num] = obj_name
    
    # 按编号排序
    sorted_nums = sorted(obj_dict.keys())
    objects_list = [obj_dict[num] for num in sorted_nums]
    
    return objects_list

class RLGraspEnv1D(RegraspEnv):
    """一维抓取环境，简化了抓取任务为仅在x轴方向的移动"""
    
    def __init__(self, **kwargs):
        """初始化1D抓取环境"""
        # 在调用父类构造函数前先定义基本属性
        self.random_offset_range = (-0.08, 0.08)  # x轴随机偏移范围
        self.random_offset_x = 0.0  # 当前随机偏移值
        self.grasp_height = -0.13   # 抓取高度
        self.lift_height = -0.03    # 提升高度
        self.max_steps = 100
        self.current_step = 0
        self.current_episode = 0
        self.gravity_value = 9.81
        self.object = 'obj_0'
        
        # 初始化目标位置
        self.target_pos = np.array([0.0, 0.0, self.grasp_height])
        
        # 缓存相关
        self._cached_gripper_pos = None
        self._cached_step = -1
        
        # 定义每个阶段的步数（每个阶段1步）
        self.phase_steps = {
            GraspPhase.APPROACH: 1,  # 接近阶段
            GraspPhase.GRASP: 1,    # 抓取阶段
            GraspPhase.LIFT: 1,     # 提升阶段
            GraspPhase.HOLD: 1      # 保持阶段
        }
        
        # 初始化当前阶段
        self.current_phase = GraspPhase.APPROACH
        
        # 添加渲染控制变量
        self._last_render_step = -1
        
        try:
            super().__init__(**kwargs)
        except Exception as e:
            raise RuntimeError(f"初始化父类环境失败: {e}")
        
        # 参数验证
        if not hasattr(self, 'sim') or not hasattr(self, 'mj_data'):
            raise RuntimeError("MuJoCo模拟环境未正确初始化")
        
        # 验证动作空间范围的合理性
        if self.random_offset_range[0] >= self.random_offset_range[1]:
            raise ValueError("随机偏移范围设置不合理")
        
        # 重新定义动作空间
        self._setup_action_space()
        
        # 重新定义观察空间 (可选，如果需要简化观察)
        self._setup_observation_space()
        
        # 渲染相关
        self.render_mode = kwargs.get('render_mode', None)
        self.camera_idx = kwargs.get('camera_idx', 0)
        
        # 使用排序后的obj物体列表替换原有列表
        self.objects = get_sorted_obj_list()
        print(f"加载的物体列表: {self.objects}")
        
        # 确保当前物体在列表中，如果不在则使用第一个
        if self.object not in self.objects and len(self.objects) > 0:
            self.object = self.objects[0]

    def _setup_action_space(self):
        """设置1D动作空间"""
        scale_factor = 0.05
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]) * scale_factor,
            high=np.array([1.0]) * scale_factor,
            dtype=np.float32
        )
    
    def _setup_observation_space(self):
        """设置观察空间"""
        # 可以选择只保留需要的观察维度
        # 仅仅保留触觉
        self.observation_space = gym.spaces.Dict({
            'tactile': gym.spaces.Box(
                low=np.zeros(2 * 3 * self.tactile_rows * self.tactile_cols),
                high=np.ones(2 * 3 * self.tactile_rows * self.tactile_cols),
                dtype=np.float32
            )
        })

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 生成新的随机偏移
        # self.random_offset_x = np.random.uniform(*self.random_offset_range)
        self.random_offset_x = 0.0
        # 更新目标位置
        self.target_pos = np.array([self.random_offset_x, 0.0, self.grasp_height])
        
        # 调用父类的reset
        obs, info = super().reset(seed=seed, options=options)
        
        # 重置步数计数器和缓存
        self.current_step = 0
        self.current_episode += 1
        self._cached_step = -1
        self._last_render_step = -1
        
        # 重置当前阶段为APPROACH
        self.current_phase = GraspPhase.APPROACH
        
        return obs, info

    def step(self, action):
        """
        执行动作并返回结果
        执行一次 step 函数是完整的抓取动作
        """
        # 更新阶段
        # self._update_phase()
        print("this is step function of regrasp1d")
        # 转换1D动作为完整动作
        full_action = self._get_complete_action(action) # 转换为符合5维度的动作向量(也可能是4维度的，因为有掩码存在)
        target_action_xy = full_action[:2]
        target_action_xy[1] = 0.015
        # print("target_action_xy --> [", target_action_xy[0], target_action_xy[1], "]") # 打印目标位置xy坐标
        
        # 根据当前阶段修改动作
        # phase_action = self.get_phase_action(full_action) # 计算当前阶段实际需要执行的action，修改get_phase_action函数中的数据可以实际上修改动作模式
        self.current_phase = GraspPhase.APPROACH
        phase_action1 = self.get_phase_action(full_action)
        _, _, _, _, _ = super().step(phase_action1)
        self.current_phase = GraspPhase.GRASP
        phase_action2 = self.get_phase_action(full_action)
        _, _, _, _, _ = super().step(phase_action2)
        self.current_phase = GraspPhase.LIFT
        phase_action3 = self.get_phase_action(full_action)
        _, _, _, _, _ = super().step(phase_action3)
        self.current_phase = GraspPhase.HOLD
        phase_action4 = self.get_phase_action(full_action)
        # print("phase_action1 --> [", phase_action1[0], phase_action1[1], phase_action1[2], phase_action1[3], phase_action1[4], "]")
        # print("phase_action2 --> [", phase_action2[0], phase_action2[1], phase_action2[2], phase_action2[3], phase_action2[4], "]")
        # print("phase_action3 --> [", phase_action3[0], phase_action3[1], phase_action3[2], phase_action3[3], phase_action3[4], "]")
        # print("phase_action4 --> [", phase_action4[0], phase_action4[1], phase_action4[2], phase_action4[3], phase_action4[4], "]")
        # print("phase_action --> [", phase_action[0], phase_action[1], phase_action[2], phase_action[3], phase_action[4], "]")
        # 执行动作
        

        obs, reward, terminated, truncated, info = super().step(phase_action4)
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            truncated = True
        
        print("reward --> [", reward, "]", "terminated --> [", terminated, "]", "truncated --> [", truncated, "]", "info --> [", info, "]")
        return obs, reward, terminated, truncated, info

    def old_step(self, action):
        """执行动作并返回结果"""
        # 更新阶段
        self._update_phase()
        
        # 转换1D动作为完整动作
        full_action = self._get_complete_action(action) # 转换为符合5维度的动作向量(也可能是4维度的，因为有掩码存在)
        target_action_xy = full_action[:2]
        target_action_xy[1] = 0.015
        print("target_action_xy --> [", target_action_xy[0], target_action_xy[1], "]") # 打印目标位置xy坐标
        
        # 根据当前阶段修改动作
        # phase_action = self.get_phase_action(full_action) # 计算当前阶段实际需要执行的action，修改get_phase_action函数中的数据可以实际上修改动作模式
        phase_action = self.get_phase_action(full_action)
        print("phase_action --> [", phase_action[0], phase_action[1], phase_action[2], phase_action[3], phase_action[4], "]")
        # 执行动作
        obs, reward, terminated, truncated, info = super().step(phase_action)
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否达到最大步数
        if self.current_step >= self.max_steps:
            truncated = True
        
        return obs, reward, terminated, truncated, info


    def _get_complete_action(self, action_x):
        """将1D动作转换为完整的动作序列"""
        # 确保动作是数组格式
        if np.isscalar(action_x):
            action_x = np.array([action_x])
        
        # 创建完整动作序列（基于父类RegraspEnv的动作空间）
        complete_action = np.zeros(5)  # 固定为5维：x, y, z, yaw, gripper
        complete_action[0] = action_x[0]  # x轴移动
        complete_action[1] = 0.0       # y轴固定
        complete_action[2] = 0.0       # z轴由相位控制
        complete_action[3] = np.pi/2   # 夹爪角度固定
        complete_action[4] = 150.0     # 夹爪力度固定
        # print("complete_action --> [", complete_action[0], complete_action[1], complete_action[2], complete_action[3], complete_action[4], "]")
        
        # 根据动作掩码选择需要的维度
        if hasattr(self, 'action_mask'):
            complete_action = complete_action[self.action_mask]
        
        return complete_action

    def _get_obs(self):
        """获取观察"""
        # 先获取父类的观察
        obs = super()._get_obs()
        
        # 确保观察值的类型正确
        if isinstance(obs, dict):
            if 'image' in obs and obs['image'] is not None:
                # 确保图像数据是numpy数组
                if not isinstance(obs['image'], np.ndarray):
                    # 如果不是numpy数组，创建一个空的图像数组
                    obs['image'] = np.zeros((self.im_size, self.im_size, 3), dtype=np.float32)
                else:
                    obs['image'] = obs['image'].astype(np.float32)
                
            if 'tactile' in obs and obs['tactile'] is not None:
                # 确保触觉数据是numpy数组
                if not isinstance(obs['tactile'], np.ndarray):
                    # 如果不是numpy数组，创建一个空的触觉数组
                    obs['tactile'] = np.zeros((2 * 3, self.tactile_rows, self.tactile_cols), dtype=np.float32)
                else:
                    obs['tactile'] = obs['tactile'].astype(np.float32)
                    if self.symlog_tactile:
                        obs['tactile'] = np.sign(obs['tactile']) * np.log1p(np.abs(obs['tactile']))
        
        return obs
        
    def _compute_reward(self, obs):
        """计算奖励"""
        # print("compute reward 1d regrasp")
        gripper_pos = self._get_gripper_position()
        distance = np.linalg.norm(gripper_pos - self.target_pos)
        
        # 基础距离奖励
        reward = -distance
        
        # 根据不同阶段给予不同奖励
        if self.current_phase == GraspPhase.APPROACH:
            # 鼓励接近目标
            # reward = -distance * 2.0
            reward += 0
            
        elif self.current_phase == GraspPhase.GRASP:
            # 鼓励精确定位
            reward = -distance * 3.0
            if distance < 0.02:  # 2cm阈值
                # reward += 2.0
                reward += 0
                
        elif self.current_phase == GraspPhase.LIFT:
            # 鼓励抬升
            lift_reward = (gripper_pos[2] - self.grasp_height) * 5.0
            # reward += lift_reward
            reward += 0
            
        elif self.current_phase == GraspPhase.HOLD:
            # 鼓励稳定
            if distance < 0.05:  # 5cm阈值
                # reward += 1.0
                reward += 0
            # 检查是否成功
            is_success = self._check_success(obs)
            percentage_force = self.compute_force_percentage(obs)
            # print("current phase is hold, it is success --> ", is_success)
            if is_success:
                reward += 10.0 + percentage_force * 10.0
                print("sucessand reward is added--> ", 20.0)
            else:
                reward += percentage_force * 10.0
                print("fail and reward is added--> ", percentage_force * 10.0)
        
        return reward


        
    def _get_gripper_position(self):
        """获取夹爪位置（带缓存）"""
        if self._cached_step != self.current_step:
            self._cached_gripper_pos = self.mj_data.qpos[-7:-4].copy()
            self._cached_step = self.current_step
        # print("gripper_pos --> [", self._cached_gripper_pos[0], self._cached_gripper_pos[1], "]")
        return self._cached_gripper_pos

    def render(self):
        """渲染环境"""
        # 调用父类的渲染
        super().render()
        
        # 只在render_mode为human时输出信息，并且控制渲染频率
        if self.render_mode == 'human' and hasattr(self, '_last_render_step') and self._last_render_step != self.current_step:
            # print("\n" + "="*50)
            # print(f"Phase: {self.current_phase.name}")
            # print(f"Step: {self.current_step}/{self.max_steps}")
            # print(f"Gripper Position: {self._get_gripper_position()}")
            # print(f"Target Position: {self.target_pos}")
            # print(f"Distance: {np.linalg.norm(self._get_gripper_position() - self.target_pos):.4f}")
            # print("="*50 + "\n")
            self._last_render_step = self.current_step

    def _get_debug_info(self):
        """返回调试信息"""
        return {
            'env_state': {
                'phase': self.current_phase.name if hasattr(self, 'current_phase') else 'UNKNOWN',
                'step': self.current_step,
                'max_steps': self.max_steps,
                'gripper_pos': self._get_gripper_position().tolist(),
                'target_pos': self.target_pos.tolist(),
                'distance': np.linalg.norm(self._get_gripper_position() - self.target_pos)
            },
            'action_info': {
                'action_scale': self.action_space.high[0],
                'random_offset_range': self.random_offset_range,
                'current_offset': self.random_offset_x
            }
        }

    @classmethod
    def get_default_config(cls):
        """返回默认配置"""
        return {
            'random_offset_range': (-0.1, 0.1),
            'grasp_height': -0.13,
            'lift_height': -0.03,
            'max_steps': 100,
            'success_threshold': 0.02,
            'action_scale': 0.05,
            'reward_weights': {
                'distance': 1.0,
                'action_penalty': 0.1,
                'success_bonus': 10.0
            }
        }

    def _update_phase(self):
        """更新抓取阶段"""
        # 计算每个阶段的总步数
        total_steps = sum(self.phase_steps.values())
        
        # 计算当前所在的阶段
        steps_passed = self.current_step % total_steps
        accumulated_steps = 0
        
        # 遍历所有阶段，找到当前所处的阶段
        for phase, steps in self.phase_steps.items():
            accumulated_steps += steps
            if steps_passed < accumulated_steps:
                self.current_phase = phase
                break
            
        # 如果超过了所有阶段的总步数，重置为第一个阶段
        if steps_passed >= total_steps:
            self.current_phase = GraspPhase.APPROACH

    

if __name__ == "__main__":
    # 测试脚本
    try:
        print("Regrasp1D测试脚本运行，正在创建环境...")
        env = RLGraspEnv1D(
            no_rotation=True,
            no_gripping=False,
            state_type='vision_and_touch',
            camera_idx=0,
            symlog_tactile=True,
            im_size=480,
            tactile_shape=(20, 20),
            skip_frame=5,
            render_mode='human'
        )
        print("环境创建成功！")
        
        print(f"动作空间: {env.action_space}")
        print(f"观察空间: {env.observation_space}")
        
        episodes = 5
        for episode in range(episodes):
            print(f"\n开始第 {episode + 1} 回合测试")
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            # current_action = env.action_space.sample()
            while not done:
                # if env.current_phase == GraspPhase.APPROACH: # 在hold阶段，随机采样动作
                #     action = env.action_space.sample()
                #     current_action = action

                    
                # else:
                #     action = current_action
                #     _, _, terminated, truncated, info = env.step(action)
                #     episode_reward += reward
                #     done = terminated or truncated

                action = env.action_space.sample()
                print("this time action --> [", action, "]")
                obs, reward, terminated, truncated, info = env.step(action)
                # print("obs shape --> [", obs, "]")
                episode_reward += reward
                done = terminated or truncated
                
                
                print("new step")
                
                
                
                # if episode == 0:  # 只在第一个回合打印详细信息
                #     print("Episode 0 (only) print detailed info")
                #     print(f"Phase: {info['phase']}")
                #     print(f"Step: {info['step']}")
                #     print(f"Distance: {info['distance']:.4f}")
                #     print(f"Reward: {reward:.4f}")
                #     print("-" * 30)
            
            print(f"回合 {episode + 1} 结束")
            print(f"总奖励: {episode_reward:.4f}")
            print(f"是否成功: {info['is_success']}")
            # print(f"最终距离: {info['distance']:.4f}")
            print("="*50)
            
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()