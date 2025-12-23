import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from .utils import (
    generate_room,
    collect_entity_coordinates,
    format_coordinate_render,
)
# from gym_sokoban.envs.sokoban_env.utils import generate_room
from ragen.env.base import BaseDiscreteActionEnv
from ragen.env.sokoban.config import SokobanEnvConfig
from ragen.utils import all_seed

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):
    def __init__(self, config=None, **kwargs):
        self.config = config or SokobanEnvConfig()
        self.GRID_LOOKUP = self.config.grid_lookup
        self.ACTION_LOOKUP = self.config.action_lookup
        self.search_depth = self.config.search_depth
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.render_mode = self.config.render_mode
        self.observation_format = self.config.observation_format

        BaseDiscreteActionEnv.__init__(self)
        GymSokobanEnv.__init__(
            self,
            dim_room=self.config.dim_room, 
            max_steps=self.config.max_steps,
            num_boxes=self.config.num_boxes,
            **kwargs
        )

    def reset(self, seed=None, mode=None):
        try:
            with all_seed(seed):
                self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                    dim=self.dim_room,
                    num_steps=self.num_gen_steps,
                    num_boxes=self.num_boxes,
                    search_depth=self.search_depth
                )
            self.num_env_steps, self.reward_last, self.boxes_on_target = 0, 0, 0
            self.player_position = np.argwhere(self.room_state == 5)[0]
            return self.render()
        except (RuntimeError, RuntimeWarning) as e:
            next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
            return self.reset(next_seed)
    
    # --- 新增：死锁检测辅助函数 ---
    def _check_deadlock(self, box_pos):
        """
        检查箱子是否处于简单的角落死锁位置。
        """
        room_state = self.room_state
        row, col = box_pos
        
        # 获取墙壁位置 (通常 0 代表墙)
        walls = (room_state == 0)
        
        # 边界检查（防止索引越界，虽然 gym-sokoban 通常有围墙）
        if row == 0 or row == room_state.shape[0]-1 or col == 0 or col == room_state.shape[1]-1:
            return True

        top_wall = walls[row-1, col]
        bottom_wall = walls[row+1, col]
        left_wall = walls[row, col-1]
        right_wall = walls[row, col+1]
        
        # 只要满足任意一个角落条件，即视为死锁
        if (top_wall and left_wall) or \
           (top_wall and right_wall) or \
           (bottom_wall and left_wall) or \
           (bottom_wall and right_wall):
            return True
        
        return False
    # ---------------------------

    def step(self, action: int):
        previous_pos = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action) 
        
        # --- 修改：增加死锁检测逻辑 ---
        # 如果还没结束，检查是否有箱子死锁
        if not done:
            # 获取所有未归位箱子的位置 (room_state == 4 代表普通箱子)
            box_positions = np.argwhere(self.room_state == 4)
            for box_pos in box_positions:
                if self._check_deadlock(box_pos):
                    done = True
                    print("Deadlock detected at box position:", box_pos)
                    break
        # ---------------------------

        next_obs = self.render()
        action_effective = not np.array_equal(previous_pos, self.player_position)
        
        info = {
            "action_is_effective": action_effective, 
            "action_is_valid": True, 
            "success": self.boxes_on_target == self.num_boxes
        }
        
        # 可选：如果你想在 info 里标记是因为死锁结束的，可以加这一行
        if done and not info["success"]:
            info["deadlock"] = True

        return next_obs, reward, done, info

    def render(self, mode=None):
        if mode in {'grid', 'coord', 'grid_coord'}:
            return self._render_text(mode)

        render_mode = mode if mode is not None else self.render_mode
        if render_mode == 'text':
            return self._render_text(self.observation_format)
        if render_mode == 'rgb_array':
            return self.get_image(mode='rgb_array', scale=1)
        raise ValueError(f"Invalid mode: {render_mode}")

    def _render_text(self, observation_format: str) -> str:
        if observation_format == 'grid':
            room = np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
            return '\n'.join(''.join(self.GRID_LOOKUP.get(cell, "?") for cell in row) for row in room.tolist())
        if observation_format == 'coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return format_coordinate_render(entity_coords, self.dim_room)
        if observation_format == 'grid_coord':
            entity_coords = collect_entity_coordinates(self.room_state, self.room_fixed)
            return "Coordinates: \n" + format_coordinate_render(entity_coords, self.dim_room) + "\n" + "Grid Map: \n" + self._render_text('grid')
        raise ValueError(f"Invalid observation_format: {observation_format}")
    
    def get_all_actions(self):
        return list([k for k in self.ACTION_LOOKUP.keys()])
    
    def close(self):
        self.render_cache = None
        super(SokobanEnv, self).close()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=10)
    env = SokobanEnv(config)
    for i in range(10):
        print(env.reset(seed=1010 + i))
        print()
    while True:
        keyboard = input("Enter action: ")
        if keyboard == 'q':
            break
        action = int(keyboard)
        assert action in env.ACTION_LOOKUP, f"Invalid action: {action}"
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
    np_img = env.get_image('rgb_array')
    # save the image
    plt.imsave('sokoban1.png', np_img)
