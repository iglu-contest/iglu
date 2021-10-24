import math
import textwrap
import os 

import gym

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import cv2

from ..tasks import TaskSet, Task, CustomTasks
from .common import DEFAULT_POS

class Renderer:
    def __init__(self, width, height, h264=False, fps=60):
        self.width = width
        self.height = height
        self.env = None
        self.fps = fps
        self.h264 = h264
        self.done = None
        self.rendered_texts = {}
        self.text_frac = 0.4

    def create_env(self, external_env=None, task_target=None):
        if external_env is None:
            self.env = gym.make('IGLUSilentBuilder-v0', 
                resolution=(self.width, self.height), bound_agent=False, 
                action_space='continuous', max_steps=500000
            )
            self.done = False
            if task_target is not None:
                self.env.update_taskset(CustomTasks([('', task_target)]))
        # self.env.reset()
        # wait for the sun to rise
        # self.env.step(self.env.action_space.no_op())
        # self.env.step(self.env.action_space.no_op())
        # self.env.step(self.env.action_space.no_op())

    def init_conditions(self, init_pos, init_blocks):
        self.env.unwrapped.start_position = init_pos
        self.env.unwrapped.initial_blocks = init_blocks
        self.env.should_reset(True)

    def render_video(self, output, 
            event_sequence, init_conds=None, on_env_step=None,
            render_text=False, text=None, render_crosshair=False, 
            render_debug=False, reset=True, close=True, writer=None,
            verbose=False
        ):
        # TODO: set current task for the env
        if init_conds is not None:
            self.init_conditions(*init_conds)
        elif reset:
            self.init_conditions(DEFAULT_POS, None)
        if reset:
            obs = self.env.reset()
        done = False
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output.parent.mkdir(exist_ok=True)
            if render_text:
                width = self.width + int(math.ceil(self.width * self.text_frac))
            else:
                width = self.width
            writer = cv2.VideoWriter(f'{output}.mp4', fourcc, self.fps, (width, self.height))
            if reset:
                image = self.postprocess_frame(
                    obs['pov'][:], render_text=render_text, render_crosshair=render_crosshair,
                    render_debug=render_debug, text=text,
                    g=None, n=None, last_action='None'
                )
                if callable(on_env_step):
                    on_env_step(None, -1, 0, output, obs, 0, done, None)
                writer.write(np.transpose(image, (0, 1, 2))[:, :, ::-1])
        tq = tqdm(total=len(event_sequence), disable=not verbose)
        i = 0
        last_key_action = 'None'
        while not done:
            if i == len(event_sequence):
                done = True
                break
            event = event_sequence[i]        
            if event.kind == 'action':
                last_key_action = event.params[0]
            for k, action in enumerate(event.actions):
                obs, reward, done, info = self.env.step(action)
                image = self.postprocess_frame(
                    obs['pov'][:], render_text=render_text, render_crosshair=render_crosshair,
                    render_debug=render_debug, text=text,
                    g=event.turn, n=event.step, last_action=last_key_action
                )
                if callable(on_env_step):
                    on_env_step(event, i, k, output, obs, reward, done, info)
                writer.write(np.transpose(image, (0, 1, 2))[:, :, ::-1])
            tq.update(1)
            i += 1 
        tq.close()
        if close:
            writer.release()
            self.env.close()
            # self.postproc_video(output)
        else:
            return writer

    def postproc_video(self, output):
        if self.h264:
            code = os.system(f"ffmpeg -hide_banner -loglevel error -i {output}.mp4 -vcodec libx264 {output}2.mp4")
            if code == 0:
                os.system(f"mv {output}2.mp4 {output}.mp4")
            else:
                raise ValueError('Install the latest version of ffmpeg')
    
    def put_multiline_text(self, lines, height, width):
        if tuple(lines) not in self.rendered_texts:
            width = int(math.ceil(width * self.text_frac))
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            canvas = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas)
            fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 22)
            char_width = np.mean([fnt.getsize(char)[0] for char in set(' '.join([l for l in lines if l is not None]))])
            chars = int(0.9 * canvas.size[0] / char_width)
            text = []
            for line in lines:
                if line is None: continue
                lined = textwrap.fill(line, width=chars)
                text.append(lined)
            text = '\n'.join(text)
            pos = (int(0.05 * canvas.size[0]), int(0.03 * canvas.size[1]))
            draw.multiline_text(pos, text, font=fnt, fill=(0,0,0,0), stroke_width=1)
            self.rendered_texts[tuple(lines)] = np.array(canvas)
        return np.copy(self.rendered_texts[tuple(lines)])

    def postprocess_frame(self, 
            frame, render_debug=False, render_text=False,  render_crosshair=False,
            text=None, **kwargs
        ):
        image = np.copy(frame)
        if render_crosshair:
            d = int(math.floor(0.01 * (image.shape[0] + image.shape[1]) / 2))
            image[image.shape[0]//2-d:image.shape[0]//2+d+1,image.shape[1]//2] = 0
            image[image.shape[0]//2, image.shape[1]//2-d:image.shape[1]//2+d+1] = 0
        if render_debug:
            g = kwargs.get('g', 0)
            n = kwargs.get('n', 0)
            i = kwargs.get('i', 0)
            last_action = kwargs.get('last_action', None)
            image = cv2.putText(
                img=image, text=f'{g} {n}', 
                org=(int(0.01 * self.width), int(0.97 * self.height)), fontFace=3, 
                fontScale=1, color=(0,0,0), thickness=2)
            image = cv2.putText(
                img=np.copy(image), text=f'last action: {last_action}', 
                org=(int(0.15 * self.width), int(0.97 * self.height)), fontFace=3, 
                fontScale=1, color=(0,0,0), thickness=2)
        if render_text:
            text_img = self.put_multiline_text(text, image.shape[0], image.shape[1])
            image = np.concatenate([image, text_img], 1)
        return image

    def render_screenshots(self, writefile):
        dirs, _ = os.path.split(writefile)
        if not os.path.exists(dirs):
            os.makedirs(dirs, exist_ok=True)
        for i, data in enumerate(tqdm(self.data_sequence)):
            pos, grid = self.parse_init_conds(data)
            self.env.unwrapped.start_position = pos
            self.env.unwrapped.initial_blocks = grid
            self.env.should_reset(True)
            self.env.reset()
            for _ in range(20):
                self.env.step(self.env.action_space.no_op())
            obs, reward, done, info = self.env.step(self.env.action_space.no_op())
            Image.fromarray(obs['pov']).save(f'{writefile}{i}.jpg')
