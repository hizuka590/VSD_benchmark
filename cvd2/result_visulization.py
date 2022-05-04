
import os
import os.path as osp
from utils.visualization import visualize_depth_dir

# Change based on your output path.
# output_dir = "baby_wave1_run_output"
# output_dir = "book2_run_output"
# output_dir = "crossing_run_output"
# output_dir = "hand-shake_run_output"
output_dir = "walk_dog_run_output"


depth_midas_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir, "depth_midas2/depth")
depth_vis_midas_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir, "depth_vis_midas2")
os.makedirs(depth_vis_midas_dir, exist_ok=True)
visualize_depth_dir(depth_midas_dir, depth_vis_midas_dir)

# depth_result_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir, "R0-30_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth/")
depth_result_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir, "R_hierarchical2_midas2/StD100.0_StR1.0_SmD0_SmR0.0/depth_e0000/e0000_filtered/depth/")

depth_vis_result_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir,"depth_vis_result")
os.makedirs(depth_vis_result_dir, exist_ok=True)
visualize_depth_dir(depth_result_dir, depth_vis_result_dir)

import glob
import moviepy as mvp
from moviepy.editor import *

fps = 30

color_dir = osp.join("/opt/sdb/polyu/VSD_benchmark/cvd2", output_dir, "color_down_png")
clip_color = ImageSequenceClip(color_dir, fps=fps)
clip_midas = ImageSequenceClip(depth_vis_midas_dir, fps=fps)
clip_result = ImageSequenceClip(depth_vis_result_dir, fps=fps)

clip_color = clip_color.set_duration(clip_result.duration)
clip_midas = clip_midas.set_duration(clip_result.duration)

clip_color.write_videofile('clip_color.mp4', fps=fps)
clip_midas.write_videofile('clip_midas.mp4', fps=fps)
clip_result.write_videofile('clip_result.mp4', fps=fps)

video_color = VideoFileClip('clip_color.mp4')
video_midas = VideoFileClip('clip_midas.mp4')
video_result = VideoFileClip('clip_result.mp4')

video = clips_array([[video_color, video_midas, video_result]])
video.write_videofile('video_comparison.mp4', fps=24, codec='mpeg4')

ipython_display(video, autoplay=1, loop=1)