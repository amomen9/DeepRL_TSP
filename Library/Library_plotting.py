"""
Library_plotting.py - Plotting classes for agent behavior and learning curves.

Contents
--------
TSPAgentPlot     - Per-episode trajectory plot + animation helpers.
LearningCurvePlot - Mean / CI learning-curve figure.
"""
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pyplot as visplt
import torch
import torch.nn as nn


# Begin Class TSPAgentPlot ##############################################################
class TSPAgentPlot:
    ''' Class for plotting TSP agent behavior during training '''
    def __init__(self, env, title=None, curve_plot=False):
        self.env = env
        if curve_plot:
            self.fig,self.ax = visplt.subplots()
            ###### Setting plot parameters for better visualization
            self.ax.set_xlim(-2.4,2.4)
            self.ax.set_ylim(-0.5,0.5)
            self.ax.set_xlabel('Cart Position')
            self.ax.set_ylabel('Pole Angle (radians)')
            visplt.rcParams['figure.figsize'] = (8,6) # set default figure size
            visplt.rcParams['animation.embed_limit'] = (1000*1000)*10*10  # 100 MB
            mpl.rcParamsDefault.update(visplt.rcParams)
            visplt.rc('font', size=14)
            visplt.rc('axes', labelsize=14, titlesize=14)
            visplt.rc('legend', fontsize=14)
            visplt.rc('xtick', labelsize=10)
            visplt.rc('ytick', labelsize=10)
            visplt.rc('animation', html='jshtml')
        ##################################################################################################################
            if title is not None:
                self.ax.set_title(title)

    def add_episode(self,obs_history,label=None, ls="solid"):
        ''' obs_history: list of observations during episode (list of 4D vectors) '''
        cart_positions = [obs[0] for obs in obs_history]
        pole_angles = [obs[2] for obs in obs_history]
        if label is not None:
            self.ax.plot(cart_positions,pole_angles,label=label, ls=ls)
        else:
            self.ax.plot(cart_positions,pole_angles, ls=ls)


    def plot_environment(self, env, figsize=(5, 4)):
        visplt.figure(figsize=figsize)
        img = env.render()
        visplt.imshow(img)
        visplt.axis("off")
        return img

    # extra code – this cell displays an animation of one episode

    def update_scene(self, num, frames, patch):
        patch.set_data(frames[num])
        return patch,

    def plot_animation(self, frames, repeat=False, interval=40):
        fig = visplt.figure()
        patch = visplt.imshow(frames[0])
        visplt.axis('off')
        visplt.tight_layout(pad=0)

        # Use interactive-mode frame stepping instead of FuncAnimation so that
        # every single frame is guaranteed to be drawn on all backends (TkAgg
        # on Windows drops frames when timer-driven redraws can't keep up).
        pause_sec = interval / 1000.0
        visplt.ion()
        try:
            while True:
                for frame in frames:
                    if not visplt.fignum_exists(fig.number):
                        print(f"\nNumber of frames: {len(frames)}")
                        return fig
                    patch.set_data(frame)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    visplt.pause(pause_sec)
                if not repeat:
                    break
        finally:
            visplt.ioff()

        self._last_anim = fig          # keep figure alive
        print(f"Number of frames: {len(frames)}")
        return fig



    @staticmethod
    def _resolve_action(policy, obs):
        # Support passing a torch.nn.Module directly as policy.
        if isinstance(policy, nn.Module):
            with torch.no_grad():
                state = torch.as_tensor(obs, dtype=torch.float32)
                logit = policy(state)
                prob = torch.sigmoid(logit).item()
            return int(prob >= 0.5)

        action = policy(obs)
        if isinstance(action, tuple):
            action = action[0]
        if torch.is_tensor(action):
            action = action.item()
        return int(action)
# End Class TSPAgentPlot ##############################################################


# Begin Class LearningCurvePlot ##############################################################
class LearningCurvePlot:

    def __init__(self, title=None, title_fontsize=14, summary_text=None, summary_fontsize=10):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Timestep')
        self.ax.set_ylabel('Episode Return')
        self._title_fontsize = title_fontsize
        self._summary_fontsize = summary_fontsize
        if title is not None:
            self.fig.suptitle(title, fontsize=title_fontsize, y=0.985)
        if summary_text is not None:
            self.fig.text(
                0.5,
                0.945,
                summary_text,
                ha="center",
                va="top",
                fontsize=summary_fontsize,
            )

    def add_curve(self, x, y, label=None, ls="solid", color=None):
        ''' y: vector of average reward results
        label: string to appear as label in plot legend '''
        if label is not None and color is not None:
            self.ax.plot(x, y, label=label, linestyle=ls, color=color)
        elif label is not None:
            self.ax.plot(x, y, label=label, linestyle=ls)
        elif color is not None:
            self.ax.plot(x, y, linestyle=ls, color=color)
        else:
            self.ax.plot(x, y, linestyle=ls)

    def add_shaded_ci(self, x, y_mean, y_std, n, alpha=0.2, fill_opacity=0.15, y_lower_cap=None, y_upper_cap=None, color=None):
        '''Add a shaded confidence band around the mean curve.
        alpha controls CI significance (e.g., 0.05 for 95% CI),
        fill_opacity controls the visual transparency of the shaded area.'''
        from scipy.stats import t as t_dist

        x_arr = np.asarray(x, dtype=np.float32)
        y_mean_arr = np.asarray(y_mean, dtype=np.float32)
        y_std_arr = np.asarray(y_std, dtype=np.float32)
        t_crit = t_dist.ppf(1 - alpha / 2, df=max(n - 1, 1))
        margin = t_crit * y_std_arr / np.sqrt(max(n, 1))
        y_lower = y_mean_arr - margin
        y_upper = y_mean_arr + margin
        if y_lower_cap is not None:
            y_lower = np.maximum(y_lower, y_lower_cap)
        if y_upper_cap is not None:
            y_upper = np.minimum(y_upper, y_upper_cap)
        if color is None:
            color = self.ax.get_lines()[-1].get_color()  # match the last plotted line
        self.ax.fill_between(x_arr, y_lower, y_upper,
                             alpha=fill_opacity, color=color)
        # Return the (capped) band so callers can fold its extent into the
        # y-limits — otherwise the shaded band can rise above the mean curve's
        # max and get hidden under the legend (notably on smoothed plots, where
        # the mean is a clean plateau but the band still extends past it).
        return y_lower, y_upper

    def set_ylim(self,lower,upper):
        self.ax.set_ylim(lower, upper)

    def add_hline(self,height,label):
        self.ax.axhline(height,ls='--',c='k',label=label)

    def _reserve_legend_headroom(self, legend, upper_gap_frac=1.0 / 6.0, lower_margin_frac=1.0 / 4.0):
        '''Apply the display margins around the tracked data extent.

        On entry the axes y-limits are the *exact* tracked data extent
        ``(y_min, y_max)`` (mean curves + CI bands + benchmark). This method
        expands them to:
          * a lower margin of ``lower_margin_frac`` * range below ``y_min``;
          * an upper gap of ``upper_gap_frac`` * range between ``y_max`` and the
            **bottom of the legend box**, so the highest curve/band sits just
            below the top-anchored legend and nothing hides under it.

        The legend is anchored in axes-fraction coordinates, so its bottom edge
        stays at a fixed fraction of the axes regardless of the y-limits. We read
        that fraction once and solve for the top limit that puts the legend box
        bottom exactly ``upper_gap_frac`` * range above ``y_max``.'''
        if legend is None:
            return
        y_min, y_max = self.ax.get_ylim()
        data_range = y_max - y_min
        if not np.isfinite(data_range) or data_range <= 0:
            return
        try:
            self.fig.canvas.draw()  # realise renderer + final axes/legend geometry
            renderer = self.fig.canvas.get_renderer()
            leg_bbox = legend.get_window_extent(renderer)
            ax_bbox = self.ax.get_window_extent(renderer)
        except Exception:
            return
        if ax_bbox.height <= 0:
            return
        # Legend box bottom as a fraction of the axes height (0 = axes bottom,
        # 1 = axes top). Clamp so an oversized legend can't blow up the solve.
        leg_bottom_frac = (leg_bbox.y0 - ax_bbox.y0) / ax_bbox.height
        leg_bottom_frac = min(max(leg_bottom_frac, 0.15), 0.999)

        y_lower = y_min - lower_margin_frac * data_range
        legend_bottom_data = y_max + upper_gap_frac * data_range
        # Map legend_bottom_data -> leg_bottom_frac in the new limits:
        #   (legend_bottom_data - y_lower) / (y_upper - y_lower) = leg_bottom_frac
        y_upper = y_lower + (legend_bottom_data - y_lower) / leg_bottom_frac
        self.ax.set_ylim(y_lower, y_upper)

    def save(self, name='test.png', out_dir="plots"):
        ''' name: string for filename of saved figure
            out_dir: directory to save into when ``name`` is not absolute
                     (e.g. "Trial Continuation Analysis" in checkpoint-reuse mode) '''
        legend = self.ax.legend(
            loc="upper center",
            fontsize=8,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.3,
        )
        self.fig.tight_layout(rect=(0, 0, 1, 0.90))
        self._reserve_legend_headroom(legend)
        output_path = name
        if not os.path.isabs(name):
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, os.path.basename(name))

        from .Helper_progress_bar import get_unique_filepath
        output_path = get_unique_filepath(output_path)
        self.fig.savefig(output_path, dpi=300)
        return output_path
# End Class LearningCurvePlot ##############################################################
