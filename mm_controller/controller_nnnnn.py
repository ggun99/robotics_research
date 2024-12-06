import numpy as np
import matplotlib.pyplot as plt

class Player:
    def __init__(self, position):
        self.position = np.array(position)
        self.history = []

    def move(self, new_position):
        self.position = np.array(new_position)
        self.history.append(self.position)

class CubicBezierPath:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

    def evaluate(self, t):
        return (1-t)**3 * self.p0 + 3*(1-t)**2*t * self.p1 + 3*(1-t)*t**2 * self.p2 + t**3 * self.p3

class Game:
    def __init__(self, cur_position, goal_position, time_step, total_time):
        self.player = Player([0, 0])
        self.path = CubicBezierPath(cur_position, [goal_position[0]/2, 0], [goal_position[0]*5/4, goal_position[1]/2], goal_position)
        self.t_values = np.linspace(0, 1, time_step)
        self.d_time = total_time/time_step
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-np.abs(goal_position[0])-1, np.abs(goal_position[0])+2)
        self.ax.set_ylim(-np.abs(goal_position[1])-1, np.abs(goal_position[1])+2)
        self.ax.set_title('Player Movement Along Cubic Bezier Path')
        self.ax.set_aspect('equal')
        self.line, = self.ax.plot([], [], lw=2, label='Player Path')
        self.player_marker, = self.ax.plot([], [], 'ro', markersize=10, label='Player')
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.animation_running = False
        self.animation = None
        self.ax.legend()
        plt.show()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if not self.animation_running:
            self.animate_player_movement()

    def animate_player_movement(self):
        self.animation_running = True
        self.animation = self.animate()

    def animate(self):
        for t in self.t_values:
            new_position = self.path.evaluate(t)
            self.player.move(new_position)
            self.update_plot()
            plt.pause(0.05)
        self.animation_running = False

    def update_plot(self):
        self.line.set_data([p[0] for p in self.player.history], [p[1] for p in self.player.history])
        self.player_marker.set_data(self.player.position[0], self.player.position[1])
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    cur_position = [0,0]
    goal_position = [-6,-7]
    game = Game(cur_position, goal_position, 100, 2)