from pylab import subplots


class AccuracyPlot(object):
    def set_labels(self):
        self.ax2.set_xlabel('iteration')
        # ax1.set_ylabel('train loss')
        self.ax2.set_ylabel('test accuracy')
        # ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
        self.ax2.set_title('Test Mean Accuracy: {:.2f}'.format(np.mean(self.points)))

    def __init__(self, points, output_plot_path, test_interval=1):
        # Generates plot
        self.points = points
        self.fig, self.ax2 = subplots()
        self.ax2.plot(test_interval * range(len(points)), points, 'r')
        self.set_labels()
        # Saves plot
        self.fig.savefig(output_plot_path)
