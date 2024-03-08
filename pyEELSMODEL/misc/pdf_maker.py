from fpdf import FPDF
import time
import os


class PDF(FPDF):
    """
    Class which makes a pdf, this is useful when wanting to have some
    information used on a procedure you applied to the data. For instance,
    the FastAlignZeroLoss class can generate a file with the relevant
    information which can be checked if everything went fine.
    """

    def get_time_name(self):
        return time.ctime(time.time()).replace(" ", "_").replace(":", "_")

    def get_savename(self):
        name = self.get_time_name() + '_shift_overview.pdf'
        return os.path.join(self.savepath, name)

    def get_figname(self):
        name = self.get_time_name() + '_img' + str(self.counter) + '.png'
        self.counter += 1
        return os.path.join(self.savepath, name)

    def add_figure(self, fig):
        savename = self.get_figname()
        fig.savefig(savename, bbox_inches='tight', dpi=300)
        wd = fig.get_figwidth()
        he = fig.get_figheight()
        self.set_xy(0.0, self.get_y() + self.dh)

        self.image(savename, link='', type='', w=wd, h=he)
        os.remove(savename)

    def intitialize(self, path):
        self.set_font('Arial', size=12)
        self.counter = 0
        self.dh = 0.2
        self.savepath = path
