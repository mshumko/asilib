class Plugin:
    def __init__(self, location, instument=None):
        self.location = location
        self._check_location()
        print('in asilib_themis.plugin.Plugin().__init__')
    
    def download_img(self, time):
        print('in asilib_themis.plugin.Plugin().download_img()')

    def download_cal(self, time):
        print('in asilib_themis.plugin.Plugin().download_cal()')

    def load_img(self, time):
        print('in asilib_themis.plugin.Plugin().load_img()')

    def load_cal(self, time):
        print('in asilib_themis.plugin.Plugin().load_cal()')

    def _check_location(self):

        return