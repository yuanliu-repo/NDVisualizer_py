import os, sys

def get_resource_path(relative_path):
        """
        Get absolute path to resource, works for dev and for Nuitka onefile.
        """
        # if hasattr(sys, '_MEIPASS'):
        #     # PyInstaller
        #     return os.path.join(sys._MEIPASS, relative_path)
        # elif hasattr(sys, 'frozen') and hasattr(sys, 'executable'):
        #     # Nuitka
        #     return os.path.join(os.path.dirname(sys.executable), relative_path)
        # else:
        #     # Development
        #     return os.path.join(os.path.dirname(__file__), relative_path)
        return os.path.join(os.path.dirname(__file__), relative_path)

def get_icon_path():
    """
    Get the path to the icon.ico file, whether running standalone or in development.
    """
    return get_resource_path('icon.ico')
    