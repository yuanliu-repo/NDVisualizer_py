from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QLabel, QWidget, QFileDialog, QMessageBox, QGroupBox, QSplitter, QSizePolicy, QListWidgetItem
)
from PySide6.QtGui import QIcon  # Import QIcon
from PySide6.QtCore import Qt
import sys
from .NDVisualizer_PySide6 import NDVisualizer, trim_singular_list_2
import os
from datetime import datetime
from .utils import get_icon_path
from ._version import __version__


class MatFileSelector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDVisualizer File Selector")
        self.setMinimumSize(400, 400)
        self.setFixedSize(self.minimumSize())  # Set the window size to the minimum size

        # Set the window icon
        self.setWindowIcon(QIcon(get_icon_path()))  # Ensure icon.ico is in the same directory as the script

        # Enable drag-and-drop for the entire window
        self.setAcceptDrops(True)

        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # File selection layout
        file_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_button.setStyleSheet("padding: 8px;")  # Add text margins (padding) to the button
        button_layout.addWidget(self.file_button)
        # button_layout.addStretch()  # Add stretch to push the button to the left

        # File save as button
        self.file_save_as_button = QPushButton("Save As")
        self.file_save_as_button.clicked.connect(self.save_file_as)
        self.file_save_as_button.setStyleSheet("padding: 8px;")  # Add text margins (padding) to the button
        # self.file_save_as_button.setEnabled(False)  # Initially disabled until a file is loaded
        self.file_save_as_button.setToolTip("Save the currently loaded file to a .mat file")
        button_layout.addWidget(self.file_save_as_button)
        button_layout.addStretch()  # Add stretch to push the button to the right
        file_layout.addLayout(button_layout)

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: gray; font-size: 12px;")  # Optional: Style the label
        self.file_info_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Fix the height
        file_layout.addWidget(self.file_info_label)
        self.layout.addLayout(file_layout)

        # Splitter for Keys and Attributes
        self.splitter = QSplitter()
        self.splitter.setHandleWidth(0)  # Disable the handle for dragging
        self.splitter.setChildrenCollapsible(False)  # Prevent collapsing of child widgets
        self.layout.addWidget(self.splitter)

        # Keys Group Box
        keys_group = QGroupBox("Datasets")
        keys_layout = QVBoxLayout()
        self.keys_list = QListWidget()
        self.keys_list.itemSelectionChanged.connect(self.update_attributes)
        keys_layout.addWidget(self.keys_list)
        keys_group.setLayout(keys_layout)
        self.splitter.addWidget(keys_group)

        # Attributes Group Box
        attributes_group = QGroupBox("Attributes")
        attributes_layout = QVBoxLayout()
        self.attributes_list = QListWidget()
        attributes_layout.addWidget(self.attributes_list)
        attributes_group.setLayout(attributes_layout)
        self.splitter.addWidget(attributes_group)

        # Set initial splitter sizes to 50% ratio
        self.splitter.setSizes([self.width() // 2, self.width() // 2])

        # Launch NDVisualizer button
        self.launch_button = QPushButton("Launch NDVisualizer")
        self.launch_button.clicked.connect(self.launch_visualizer)
        self.launch_button.setEnabled(False)  # Disabled until a valid selection is made
        self.launch_button.setStyleSheet("padding: 8px;")  # Add text margins (padding) to the button
        
        
        # Add a clickable GitHub hyperlink
        link_label = QLabel()
        link_label.setTextFormat(Qt.RichText)
        link_label.setStyleSheet("color: gray;")
        link_label.setText(f'v{__version__}  <a href="https://github.com/yuanliu-repo/NDVisualizer_py" style="color: gray;">GitHub Page</a>')
        link_label.setOpenExternalLinks(True)

        launch_button_layout = QHBoxLayout()
        launch_button_layout.addWidget(link_label, alignment=Qt.AlignLeft)
        launch_button_layout.addStretch()
        launch_button_layout.addWidget(self.launch_button, alignment=Qt.AlignRight)

        self.layout.addLayout(launch_button_layout)
        

        # MAT file data
        self.mat_data = None
        self.selected_key = None



    def resizeEvent(self, event):
        """Override resizeEvent to maintain a 50% split ratio."""
        super().resizeEvent(event)
        total_width = self.splitter.width()
        self.splitter.setSizes([total_width // 2, total_width // 2])

    def dragEnterEvent(self, event):
        """Handle drag enter events to validate the dragged file."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1 and urls[0].toLocalFile().endswith((".mat", ".csv")):
                event.acceptProposedAction()  # Accept the drag event
                return
        event.ignore()

    def dropEvent(self, event):
        """Handle drop events to load the dropped .mat file."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                mat_filename = urls[0].toLocalFile()
                if mat_filename.endswith((".mat", ".csv")):
                    self.load_data(mat_filename)
                    event.acceptProposedAction()  # Accept the drop event
                    return
        event.ignore()

    def load_data(self, mat_data):
        """Load the .mat file and populate the datasets list."""
        

        try:
            self.mat_data = trim_singular_list_2(mat_data)[0]
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load .mat file: {e}")
            return

        # Filter keys
        keys = []
        DATASET_KEYS = ['dataset', 'Lumerical_dataset']
        for key in self.mat_data.keys():
            if key[0:2] != "__" and isinstance(self.mat_data[key][0], dict) and any(dataset_key in self.mat_data[key][0] for dataset_key in DATASET_KEYS):
                keys.append(key)

        if not keys:
            QMessageBox.warning(self, "No Valid Dataset", "No valid datasets found in the .mat file.")
            return

        # Sort the keys
        keys.sort()
        # Populate the keys list
        self.keys_list.clear()
        self.keys_list.addItems(keys)
        self.selected_key = None
        self.attributes_list.clear()
        self.launch_button.setEnabled(False)

    def select_file(self):
        """Open a file dialog to select a .mat or .csv file."""
        mat_filename, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "All Supported (*.csv *.mat);;MAT Files (*.mat);;CSV Files (*.csv)"
        )

        if not mat_filename:
            # QMessageBox.warning(self, "No File Selected", "Please select a valid .mat or .csv file.")
            return

        if mat_filename.endswith('.csv'):
            from .CSV_dlg import load_csv_file
            try:
                data = load_csv_file(mat_filename)
                self.load_data(data)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading .csv file: \n{e}")

        elif mat_filename.endswith('.mat'):
            from pymatreader import read_mat
            try:
                self.load_data(read_mat(mat_filename))
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading .mat file: \n{e}")

        # Display file name and file date
        file_date = datetime.fromtimestamp(os.path.getmtime(mat_filename)).strftime('%Y-%m-%d %H:%M:%S')
        self.file_info_label.setText(f"File: {os.path.basename(mat_filename)} | Date: {file_date}")

    def save_file_as(self):
        """Open a file dialog to save the currently loaded .mat file."""
        if not self.mat_data:
            QMessageBox.warning(self, "No Data", "No data loaded to save.")
            return

        mat_filename, _ = QFileDialog.getSaveFileName(
            self, "Save File As", "", "MAT Files (*.mat)"
        )

        if not mat_filename:
            return
        
        from scipy.io import savemat
        try:
            savemat(mat_filename, self.mat_data)
            QMessageBox.information(self, "Save Successful", f"File saved as {mat_filename}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file: {e}")   

    def update_attributes(self):
        """Update the attributes list based on the selected key."""
        selected_items = self.keys_list.selectedItems()
        if not selected_items:
            self.selected_key = None
            self.attributes_list.clear()
            self.launch_button.setEnabled(False)
            return

        self.selected_key = selected_items[0].text()
        dataset = self.mat_data[self.selected_key][0]

        # Determine the dataset field
        if 'dataset' in dataset:
            attributes = dataset['dataset'][0]['attributes'][0]
        elif 'Lumerical_dataset' in dataset:
            attributes = dataset['Lumerical_dataset'][0]['attributes'][0]
        else:
            attributes = {}

        # Populate the attributes list
        self.attributes_list.clear()
        if isinstance(attributes, dict) and 'name' in attributes and 'variable' in attributes:
            for name, variable in zip(attributes['name'], attributes['variable']):
                item = QListWidgetItem(f"{name}")  # Display text
                item.setData(Qt.UserRole, variable)  # Store the variable as custom data
                self.attributes_list.addItem(item)

        # Enable the launch button if attributes are available
        self.launch_button.setEnabled(self.attributes_list.count() > 0)

    def launch_visualizer(self):
        """Launch the NDVisualizer for the selected key and attribute."""
        selected_attribute_items = self.attributes_list.selectedItems()
        if not self.selected_key or not selected_attribute_items:
            QMessageBox.warning(self, "No Selection", "Please select a key and an attribute.")
            return

        selected_item = selected_attribute_items[0]
        attribute_key = selected_item.data(Qt.UserRole)  # Retrieve the stored variable

        if not attribute_key:
            QMessageBox.warning(self, "Invalid Attribute", "The selected attribute is not valid.")
            return

        # Launch the NDVisualizer
        self.visualizer = NDVisualizer(self.mat_data[self.selected_key][0], attribute_key)
        self.visualizer.show()


def main():
    app = QApplication(sys.argv)
    selector = MatFileSelector()

    # Check if a file path is provided as a command-line argument
    if len(sys.argv) > 1:
        mat_file_path = sys.argv[1]
        if os.path.isfile(mat_file_path) and mat_file_path.endswith(".mat"):
            try:
                selector.load_data(mat_file_path)
            except Exception as e:
                print(f"Error loading file: {e}")
        else:
            print("Invalid file path or file is not a .mat file.")
    

    import os
    import tempfile
    # Use this code to signal the splash screen removal.
    if "NUITKA_ONEFILE_PARENT" in os.environ:
        splash_filename = os.path.join(
            tempfile.gettempdir(),
            "onefile_%d_splash_feedback.tmp" % int(os.environ["NUITKA_ONEFILE_PARENT"]),
        )

        if os.path.exists(splash_filename):
            os.unlink(splash_filename)

    selector.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()