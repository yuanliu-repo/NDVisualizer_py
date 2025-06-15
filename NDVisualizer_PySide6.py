import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QMessageBox,
    QGridLayout, QLabel, QComboBox, QPushButton, QSlider, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon  # Import QIcon

from superqt import QRangeSlider

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from pymatreader import read_mat
from PySide6.QtWidgets import QFileDialog


# Patch hdf5storage to replace np.unicode_ with np.str_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

class MatlabDict(dict):

    def process_item(self,  value):
        if isinstance(value, list):
            if len(value) == 1:
                return self.process_item(value[0])
            else:
                return [MatlabDict(val) if isinstance(val, dict) else val for val in value]
        elif isinstance(value, dict):
            return MatlabDict(value)  # Convert nested dictionaries to TrimmedDict
        return value
    
    def __getitem__(self, key):
        value = super().__getitem__(key)
        return self.process_item(value)

def trim_singular_list_2(data):
    """
    Recursively trim all lists with only one element in a nested dictionary or list.
    Also converts numpy string arrays to Python strings with strip() and wraps in numpy arrays with dtype=object.
    """
    if isinstance(data, dict):
        return [{key: trim_singular_list_2(value) for key, value in data.items()}]
    elif isinstance(data, list):
        if len(data) == 1:
            return trim_singular_list_2(data[0])
        else:
            return [trim_singular_list_2(item)[0] if len(trim_singular_list_2(item)) > 0 else trim_singular_list_2(item) for item in data]
    elif isinstance(data, np.ndarray):
        # Check if it's a string array and convert to Python strings with strip()
        if data.dtype.kind in ['U', 'S']:  # Unicode or byte string
            if data.ndim == 0:  # scalar string
                return np.array(str(data.item()).strip(), dtype=object)
            else:  # array of strings
                return np.array([str(item).strip() for item in data.flat], dtype=object)
        else:
            return data
    elif isinstance(data, (np.str_, np.unicode_)):
        return np.array(str(data).strip(), dtype=object)
    elif not isinstance(data, np.ndarray):
        return [data]
    else:
        return data

def trim_singular_list(data):
    """
    Recursively trim all lists with only one element in a nested dictionary or list.
    """
    if isinstance(data, dict):
        return {key: trim_singular_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        if len(data) == 1:
            return trim_singular_list(data[0])
        else:
            return [trim_singular_list(item) for item in data]
    else:
        return data


def squeeze_all_fields(data):
    """
    Recursively squeeze all NumPy array values in a nested dictionary or list.
    """
    if isinstance(data, dict):
        return {key: squeeze_all_fields(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [squeeze_all_fields(item) for item in data]
    elif isinstance(data, np.ndarray):
        return np.squeeze(data)
    else:
        return data


class QSlider_helper(QRangeSlider):
    def setValue(self, value):
        """
        Override setValue to handle both single values and tuples.
        Single values are converted to single-element tuples.
        """
        if isinstance(value, (int, float)):
            value = (value,)
        elif isinstance(value, (list, tuple)):
            value = tuple(value)
        super().setValue(value)


class PlotWindow(QMainWindow):
    def __init__(self, title):
        super().__init__()
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        # Set the window icon
        self.setWindowIcon(QIcon("icon.ico"))  # Ensure icon.ico is in the same directory as the script

        # Create a QWidget to act as the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a layout for the central widget
        self.layout = QVBoxLayout(self.central_widget)

        # Initialize the matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

    def clear_plot(self):
        # Clear the matplotlib figure
        self.figure.clear()

    def plot_1d(self, xvals, yvals, xlabel, ylabel, title):
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        ax.plot(xvals, yvals, marker='o')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_2d(self, xvals, yvals, data, xlabel, ylabel, title):
        self.plot_2d_pcolormesh(xvals, yvals, data, xlabel, ylabel, title)

    def plot_2d_imshow(self, data, extent, xlabel, ylabel, title):
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        cax = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.figure.colorbar(cax, ax=ax)

        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_2d_pcolormesh(self, xvals, yvals, data, xlabel, ylabel, title):
        self.clear_plot()
        ax = self.figure.add_subplot(111)

        # Create a meshgrid from xvals and yvals
        X, Y = np.meshgrid(xvals, yvals, indexing='ij')

        # Plot the data using pcolormesh
        cax = ax.pcolormesh(X, Y, data, shading='auto', cmap='viridis')

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Add a colorbar
        self.figure.colorbar(cax, ax=ax)

        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_2d_contour(self, xvals, yvals, data, xlabel, ylabel, title):
        self.clear_plot()
        ax = self.figure.add_subplot(111)

        # Create a meshgrid from xvals and yvals
        X, Y = np.meshgrid(xvals, yvals)

        # Plot the data using contourf
        cax = ax.contourf(X, Y, data, levels=50, cmap='viridis')

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Add a colorbar
        self.figure.colorbar(cax, ax=ax)

        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_3d(self, xvals, yvals, zvals, vals, xlabel, ylabel, zlabel, title):
        self.clear_plot()
        ax = self.figure.add_subplot(111, projection='3d')
        scatter = ax.scatter(xvals, yvals, zvals, c=vals, cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        self.figure.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()


class NDVisualizer(QWidget):
    def __init__(self, mat_data, attribute_key):
        """
        mat_data: dictionary loaded from the .mat file using pymatreader.read_mat.
        attribute_key: the key of the attribute to visualize.
        Expected structure:
            {
              'dataset': {
                  'parameters': [
                        {'variable': [variable name(s)], 'name': [display name(s)]}, ...
                  ],
                  'attributes': [
                        {'variable': [attribute name(s)], 'name': [display attribute name(s)]}
                  ],
              },
              '[attribute name]': [attribute data],
              '[variable name]': [variable data]
            }
        """
        super().__init__()
        # Set the window icon
        self.setWindowIcon(QIcon("icon.ico"))  # Ensure icon.ico is in the same directory as the script

        # Save the full .mat data for later lookups
        self.mat_data = trim_singular_list_2(mat_data)[0]

        # Determine the dataset field
        if 'dataset' in self.mat_data:
            self.ds_field = 'dataset'
        elif 'Lumerical_dataset' in self.mat_data:
            self.ds_field = 'Lumerical_dataset'
        else:
            raise ValueError("Dataset format not recognized.")
        
        # Use the provided attribute_key to extract the data
        if attribute_key not in self.mat_data:
            raise ValueError(f"Attribute key '{attribute_key}' not found in the provided data.")
        
        self.attribute_key = attribute_key
        
        # Get the dataset structure
        self.dataset = self.mat_data[self.ds_field][0]



        # Check for Lumerical rectilinear datasets
        if 'geometry' in self.dataset and self.dataset['geometry'][0] == 'rectilinear':
            for dim, label in zip(['z', 'y', 'x'], ['Z', 'Y', 'X']):  # Reverse order to insert z, y, x at the top
                if dim in self.mat_data:
                    self.dataset['parameters'].insert(0, {  # Add to the top of the parameter list
                        'variable': [dim],
                        'name': [label]
                    })

        # Number of parameters (dimensions)
        self.nd = len(self.dataset['parameters'])


        # Build size list (for each parameter, look up the corresponding variable array)
        self.sz = []
        for d in range(self.nd):
            var_key = self.dataset['parameters'][d]['variable'][0]  # Use the first variable if it's a list
            self.sz.append(len(self.mat_data[var_key]))


        # self.data = np.squeeze(np.array(self.mat_data[attribute_key]))
        # self.data = self.data.reshape(self.sz)

        # Create labels for each dimension (using the 'name' field)
        self.dimLabels = [self.dataset['parameters'][d]['name'][0] for d in range(self.nd)]

        # Default plot type: 1D if only one dimension, 2D if >=2, 3D if >=3.
        self.plotType = min(2, self.nd)

        # Operator labels and functions:
        self.operatorLabels = ["real", "imag", "abs", "abs^2", "angle", "unwrapped angle", "10log10", "20log10"]
        self.operatorFunctions = [
            np.real,
            np.imag,
            np.abs,
            lambda x: np.abs(x)**2,
            np.angle,
            lambda x: np.unwrap(np.angle(x)),
            lambda x: 10 * np.log10(np.abs(x)),
            lambda x: 20 * np.log10(np.abs(x))
        ]
        self.operator = 0  # default operator (0-indexed)

        # Default axis assignments (0-indexed)
        self.dimX = 0
        self.dimY = 1 if self.nd > 1 else 0
        self.dimZ = 2 if self.nd > 2 else 0

        # Initialize slice indices for dimensions not on the axes
        self.sliceIndex = [(0, )] * self.nd

        # Initialize parameter indices for each dimension
        self.parameterIndex = [0] * self.nd  # Default to the first parameter for each dimension

        # Access the system clipboard
        self.clipboard = QApplication.clipboard()

        
        self.updateData(0)  # Initialize the data for the first component
        # Set up the control GUI
        self.initUI()
        # Create a separate plot window
        title_attr = self.get_attr_name_by_var(self.attribute_key)
        self.plotWindow = PlotWindow(title_attr + " Visualizer")
        self.plotWindow.show()
        self.updateWidgetStates()
        self.updatePlot()

    def get_attr_name_by_var(self, variable_name):
        title_attr = next(
            # (attr['name'][0] for attr in self.dataset['attributes'] if attr['variable'][0] == self.attribute_key),
            # self.get_attr_name_by_var(self.attribute_key)  # Fallback to the first attribute's name
            (name for name, variable in zip(self.dataset['attributes'][0]['name'], self.dataset['attributes'][0]['variable']) if variable == variable_name),
            self.attribute_key  # Fallback to the attribute key if no name is found
        )
        
        return title_attr


    def initUI(self):
        self.setWindowTitle(self.get_attr_name_by_var(self.attribute_key) + " Visualizer Controls")
        layout = QVBoxLayout()

        # Grid layout for top-level controls
        grid = QGridLayout()
        row = 0

        # Plot Type selection and buttons
        grid.addWidget(QLabel("Plot Type:"), row, 0)
        self.plotTypeCombo = QComboBox()
        # Only include valid types based on dimensions available
        if self.nd >= 3:
            types = ["1D", "2D", "3D"]
        elif self.nd == 2:
            types = ["1D", "2D"]
        else:
            types = ["1D"]
        self.plotTypeCombo.addItems(types)
        self.plotTypeCombo.setCurrentIndex(self.plotType - 1)
        self.plotTypeCombo.currentIndexChanged.connect(self.onPlotTypeChange)
        grid.addWidget(self.plotTypeCombo, row, 1)

        self.newPlotButton = QPushButton("New Plot")
        self.newPlotButton.clicked.connect(self.onNewPlot)
        grid.addWidget(self.newPlotButton, row, 2)
        self.newPlotButton.hide()

        self.heatmapButton = QPushButton("Heatmap")
        self.heatmapButton.clicked.connect(self.onHeatmap)
        grid.addWidget(self.heatmapButton, row, 3)
        # self.heatmapButton.hide()

        self.addPlotButton = QPushButton("+")
        self.addPlotButton.clicked.connect(self.onAddPlot)
        grid.addWidget(self.addPlotButton, row, 4)
        self.addPlotButton.hide()

        
        row += 1

        # X-Axis selection
        
        grid.addWidget(QLabel("X-Axis:"), row, 0)
        self.xAxisCombo = QComboBox()
        self.xAxisCombo.addItems(self.dimLabels)
        self.xAxisCombo.setCurrentIndex(self.dimX)
        self.xAxisCombo.currentIndexChanged.connect(lambda idx: self.onAxisChange('X', idx))
        grid.addWidget(self.xAxisCombo, row, 1)
        row += 1

        # Y-Axis selection
        
        grid.addWidget(QLabel("Y-Axis:"), row, 0)
        self.yAxisCombo = QComboBox()
        self.yAxisCombo.addItems(self.dimLabels)
        self.yAxisCombo.setCurrentIndex(self.dimY)
        self.yAxisCombo.currentIndexChanged.connect(lambda idx: self.onAxisChange('Y', idx))
        grid.addWidget(self.yAxisCombo, row, 1)
        row += 1

        # Z-Axis selection
        
        grid.addWidget(QLabel("Z-Axis:"), row, 0)
        self.zAxisCombo = QComboBox()
        self.zAxisCombo.addItems(self.dimLabels)
        self.zAxisCombo.setCurrentIndex(self.dimZ)
        self.zAxisCombo.currentIndexChanged.connect(lambda idx: self.onAxisChange('Z', idx))
        grid.addWidget(self.zAxisCombo, row, 1)
        row += 1


        # Support for Lumerical rectilinear datasets for vectors and tensors 
        # Add a drop-down for selecting the vector component if applicable
        if 'geometry' in self.dataset and self.dataset['geometry'][0] == 'rectilinear' and self.mat_data[self.attribute_key].ndim > 1 and self.mat_data[self.attribute_key].shape[1] in [3, 9]:
            grid.addWidget(QLabel("Component:"), row, 0)
            self.componentCombo = QComboBox()
            components = ["X", "Y", "Z"] if self.mat_data[self.attribute_key].shape[1] == 3 else ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
            self.componentCombo.addItems(components)
            self.componentCombo.setCurrentIndex(0)
            self.componentCombo.currentIndexChanged.connect(self.onComponentChange)
            grid.addWidget(self.componentCombo, row, 1)
            row += 1

        # Operator selection
        grid.addWidget(QLabel("Operator:"), row, 0)
        self.operatorCombo = QComboBox()
        self.operatorCombo.addItems(self.operatorLabels)
        self.operatorCombo.setCurrentIndex(self.operator)
        self.operatorCombo.currentIndexChanged.connect(self.onOperatorChange)
        grid.addWidget(self.operatorCombo, row, 1)
        row += 1

        # Add "Copy to Clipboard" button
        self.copyButton = QPushButton("Copy to Clipboard")
        self.copyButton.clicked.connect(self.copyToClipboard)
        grid.addWidget(self.copyButton, row, 0, 1, 2)  # Span across two columns
        self.copyButton.setToolTip("Copy the current data slice to the clipboard.")
        # Add a clickable GitHub hyperlink
        link_label = QLabel()
        link_label.setTextFormat(Qt.RichText)
        link_label.setText('<a href="https://github.com/yuanliu-repo/NDVisualizer-py" style="color: gray;" >GitHub Page</a>')
        link_label.setOpenExternalLinks(True)
        grid.addWidget(link_label, row, 3, 1, 1, Qt.AlignCenter)
        row += 1

        layout.addLayout(grid)

        # Create sliders for slicing (one per dimension)
        sliderGroup = QGroupBox("Slice Controls")
        sliderLayout = QVBoxLayout()
        self.sliders = []
        self.sliderLabels = []
        self.nameCombos = []
        self.overlapCheckboxes = []
        for d in range(self.nd):
            vLayout = QVBoxLayout()

            # Horizontal layout for the name_combo and label
            hLayout = QHBoxLayout()

            # Add a drop-down menu for selecting the parameter name
            name_combo = QComboBox()
            name_combo.addItems(self.dataset['parameters'][d]['name'])
            name_combo.setCurrentIndex(0)  # Default to the first name
            name_combo.currentIndexChanged.connect(lambda idx, d=d: self.onParameterNameChange(d, idx))
            hLayout.addWidget(name_combo)
            self.nameCombos.append(name_combo)

            # Set the width of name_combo to match xAxisCombo
            name_combo.setFixedWidth(self.xAxisCombo.sizeHint().width())

            # Get the variable name for this dimension
            var_key = self.dataset['parameters'][d]['variable'][0]
            init_val = self.mat_data[var_key][0]
            label = QLabel(f" = {init_val:g}")
            hLayout.addWidget(label)


            # Add a checkbox for overlap lines
            overlap_checkbox = QCheckBox("Overlap")
            overlap_checkbox.setChecked(False)
            overlap_checkbox.setEnabled(False) # Initially disabled, enable when appropriate
            overlap_checkbox.stateChanged.connect(lambda state, d=d: self.onOverlapCheckboxChange(d, state))
            hLayout.addStretch(1) # Add stretch to push the checkbox to the right
            hLayout.addWidget(overlap_checkbox)
            self.overlapCheckboxes.append(overlap_checkbox)
            

            # Add the horizontal layout to the vertical layout
            vLayout.addLayout(hLayout)

            slider = QSlider_helper(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(self.sz[d] - 1)
            slider.setValue((0,))
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            slider.valueChanged.connect(lambda value, d=d, label=label: self.onSliderChange(d, value, label))
            vLayout.addWidget(slider)

            self.sliders.append(slider)
            self.sliderLabels.append(label)
            sliderLayout.addLayout(vLayout)
        sliderGroup.setLayout(sliderLayout)
        layout.addWidget(sliderGroup)

        self.setLayout(layout)

        # Set the window size to minimal and disable resizing
        self.setFixedSize(self.minimumSizeHint())
        self.updateWidgetStates()
        self.show()

    def onPlotTypeChange(self, index):
        self.plotType = index + 1
        self.updateWidgetStates()
        self.updatePlot()

    def onNewPlot(self):
        title_attr = self.get_attr_name_by_var(self.attribute_key)
        self.plotWindow = PlotWindow(title_attr + " New Plot")
        self.plotWindow.show()
        self.updatePlot()

    def onHeatmap(self):
        if self.plotType == 2:
            title_attr = self.get_attr_name_by_var(self.attribute_key)
            self.heatmapWindow = PlotWindow(title_attr + " Heatmap")
            self.heatmapWindow.show()
            self.updatePlot(heatmap=True)

    def onAddPlot(self):
        # For simplicity, open a new plot window
        self.onNewPlot()

    def onAxisChange(self, axis, idx):
        if axis == 'X':
            self.dimX = idx
        elif axis == 'Y':
            self.dimY = idx
        elif axis == 'Z':
            self.dimZ = idx

        # Ensure exclusivity among the axes
        axes = [self.dimX, self.dimY, self.dimZ][0:min(self.nd, 3)]
        unique_axes = set(axes)

        if len(unique_axes) < len(axes):  # If there are duplicates
            all_indices = set(range(self.nd))  # All possible indices
            # used_indices = {self.dimX, self.dimY, self.dimZ}
            used_indices = unique_axes
            available_indices = list(all_indices - used_indices)

            if len(available_indices):
                # Resolve duplicates by assigning unused indices
                if axis == 'X':
                    if self.dimX == self.dimY and self.nd > 1:
                        self.dimY = available_indices.pop(0)
                    if self.dimX == self.dimZ and self.nd > 2:
                        self.dimZ = available_indices.pop(0)
                elif axis == 'Y':
                    if self.dimY == self.dimX:
                        self.dimX = available_indices.pop(0)
                    if self.dimY == self.dimZ and self.nd > 2:
                        self.dimZ = available_indices.pop(0)
                elif axis == 'Z':
                    if self.dimZ == self.dimX:
                        self.dimX = available_indices.pop(0)
                    if self.dimZ == self.dimY and self.nd > 1:
                        self.dimY = available_indices.pop(0)

        # Block signals to avoid triggering onAxisChange while updating combo boxes
        self.xAxisCombo.blockSignals(True)
        self.yAxisCombo.blockSignals(True)
        self.zAxisCombo.blockSignals(True)

        # Update combo boxes to reflect the changes
        self.xAxisCombo.setCurrentIndex(self.dimX)
        self.yAxisCombo.setCurrentIndex(self.dimY)
        self.zAxisCombo.setCurrentIndex(self.dimZ)

        # Re-enable signals after updating
        self.xAxisCombo.blockSignals(False)
        self.yAxisCombo.blockSignals(False)
        self.zAxisCombo.blockSignals(False)
        
        self.updateWidgetStates()
        self.updatePlot()

    def onOperatorChange(self, index):
        self.operator = index
        self.updatePlot()

    def onSliderChange(self, d, value, label):
        """
        Update the slice index, label, and plot when the slider value changes.
        """
        self.sliceIndex[d] = value
        var_key = self.dataset['parameters'][d]['variable'][self.parameterIndex[d]]

        label_text = " = " + " - ".join(
            f"{self.mat_data[var_key][v]:g}"
            for v in value
        )
        label.setText(label_text)

        # if len(value) == 1: 
        #     current_value = self.mat_data[var_key][value]
        #     label.setText(f" = {current_value:g}")
        # else:
        #     current_value_left = self.mat_data[var_key][value[0]]
        #     current_value_right = self.mat_data[var_key][value[1]]
        #     label.setText(f" = {current_value_left:g} - {current_value_right:g}")

        self.updatePlot()

    def onParameterNameChange(self, d, idx):
        """
        Update the axis labels and plot when a new parameter name is selected,
        while keeping the current slider index.
        """
        # Update the parameter index for the dimension
        self.parameterIndex[d] = idx

        # Update the dimension label
        self.dimLabels[d] = self.dataset['parameters'][d]['name'][idx]

        # Update the variable key and slider settings
        var_key = self.dataset['parameters'][d]['variable'][idx]
        self.sz[d] = len(self.mat_data[var_key])
        self.sliders[d].setMaximum(self.sz[d] - 1)

        # Keep the current slider value (slice index)
        current_index = self.sliceIndex[d]
        max_val = max(current_index)
        if max_val >= self.sz[d]:
            # Adjust the range to fit within bounds
            if len(current_index) > 1:
                range_size = current_index[1] - current_index[0]
                new_max = self.sz[d] - 1
                new_min = max(0, new_max - range_size)
                current_index = (new_min, new_max)
            else:
                current_index = (min(current_index[0], self.sz[d] - 1),)
            self.sliceIndex[d] = current_index

        self.sliders[d].setValue(current_index)

        # Update the label text
        # current_value = self.mat_data[var_key][current_index]
        # self.sliderLabels[d].setText(f" = {current_value:g}")

        # current_value = [self.mat_data[var_key][idx] for idx in current_index] 
        self.sliderLabels[d].setText(" = " + " - ".join(
            f"{self.mat_data[var_key][idx]:g}"
            for idx in current_index
        ))  

        # if len(current_value) == 1:
        #     self.sliderLabels[d].setText(f" = {current_value[0]:g}")
        # else:
        #     self.sliderLabels[d].setText(f" = {current_value[0]:g} - {current_value[1]:g}")
        
        # Update the axis combo boxes
        self.xAxisCombo.setItemText(d, self.dimLabels[d])
        self.yAxisCombo.setItemText(d, self.dimLabels[d])
        self.zAxisCombo.setItemText(d, self.dimLabels[d])

        # Update the plot
        self.updatePlot()


    def onOverlapCheckboxChange(self, d, state):
        """
        Handles the state change of an overlap checkbox.

        When a checkbox is checked, it signifies that this dimension should be
        "overlapped" or plotted across all its values in a 1D plot, rather than
        being fixed at a single slice. The corresponding slider is disabled.
        Only one dimension can be overlapped at a time.

        Args:
            d (int): The index of the dimension whose overlap checkbox changed.
            state (int): The new state of the checkbox (Qt.CheckState enum value).
        """
        # Determine if the checkbox is now checked
        # Qt.CheckState.Checked.value is typically 2
        is_checked = (state == Qt.CheckState.Checked.value)

        # # Enable/disable the slider for this dimension
        # # If overlap is checked, the slider for this dimension is disabled
        # # as all its values will be shown.
        # self.sliders[d].setEnabled(not is_checked)

        if is_checked:
            # Set the slice index to cover the entire range of this dimension
            self.sliceIndex[d] = (0, self.sz[d] - 1)
            # Update the slider value to reflect the full range
            self.sliders[d].setValue(self.sliceIndex[d])
            # This enforces that only one dimension can be chosen for overlap at a time.
            for i, checkbox in enumerate(self.overlapCheckboxes):
                if i != d and checkbox.isChecked():
                    checkbox.setChecked(False)  # Uncheck other overlap checkboxes
        self.updateWidgetStates()
        # Update the plot to reflect the change in overlap state
        self.updatePlot()


    def updateData(self, index):
        """
        Update the data to use the selected component of the vector attribute.
        """
        # self.data = np.squeeze(np.array(self.mat_data[self.attribute_key]))
        self.data = np.array(self.mat_data[self.attribute_key])
        if 'geometry' in self.dataset and self.dataset['geometry'][0] == 'rectilinear' and self.data.ndim > 1 and self.data.shape[1] in [3, 9]:
            if self.data.shape[1] == 3 and index < 3 or self.data.shape[1] == 9 and index < 9:
                # Select the component based on the index
                self.data = self.data[:, index, ...]  # Select the component based on the index
            else:
                pass
                # self.data = self.data[..., index]  # Select the component based on the index
        
        self.data = self.data.reshape(self.sz, order='F')  # Ensure the data is reshaped correctly
        
    def onComponentChange(self, index):
        """
        Update the data to use the selected component of the vector attribute.
        """
        # Update the data based on the selected component index
        self.updateData(index)

        # Update the plot with the new data
        self.updatePlot()

    def updatePlot(self, heatmap=False):
        """
        Update the plot based on the current data, axes, and operator.
        """
        # Determine which dimensions are used for plotting
        if self.plotType == 1:
            plotDims = [self.dimX]
        elif self.plotType == 2:
            plotDims = [self.dimX, self.dimY]
        elif self.plotType == 3:
            plotDims = [self.dimX, self.dimY, self.dimZ]
        else:
            plotDims = [self.dimX]

        
        allDims = range(self.nd)
        overlapDims = [d for d in allDims if d not in plotDims and self.plotType == 1 and self.overlapCheckboxes[d].isChecked()]

        # Build an index tuple for slicing the data
        idx = [
            # slice(0, self.sz[d]) if d in plotDims or d in overlapDims
            slice(self.sliceIndex[d][0], self.sliceIndex[d][1]+1) if d in plotDims or d in overlapDims
            else slice(self.sliceIndex[d][0], self.sliceIndex[d][0] + 1)
            for d in range(self.nd)
        ]
        dataSlice = self.data[tuple(idx)]

        # Rearrange dimensions so that the plot dimensions come first
        otherDims = [d for d in allDims if d not in plotDims and d not in overlapDims]
        newOrder = plotDims + overlapDims + otherDims
        dataSlice = np.transpose(dataSlice, newOrder)

        # Reshape to just the plot dimensions
        newShape = dataSlice.shape[:len(plotDims + overlapDims)]
        dataSlice = dataSlice.reshape(newShape)

        # Apply the chosen operator
        func = self.operatorFunctions[self.operator]
        dataSlice = func(dataSlice)

        # Clear and update the plot in the plot window
        if self.plotType == 1:
            var_key_x = self.dataset['parameters'][self.dimX]['variable'][self.parameterIndex[self.dimX]]
            xvals = np.squeeze(np.array(self.mat_data[var_key_x][idx[newOrder[0]]]))  # Use the first index for 1D plots
            self.plotWindow.plot_1d(
                xvals, dataSlice,
                xlabel=self.dimLabels[self.dimX],
                ylabel=f"{self.operatorLabels[self.operator]} of {self.get_attr_name_by_var(self.attribute_key)}",
                title=f"{self.operatorLabels[self.operator]} of {self.get_attr_name_by_var(self.attribute_key)}"
            )
        elif self.plotType == 2:
            var_key_x = self.dataset['parameters'][self.dimX]['variable'][self.parameterIndex[self.dimX]]
            var_key_y = self.dataset['parameters'][self.dimY]['variable'][self.parameterIndex[self.dimY]]
            xvals = np.squeeze(np.array(self.mat_data[var_key_x][idx[newOrder[0]]]))
            yvals = np.squeeze(np.array(self.mat_data[var_key_y][idx[newOrder[1]]]))
            self.plotWindow.plot_2d(
                xvals,
                yvals,
                dataSlice,
                xlabel=self.dimLabels[self.dimX],
                ylabel=self.dimLabels[self.dimY],
                title=f"{self.operatorLabels[self.operator]} of {self.get_attr_name_by_var(self.attribute_key)}"
            )
        elif self.plotType == 3:
            var_key_x = self.dataset['parameters'][self.dimX]['variable'][self.parameterIndex[self.dimX]]
            var_key_y = self.dataset['parameters'][self.dimY]['variable'][self.parameterIndex[self.dimY]]
            var_key_z = self.dataset['parameters'][self.dimZ]['variable'][self.parameterIndex[self.dimZ]]
            xvals = np.squeeze(np.array(self.mat_data[var_key_x][idx[newOrder[0]]]))
            yvals = np.squeeze(np.array(self.mat_data[var_key_y][idx[newOrder[1]]]))
            zvals = np.squeeze(np.array(self.mat_data[var_key_z][idx[newOrder[2]]]))
            xx, yy, zz = np.meshgrid(xvals, yvals, zvals, indexing='ij')
            vals = dataSlice.flatten()
            self.plotWindow.plot_3d(
                xx, yy, zz, vals,
                xlabel=self.dimLabels[self.dimX],
                ylabel=self.dimLabels[self.dimY],
                zlabel=self.dimLabels[self.dimZ],
                title=f"{self.operatorLabels[self.operator]} of {self.get_attr_name_by_var(self.attribute_key)}"
            )

        # If a heatmap is requested in 2D mode, update the heatmap window similarly.
        if heatmap and self.plotType == 2:
            var_key_x = self.dataset['parameters'][self.dimX]['variable'][0]
            var_key_y = self.dataset['parameters'][self.dimY]['variable'][0]
            xvals = np.squeeze(np.array(self.mat_data[var_key_x][idx[newOrder[0]]]))
            yvals = np.squeeze(np.array(self.mat_data[var_key_y][idx[newOrder[1]]]))
            # extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]
            self.heatmapWindow.plot_2d(
            xvals, yvals,
            dataSlice, 
            # extent, 
            self.dimLabels[self.dimX], 
            self.dimLabels[self.dimY], 
            f"{self.operatorLabels[self.operator]} of {self.get_attr_name_by_var(self.attribute_key)} (Heatmap)"
            )

            # Add value labels at the center of each cell in the heatmap
            ax = self.heatmapWindow.figure.gca()
            # num_rows, num_cols = dataSlice.shape
            # x_centers = xvals + (xvals[1] - xvals[0]) / 2
            # y_centers = yvals + (yvals[1] - yvals[0]) / 2
            x_centers = xvals 
            y_centers = yvals 
            for i, y in enumerate(y_centers):
                for j, x in enumerate(x_centers):
                    if not np.isnan(dataSlice[j, i]):
                        ax.text(x, y, f"{dataSlice[j, i]:g}", color="black", ha="center", va="center", fontsize=8)
            self.heatmapWindow.canvas.draw()

    def updateWidgetStates(self):
        """
        Dynamically enable or disable widgets based on the current state of the application.
        """
        # Block signals to avoid triggering onAxisChange while updating combo boxes
        self.xAxisCombo.blockSignals(True)
        self.yAxisCombo.blockSignals(True)
        self.zAxisCombo.blockSignals(True)

        # Example: Enable/disable Z-axis combo box based on the selected plot type
        if self.plotType > 2:  # 3D plot requires Z-axis
            if not self.zAxisCombo.isEnabled():
                self.zAxisCombo.addItems(self.dimLabels)
                self.zAxisCombo.setCurrentIndex(self.dimZ)
            self.zAxisCombo.setEnabled(True)
            # self.zAxisCombo.show()
        else:
            self.zAxisCombo.setEnabled(False)
            self.zAxisCombo.clear()  # Clear items when disabled
            # self.zAxisCombo.hide()
                
        if self.plotType > 1:  # 2D,3D plot requires Y-axis
            if not self.yAxisCombo.isEnabled():
                self.yAxisCombo.addItems(self.dimLabels)
                self.yAxisCombo.setCurrentIndex(self.dimY)
            self.yAxisCombo.setEnabled(True)
            # self.yAxisCombo.show()
        else:
            self.yAxisCombo.setEnabled(False)
            self.yAxisCombo.clear()  # Clear items when disabled
            # self.yAxisCombo.hide()

        # Example: Enable/disable heatmap button only for 2D plots
        if self.plotType == 2:
            self.heatmapButton.setEnabled(True)
        else:
            self.heatmapButton.setEnabled(False)

        # Re-enable signals after updating
        self.xAxisCombo.blockSignals(False)
        self.yAxisCombo.blockSignals(False)
        self.zAxisCombo.blockSignals(False)

    
        # Example: Enable/disable sliders based on the selected dimensions
        for d, slider in enumerate(self.sliders):
            slider.blockSignals(True)  # Block signals to avoid triggering onSliderChange while updating sliders
            if d in [self.dimX, self.dimY, self.dimZ][0:self.plotType]:
                slider.setValue((0, self.sz[d] - 1))  # Reset slider to full range for axes in use
                self.sliceIndex[d] = (0, self.sz[d] - 1)
                # slider.setEnabled(False)  # Disable sliders for axes currently in use for 2D or 3D plots
                self.overlapCheckboxes[d].setEnabled(False)  
            else:
                self.overlapCheckboxes[d].setEnabled(self.plotType == 1)  # Enable overlap checkbox only for 1D plots 
                if not self.overlapCheckboxes[d].isChecked():
                    slider.setValue((0,))  # Reset slider to full range for axes in use
                    self.sliceIndex[d] = (0, )
                    # slider.setEnabled(True)  # Enable sliders for other dimensions
            slider.blockSignals(False)  # Re-enable signals after updating sliders

    def copyToClipboard(self):
        """
        Copy the current data slice and axes data to the clipboard in tab-delimited text format.
        """
        # Determine which dimensions are used for plotting
        if self.plotType == 1:
            plotDims = [self.dimX]
        elif self.plotType == 2:
            plotDims = [self.dimX, self.dimY]
        elif self.plotType == 3:
            plotDims = [self.dimX, self.dimY, self.dimZ]
        else:
            plotDims = [self.dimX]

        # Build an index tuple for slicing the data
        idx = [slice(0, self.sz[d]) if d in plotDims else slice(self.sliceIndex[d], self.sliceIndex[d] + 1) for d in range(self.nd)]
        dataSlice = self.data[tuple(idx)]

        # Rearrange dimensions so that the plot dimensions come first
        allDims = list(range(len(idx)))
        otherDims = [d for d in allDims if d not in plotDims]
        newOrder = plotDims + otherDims
        dataSlice = np.transpose(dataSlice, newOrder)

        # Reshape to just the plot dimensions
        newShape = dataSlice.shape[:len(plotDims)]
        dataSlice = dataSlice.reshape(newShape)

        # Apply the chosen operator
        func = self.operatorFunctions[self.operator]
        dataSlice = func(dataSlice)

        # Get axes data
        axesData = []
        for dim in plotDims:
            var_key = self.dataset['parameters'][dim]['variable'][0]
            axesData.append(np.squeeze(np.array(self.mat_data[var_key])))

        # Format data as tab-delimited text
        text = ""
        if len(axesData) == 1:  # 1D data
            text += "X\tData\n"
            for x, val in zip(axesData[0], dataSlice):
                text += f"{x}\t{val}\n"
        elif len(axesData) == 2:  # 2D data
            text += "X\t" + "\t".join(map(str, axesData[1])) + "\n"
            for x, row in zip(axesData[0], dataSlice):
                text += f"{x}\t" + "\t".join(map(str, row)) + "\n"
        else:
            QMessageBox.warning(self, "Unsupported Format", "Copying 3D data to MATLAB in text mode is not supported.")
            return

        # Copy to clipboard
        self.clipboard.setText(text)
        print("Data copied to clipboard in tab-delimited text format.")


def main():
    from PySide6.QtWidgets import QInputDialog
    app = QApplication(sys.argv)

    # Open a file dialog to select the .mat file
    mat_filename, _ = QFileDialog.getOpenFileName(
        None, "Select .mat File", "", "MAT Files (*.mat)"
    )

    if not mat_filename:
        QMessageBox.warning(None, "No File Selected", "Please select a valid .mat file.")
        sys.exit(0)

    # Load the .mat file
    # t  = trim_singular_list_2(read_mat(mat_filename))
    try:
        mat_data  = trim_singular_list_2(read_mat(mat_filename))[0]
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load .mat file: {e}")
        sys.exit(1)

    # Filter keys
    keys = []
    DATASET_KEYS = ['dataset', 'Lumerical_dataset']
    for key in mat_data.keys():
        if key[0:2] != "__" and isinstance(mat_data[key][0], dict) and any(dataset_key in mat_data[key][0] for dataset_key in DATASET_KEYS):
                keys.append(key)
        
    if not keys:
        QMessageBox.warning(None, "No Valid Dataset", "No valid datasets found in the .mat file.")
        sys.exit(1)

    # Sort the keys
    keys.sort()

    # Prompt the user to select a dataset key
    selected_key, ok = QInputDialog.getItem(
            None, "Select Dataset Key", "Choose a dataset key:", keys, 0, False
        )

    if not ok or not selected_key:
        QMessageBox.warning(None, "No Selection", "Please select a dataset key.")
        sys.exit(0)

    # Retrieve the dataset
    dataset = mat_data[selected_key][0]

    # Determine the dataset field
    if 'dataset' in dataset:
        attributes = dataset['dataset'][0]['attributes'][0]
    elif 'Lumerical_dataset' in dataset:
        attributes = dataset['Lumerical_dataset'][0]['attributes'][0]
    else:
        QMessageBox.warning(None, "Invalid Dataset", "The selected key does not contain a valid dataset.")
        sys.exit(1)

    # Check if attributes are valid
    if not (isinstance(attributes, dict) and 'name' in attributes and 'variable' in attributes):
        QMessageBox.warning(None, "Invalid Attributes", "No valid attributes found in the selected dataset.")
        sys.exit(1)

    # Prompt the user to select an attribute
    attribute_names = [f"{name} ({variable})" for name, variable in zip(attributes['name'], attributes['variable'])]
    selected_attribute, ok = QInputDialog.getItem(
            None, "Select Attribute", "Choose an attribute to visualize:", attribute_names, 0, False
        )

    if not ok or not selected_attribute:
        QMessageBox.warning(None, "No Selection", "Please select an attribute.")
        sys.exit(0)

        # Extract the variable corresponding to the selected attribute
    attribute_index = attribute_names.index(selected_attribute)
    attribute_key = attributes['variable'][attribute_index]

        # Launch the NDVisualizer
    visualizer = NDVisualizer(dataset, attribute_key)
    visualizer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()