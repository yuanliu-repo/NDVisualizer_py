from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QMessageBox,
    QLabel, QComboBox, QPushButton
)
from superqt import QRangeSlider
# from matplotlib import colormaps
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
import numpy as np
from .utils import get_icon_path
from ._version import __version__



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


class Plot_toolbar_1d(NavigationToolbar2QT):
    """
    Custom matplotlib navigation toolbar with standard functionality
    including zoom, pan, save, and other navigation tools.
    """
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        # Get reference to existing actions to position new action correctly
        # existing_actions = self.actions()
        # print(existing_actions)
        # Customize toolbar appearance or add custom actions here if needed
        # self.setStyleSheet("QToolBar { border: 1px solid gray; }")

        
        
        # Add separator
        self.addSeparator()

        # Add line marker combo box
        if parent.parent_window.plotType == 1:
            self.addWidget(QLabel("Marker:"))
            self.marker_combo = QComboBox()
            self.marker_combo.addItems([
                "None", "o", ".", "□", 
                "^", "v", "◇",
                "*", "+", "x"
            ])
            self.marker_combo.setCurrentIndex(0)  # Default to circle
            self.marker_combo.currentIndexChanged.connect(self.on_marker_change)
            self.addWidget(self.marker_combo)

            # Add separator
            self.addSeparator()

            # Add grid toggle button
            self.grid_action = self.addAction("Grid", self.on_grid_toggle)
            self.grid_action.setCheckable(True)
            self.grid_action.setChecked(False)  # Default to no grid
            self.grid_action.setToolTip("Toggle grid")

        # Add separator
        # self.addSeparator()

        # Add colormap combo box
        if parent.parent_window.plotType in [2, 3]:
            self.addWidget(QLabel("Colormap:"))
            self.cmap_combo = QComboBox()
            self.cmap_combo.addItems([
                "viridis", "plasma", "inferno", "magma", "cividis",
                "Blues", "Greens", "Reds", "Oranges", "Purples",
                "coolwarm", "RdYlBu", "RdBu", "seismic",
                "jet", "rainbow", "hot", "cool", "spring", "summer",
                "autumn", "winter", "gray", "bone", "copper"
            ])
            # self.cmap_combo.addItems(list(colormaps))  # Use matplotlib's colormaps
            # self.cmap_combo.setCurrentIndex(12)  # Default to viridis
            self.cmap_combo.currentIndexChanged.connect(self.on_colormap_change)
            self.addWidget(self.cmap_combo)

        # Add custom copy to clipboard button
        self.addSeparator()
        # self.copy_fig_action = self.addAction(self.style().standardIcon(self.style().StandardPixmap.SP_DialogSaveButton), "Copy", self.copy_figure_as_png)
        self.copy_fig_action = self.addAction(".PNG", self.copy_figure_as_png)
        self.copy_fig_action.setToolTip("Copy figure")
        # self.update_widges(parent.parent_window.plotType)


    def update_widges(self, plotType):
        """
        Update the toolbar based on the plot type.
        For 1D plots, show marker and colormap options.
        For 2D plots, show colormap options only.
        """
        if plotType == 1:
            # For 1D plots, show only relevant buttons
            # self.copy_fig_action.setVisible(True)
            self.marker_combo.show()
            self.cmap_combo.hide()
        elif plotType == 2 or plotType == 3:
            # For 2D plots, show all buttons
            # self.copy_fig_action.setVisible(True)
            self.marker_combo.hide()
            self.cmap_combo.show()

        
    def on_marker_change(self, index):
        """Handle marker style change"""
        markers = ['', 'o', '.', 's', '^', 'v', 'D', '*', '+', 'x']
        marker = markers[index]
        
        # Update all line plots in the current axes
        ax = self.canvas.figure.gca()
        for line in ax.get_lines():
            line.set_marker(marker)
        
        self.canvas.draw()

    def on_colormap_change(self, index):
        """Handle colormap change"""
        colormaps = [
            "viridis", "plasma", "inferno", "magma", "cividis",
            "Blues", "Greens", "Reds", "Oranges", "Purples",
            "coolwarm", "RdYlBu", "RdBu", "seismic",
            "jet", "rainbow", "hot", "cool", "spring", "summer",
            "autumn", "winter", "gray", "bone", "copper"
        ]
        
        cmap_name = list(colormaps)[index]
        
        # Update the colormap for the current plot
        ax = self.canvas.figure.gca()
        images = ax.get_images()
        if images:
            for im in images:
                im.set_cmap(cmap_name)
        # Update colorbar if it exists - check collections even if no images
        if hasattr(ax, 'collections') and ax.collections:
            for collection in ax.collections:
                collection.set_cmap(cmap_name)
        # Force redraw of the figure including colorbar
        self.canvas.figure.canvas.draw_idle()

    def on_grid_toggle(self):
        """
        Toggle the grid visibility for the current axes.
        """
        ax = self.canvas.figure.gca()
        if self.grid_action.isChecked():
            ax.grid(True)
        else:
            ax.grid(False)
        self.canvas.draw()

    def copy_figure_as_png(self):
        """
        Copy the current figure as a PNG to the clipboard.
        """
        from PySide6.QtGui import QImage
        from io import BytesIO
        buffer = BytesIO()
        self.canvas.figure.savefig(buffer, format='png')
        buffer.seek(0)

        qimage = QImage()
        qimage.loadFromData(buffer.getvalue())
        clipboard = QApplication.clipboard()
        clipboard.setImage(qimage)
        
        QMessageBox.information(self, "Copied", "Figure copied to clipboard as PNG.")



    def copy_figure_as_svg(self):
        """
        Copy the current figure as an SVG to the clipboard.
        """
        from PySide6.QtCore import QMimeData
        from io import BytesIO
        buffer = BytesIO()
        self.canvas.figure.savefig(buffer, format='svg')
        buffer.seek(0)
        
        # Copy to clipboard
        clipboard = QApplication.clipboard()
        mime_data = QMimeData()
        mime_data.setData("image/svg+xml", buffer.getvalue())
        clipboard.setMimeData(mime_data)
        
        QMessageBox.information(self, "Copied", "Figure copied to clipboard as SVG.")


class plot_toolbar_2d(NavigationToolbar2QT):
    """
    Custom matplotlib navigation toolbar for 2D plots with additional functionality.
    This toolbar includes standard navigation tools and a custom copy to clipboard button.
    """
    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent)
        

class PlotWindow(QMainWindow):
    def __init__(self, title, parent=None):
        super().__init__()
        self.parent_window = parent
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        # Set the window icon
        self.setWindowIcon(QIcon(get_icon_path()))  # Ensure icon.ico is in the same directory as the script

        # Create a QWidget to act as the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create a horizontal layout for the central widget with minimal margins
        self.layout = QHBoxLayout(self.central_widget)
        self.layout.setContentsMargins(5, 2, 2, 5)  # Minimize margins
        self.layout.setSpacing(2)  # Minimize spacing between elements

        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)  # No margins for plot layout
        plot_layout.setSpacing(2)  # Minimal spacing
        
        # Initialize the matplotlib figure and canvas
        self.figure = Figure()
        # Minimize figure margins
        # self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        
        # Create the navigation toolbar
        self.toolbar = Plot_toolbar_1d(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        
        # Create a widget to contain the plot layout
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        
        # Add the plot widget to the main horizontal layout
        self.layout.addWidget(plot_widget)

        # Create a vertical layout for the slider and its buttons
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)  # No margins for slider layout
        slider_layout.setSpacing(2)  # Minimal spacing
        
        # Add a reset button on top of the slider
        # Circular arrows (reset/refresh)
        self.reset_button = QPushButton("R") 
        self.reset_button.setMaximumSize(25, 25)
        self.reset_button.setToolTip("Reset slider range")
        self.reset_button.clicked.connect(self.reset_axis)
        slider_layout.addWidget(self.reset_button)
        
        # Add the vertical slider
        self.range_slider = QSlider_helper(Qt.Vertical)
        self.range_slider.setMinimum(-10)
        self.range_slider.setMaximum(110)
        self.range_slider.setValue((0,100))
        self.range_slider.setMaximumWidth(30)  # Limit the width to reduce right margin
        self.range_slider.valueChanged.connect(self.update_range)
        self.range_slider.sliderPressed.connect(self.update_range_pressed)  
        self.range_slider.sliderReleased.connect(self.update_range_released)  

        
        slider_layout.addWidget(self.range_slider)
        
        # Add a lock button below the slider
        self.lock_button = QPushButton("🔒")  # Using Unicode lock symbol
        self.lock_button.setMaximumSize(25, 25)
        self.lock_button.setToolTip("Lock/unlock slider")
        self.lock_button.setCheckable(True)
        self.lock_button.clicked.connect(self.toggle_lock)
        slider_layout.addWidget(self.lock_button)
        # Add the slider layout to the main layout
        self.layout.addLayout(slider_layout)


    def toggle_lock(self):
        """
        Toggle the lock state of the slider.
        When locked, the slider cannot be moved.
        """
        if self.lock_button.isChecked():
            # self.range_slider.setEnabled(False)
            self.update_range0()  # Update the initial range when locking
            # self.lock_button.setText("🔓")
        else:
            # self.range_slider.setEnabled(True)
            # self.lock_button.setText("🔒")
            pass

    def init_axis(self):
        if self.lock_button.isChecked():
            if self.parent_window.plotType == 1:
                # For 1D plots or plots without colorbar, adjust y-axis
                ax = self.figure.gca()
                ax.set_ylim(*self.range0)
            elif self.parent_window.plotType == 2 or self.parent_window.plotType == 3:
                # For 2D plots with colorbar, adjust colorbar limits
                self.plt.set_clim(*self.range0)
        else:
            self.reset_axis()


    def reset_axis(self):
        """
        Reset the slider range to its default values.
        """
        self.range_slider.setValue((0,100))
        # Reset the axes limits to auto-scale
        if self.parent_window.plotType == 1:
                # For 1D plots or plots without colorbar, adjust y-axis
            ax = self.figure.gca()
            ax.autoscale(enable=True, axis='y')

        elif self.parent_window.plotType == 2 or self.parent_window.plotType == 3:
            self.plt.autoscale()

        self.update_range0() 
        self.canvas.draw()

    def update_range0(self):
        """
        Update the initial range based on the current plot type.
        This is called when the slider is first pressed to set the initial range.
        """
        if self.parent_window.plotType == 1:
            # For 1D plots or plots without colorbar, adjust y-axis
            ax = self.figure.gca()
            self.range0 = ax.get_ylim()

        elif self.parent_window.plotType == 2 or self.parent_window.plotType == 3:
            # For 2D plots with colorbar, adjust colorbar limits
            self.range0 = self.plt.get_clim()
    
    def update_range_pressed(self):
        """
        Handle the slider being pressed to update the range.
        This is useful for ensuring the range is set correctly when the slider is first interacted with.
        """
        # print("Slider pressed, updating range...")
        self.update_range0()  # Update the initial range based on the current plot type
            
    def update_range(self, value):
        """
        Update the y-axis range based on the slider value.
        The slider value is expected to be a tuple (min_val, max_val) as percentiles.
        """
        if not hasattr(self, "range0"):
            # print("No initial range0 set, skipping update.")
            return
        
        if isinstance(value, tuple) and len(value) == 2:
            min_percentile, max_percentile = value
            
            if self.parent_window.plotType == 1:
                # For 1D plots or plots without colorbar, adjust y-axis
                ax = self.figure.gca()
                y_min, y_max = self.range0
                
                # Calculate the actual values based on percentiles
                y_range = y_max - y_min
                actual_min = y_min + (min_percentile / 100) * y_range
                actual_max = y_min + (max_percentile / 100) * y_range
                
                # Set the new y-axis limits
                ax.set_ylim(actual_min, actual_max)

            elif self.parent_window.plotType == 2 or self.parent_window.plotType == 3:
                # For 2D plots with colorbar, adjust colorbar limits
                
                vmin, vmax = self.range0
                
                # Calculate the actual values based on percentiles
                v_range = vmax - vmin
                actual_min = vmin + (min_percentile / 100) * v_range
                actual_max = vmin + (max_percentile / 100) * v_range
                
                # Set the new colorbar limits
                self.plt.set_clim(actual_min, actual_max)

            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Invalid Range", "Please select a valid range using the slider.")

    def update_range_released(self):
        """
        Handle the slider being released to finalize the range update.
        This is useful for ensuring the range is set correctly after the user has finished adjusting the slider.
        """
        self.range_slider.blockSignals(True) 
        self.range_slider.setValue((0,100))
        self.range_slider.blockSignals(False)
        self.update_range0()  # Update the initial range based on the current plot type


    # def update_range(self, value):
    #     """
    #     Update the y-axis range based on the slider value.
    #     The slider value is expected to be a tuple (min_val, max_val) as percentiles.
    #     """
    #     if isinstance(value, tuple) and len(value) == 2:
    #         self.range_slider.blockSignals(True)  # Temporarily block signals to avoid recursion
    #         self.range_slider.setValue((0,100))
    #         self.range_slider.blockSignals(False)
    #         min_percentile, max_percentile = value
            
    #         if self.parent_window.plotType == 1:
    #             # For 1D plots or plots without colorbar, adjust y-axis
    #             ax = self.figure.gca()
    #             y_min, y_max = ax.get_ylim()
                
    #             # Calculate the actual values based on percentiles
    #             y_range = y_max - y_min
    #             actual_min = y_min + (min_percentile / 100) * y_range
    #             actual_max = y_min + (max_percentile / 100) * y_range
                
    #             # Set the new y-axis limits
    #             ax.set_ylim(actual_min, actual_max)

    #         elif self.parent_window.plotType == 2 or self.parent_window.plotType == 3:
    #             # For 2D plots with colorbar, adjust colorbar limits
                
    #             vmin, vmax = self.plt.get_clim()
                
    #             # Calculate the actual values based on percentiles
    #             v_range = vmax - vmin
    #             actual_min = vmin + (min_percentile / 100) * v_range
    #             actual_max = vmin + (max_percentile / 100) * v_range
                
    #             # Set the new colorbar limits
    #             self.plt.set_clim(actual_min, actual_max)

    #         self.canvas.draw()
    #     else:
    #         QMessageBox.warning(self, "Invalid Range", "Please select a valid range using the slider.")


    def clear_plot(self):
        # Clear the matplotlib figure
        self.figure.clear()

    def update_toolbar(self):
        # self.toolbar.update_widges(self.parent_window.plotType)
        # Remove existing toolbar
        if hasattr(self, 'toolbar') and self.toolbar:
            self.toolbar.setParent(None)
            self.toolbar.deleteLater()

        # Create new toolbar based on plot type
        self.toolbar = Plot_toolbar_1d(self.canvas, self)

        # Add the new toolbar to the plot layout
        plot_layout = self.central_widget.layout().itemAt(0).widget().layout()
        plot_layout.addWidget(self.toolbar)




############# # Resize Event Handling #############
    def resizeEvent(self, event):
        from PySide6.QtCore import QTimer
        """
        Handle window resize events by adjusting the plot layout.
        """
        super().resizeEvent(event)
        # print("Resize event triggered")
        # Only apply tight layout when resize is finished to avoid excessive redraws
        if not hasattr(self, '_resize_timer'):
            self._resize_timer = QTimer()
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._on_resize_finished)
        
        self._resize_timer.stop()
        self._resize_timer.start(100)  # Wait 100ms after resize stops

    def _on_resize_finished(self):
        """
        Called when window resize is finished to update the plot layout.
        """
        self.figure.tight_layout()
        self.canvas.draw()
        # print("Resize finished, layout updated")
############# # End of Resize Event Handling #############


    def plot_1d(self, xvals, yvals, xlabel, ylabel, title, legend=None):
        marker = self.toolbar.marker_combo.currentText()
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        self.plt = ax.plot(xvals, yvals, marker=marker)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if legend:
            ax.legend(legend[1:], loc='best', title=legend[0]) 
        ax.grid(self.toolbar.grid_action.isChecked())  # Set grid based on toolbar state

        self.init_axis()
        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_2d(self, xvals, yvals, data, xlabel, ylabel, title):
        cmap = self.toolbar.cmap_combo.currentText()
        self.plt = self.plot_2d_pcolormesh(xvals, yvals, data, xlabel, ylabel, title, cmap)

        self.init_axis()
        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_2d_imshow(self, data, extent, xlabel, ylabel, title, cmap='viridis'):
        self.clear_plot()
        ax = self.figure.add_subplot(111)
        cax = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.figure.colorbar(cax, ax=ax)
        return cax
        

    def plot_2d_pcolormesh(self, xvals, yvals, data, xlabel, ylabel, title, cmap='viridis'):
        self.clear_plot()
        ax = self.figure.add_subplot(111)

        # Create a meshgrid from xvals and yvals
        X, Y = np.meshgrid(xvals, yvals, indexing='ij')

        # Plot the data using pcolormesh
        cax = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.figure.colorbar(cax, ax=ax)
        return cax

    

    
    def plot_2d_heatmap(self, xvals, yvals, data, xlabel, ylabel, title, cmap='viridis'):
        def _auto_rotate_labels(ax, xlabels):
            """
            Automatically decide the rotation angle for x-axis labels based on their length and content.
            
            Args:
                labels: List of label strings
                
            Returns:
                tuple: (rotation_angle, horizontal_alignment)
            """
            if not xlabels:
                return 0, 'center'
            
            # Calculate average label length
            avg_length = sum(len(str(label)) for label in xlabels) / len(xlabels)
            max_length = max(len(str(label)) for label in xlabels)
            
            # Check if labels contain numbers only (might be better horizontal)
            all_numeric = all(str(label).replace('.', '').replace('-', '').isdigit() 
                            for label in xlabels)
            
            # Decision logic
            if max_length <= 3 and all_numeric:
                # Short numeric labels - no rotation
                ax.set_xticklabels(xlabels, rotation=0, ha='center')

            elif avg_length <= 6 and len(xlabels) <= 10:
                # Short labels with few items - no rotation
                ax.set_xticklabels(xlabels, rotation=0, ha='center')
                
            elif avg_length <= 10:
                # Medium length labels - slight rotation
                ax.set_xticklabels(xlabels, rotation=30, ha='right')
            else:
                # Long labels - full rotation
                ax.set_xticklabels(xlabels, rotation=45, ha='right')


        ax = self.figure.add_subplot(111)

        x_labels = [f"{x:g}" if isinstance(x, (int, float)) else x for x in xvals]
        y_labels = [f"{y:g}" if isinstance(y, (int, float)) else y for y in yvals]

        x = np.arange(len(x_labels))
        y = np.arange(len(y_labels))
        X, Y = np.meshgrid(x, y, indexing='ij')

        cax = ax.pcolormesh(X, Y, data, cmap=cmap, edgecolors='k')

        # Set ticks in center of cells
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_yticks(np.arange(len(y_labels)))
        _auto_rotate_labels(ax, x_labels)
        ax.set_yticklabels(y_labels)

        # Optional: reverse Y-axis
        ax.invert_yaxis()

        # Annotate each cell
        for i in range(data.shape[1]):
            for j in range(data.shape[0]):
                ax.text(j, i, f"{data[j, i]:g}",
                        va='center', ha='center', color='white')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        self.figure.colorbar(cax, ax=ax)

        self.plt = cax
        self.init_axis()
        # Auto-adjust layout to prevent clipping
        self.figure.tight_layout()
        self.canvas.draw()
        

    # def plot_2d_heatmap(self, xvals, yvals, data, xlabel, ylabel, title, cmap='viridis'):

    #     def _auto_rotate_labels(self, labels):
    #         """
    #         Automatically decide the rotation angle for x-axis labels based on their length and content.
            
    #         Args:
    #             labels: List of label strings
                
    #         Returns:
    #             tuple: (rotation_angle, horizontal_alignment)
    #         """
    #         if not labels:
    #             return 0, 'center'
            
    #         # Calculate average label length
    #         avg_length = sum(len(str(label)) for label in labels) / len(labels)
    #         max_length = max(len(str(label)) for label in labels)
            
    #         # Check if labels contain numbers only (might be better horizontal)
    #         all_numeric = all(str(label).replace('.', '').replace('-', '').isdigit() 
    #                          for label in labels)
            
    #         # Decision logic
    #         if max_length <= 3 and all_numeric:
    #             # Short numeric labels - no rotation
    #             return 0, 'center'
    #         elif avg_length <= 6 and len(labels) <= 10:
    #             # Short labels with few items - no rotation
    #             return 0, 'center'
    #         elif avg_length <= 10:
    #             # Medium length labels - slight rotation
    #             return 30, 'right'
    #         else:
    #             # Long labels - full rotation
    #             return 45, 'right'

    #     self.clear_plot()
    #     ax = self.figure.add_subplot(111)

    #     # Convert xvals and yvals to strings for categorical plotting
    #     x_labels = [str(x) for x in xvals]
    #     y_labels = [str(y) for y in yvals]

    #     # Create the heatmap using imshow
    #     # Create a meshgrid from the indices for pcolormesh
    #     X, Y = np.meshgrid(range(len(x_labels)), range(len(y_labels)), indexing='ij')
    #     cax = ax.pcolormesh(X, Y, data, shading='auto', cmap=cmap, edgecolors='k')

    #     # Set the tick positions and labels
    #     ax.set_xticks(range(len(x_labels)))
    #     ax.set_yticks(range(len(y_labels)))
    #     ax.set_xticklabels(x_labels, rotation=45, ha='right')
    #     ax.set_yticklabels(y_labels)

    #     # Set labels and title
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel(ylabel)
    #     ax.set_title(title)

    #     # Add colorbar
    #     self.figure.colorbar(cax, ax=ax)

    #     # Adjust layout for a tight fit
    #     self.figure.tight_layout()
    #     self.canvas.draw()
    #     return cax

    def plot_2d_contour(self, xvals, yvals, data, xlabel, ylabel, title, cmap='viridis'):
        self.clear_plot()
        ax = self.figure.add_subplot(111)

        # Create a meshgrid from xvals and yvals
        X, Y = np.meshgrid(xvals, yvals)

        # Plot the data using contourf
        self.plt = ax.contourf(X, Y, data, levels=50, cmap=cmap)

        # Set labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Add a colorbar
        self.figure.colorbar(self.plt, ax=ax)

        self.init_axis()
        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_3d(self, xvals, yvals, zvals, vals, xlabel, ylabel, zlabel, title):
        self.clear_plot()
        cmap = self.toolbar.cmap_combo.currentText()
        ax = self.figure.add_subplot(111, projection='3d')
        self.plt = ax.scatter(xvals, yvals, zvals, c=vals, cmap=cmap)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        self.figure.colorbar(self.plt, ax=ax, shrink=0.5, aspect=10)

        self.init_axis()
        # Adjust layout for a tight fit
        self.figure.tight_layout()
        self.canvas.draw()
