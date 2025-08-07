
from PySide6.QtWidgets import  QMessageBox, QApplication
import sys, os
from PySide6.QtCore import Qt
import pandas as pd
# import re

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, 
                              QTableWidgetItem, QLabel, QSpinBox, QPushButton, 
                              QCheckBox, QLineEdit, QComboBox, QGroupBox, 
                              QScrollArea, QMessageBox, QApplication)

class CSVImportDialog(QDialog):
    """
    Advanced dialog class for configuring CSV import parameters with preview.
    """
    
    def __init__(self, filename, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.df = None
        self.raw_lines = []
        self.preview_df = None
        self.index_columns = []
        
        # Set window title with filename
        self.setWindowTitle(f"{filename} - Import CSV")
        # self.setMinimumSize(800, 600)
        # self.resize(1000, 700)
        
        self.setup_ui()
        self.load_raw_data()
        self.auto_detect_parameters()
        self.update_preview()
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        
        # Parameters group
        params_group = QGroupBox("Import Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Skip rows
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Skip rows:"))
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setMinimum(0)
        self.skip_spinbox.setMaximum(100)
        self.skip_spinbox.valueChanged.connect(self.on_parameters_changed)
        skip_layout.addWidget(self.skip_spinbox)
        skip_layout.addStretch()
        params_layout.addLayout(skip_layout)
        
        # Header
        header_layout = QHBoxLayout()
        self.header_checkbox = QCheckBox("First row contains headers")
        self.header_checkbox.stateChanged.connect(self.on_parameters_changed)
        header_layout.addWidget(self.header_checkbox)
        header_layout.addStretch()
        params_layout.addLayout(header_layout)
        
        # Separator
        sep_layout = QHBoxLayout()
        sep_layout.addWidget(QLabel("Separator:"))
        self.sep_combo = QComboBox()
        self.sep_combo.addItems(["Auto-detect", "Comma (,)", "Tab", "Semicolon (;)", "Space", "Custom"])
        self.sep_combo.currentTextChanged.connect(self.on_separator_changed)
        sep_layout.addWidget(self.sep_combo)
        
        self.custom_sep_edit = QLineEdit()
        self.custom_sep_edit.setPlaceholderText("Enter custom separator")
        self.custom_sep_edit.setVisible(False)
        self.custom_sep_edit.textChanged.connect(self.on_parameters_changed)
        sep_layout.addWidget(self.custom_sep_edit)
        sep_layout.addStretch()
        params_layout.addLayout(sep_layout)
        
        layout.addWidget(params_group)
        
        # Preview table
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_table = QTableWidget()
        preview_layout.addWidget(self.preview_table)
        
        layout.addWidget(preview_group)
        
        # Column configuration
        config_group = QGroupBox("Column Configuration")
        config_layout = QVBoxLayout(config_group)
        
        # Scroll area for column configs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.config_widget = QVBoxLayout()
        self.config_widget.addStretch()  # Add stretch at the end to push content to top
        scroll_content = QDialog()
        scroll_content.setLayout(self.config_widget)
        scroll.setWidget(scroll_content)
        config_layout.addWidget(scroll)
        
        layout.addWidget(config_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.auto_detect_btn = QPushButton("Auto-detect All")
        self.auto_detect_btn.clicked.connect(self.auto_detect_parameters)
        button_layout.addWidget(self.auto_detect_btn)
        
        button_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)
        
        layout.addWidget(QDialog(parent=self))  # Spacer
        layout.addLayout(button_layout)
        
    def load_raw_data(self):
        """Load raw file data for analysis."""
        try:
            with open(self.filename, 'r', encoding='utf-8', errors='ignore') as f:
                self.raw_lines = f.readlines()[:50]  # Read first 50 lines for analysis
        except UnicodeDecodeError:
            try:
                with open(self.filename, 'r', encoding='latin-1') as f:
                    self.raw_lines = f.readlines()[:50]
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file: {str(e)}")
                
    def detect_separator(self):
        """Auto-detect the most likely separator."""
        if not self.raw_lines:
            return ','
            
        separators = [',', '\t', ';', ' ', '|']
        separator_counts = {}
        
        # Analyze first few lines
        for line in self.raw_lines[:10]:
            for sep in separators:
                count = line.count(sep)
                if sep not in separator_counts:
                    separator_counts[sep] = []
                separator_counts[sep].append(count)
        
        # Find most consistent separator
        best_sep = ','
        best_score = 0
        
        for sep, counts in separator_counts.items():
            if len(counts) > 0:
                avg_count = sum(counts) / len(counts)
                consistency = 1.0 / (1.0 + sum(abs(c - avg_count) for c in counts) / len(counts))
                score = avg_count * consistency
                
                if score > best_score and avg_count > 0:
                    best_score = score
                    best_sep = sep
                    
        return best_sep
        
    def detect_skip_rows(self):
        """Auto-detect number of rows to skip."""
        if not self.raw_lines:
            return 0
            
        separator = self.detect_separator()
        skip_rows = 0
        
        # Look for consistent column count
        for i, line in enumerate(self.raw_lines[:20]):
            cols = len(line.split(separator))
            
            # Check next few lines for consistency
            consistent = True
            for j in range(i + 1, min(i + 5, len(self.raw_lines))):
                if len(self.raw_lines[j].split(separator)) != cols:
                    consistent = False
                    break
                    
            if consistent and cols > 1:
                skip_rows = i
                break
                
        return skip_rows
        
    def detect_header(self):
        """Auto-detect if first data row contains headers."""
        if not self.raw_lines:
            return False
            
        skip_rows = self.detect_skip_rows()
        separator = self.detect_separator()
        
        if skip_rows >= len(self.raw_lines):
            return False
            
        first_data_line = self.raw_lines[skip_rows]
        cols = first_data_line.split(separator)
        
        # Check if first row looks like headers (non-numeric strings)
        non_numeric_count = 0
        for col in cols:
            col = col.strip().strip('"\'')
            try:
                float(col)
            except ValueError:
                if col and not col.isspace():
                    non_numeric_count += 1
                    
        return non_numeric_count > len(cols) * 0.5
        
    def auto_detect_parameters(self):
        """Auto-detect all import parameters."""
        skip_rows = self.detect_skip_rows()
        has_header = self.detect_header()
        separator = self.detect_separator()
        
        self.skip_spinbox.setValue(skip_rows)
        self.header_checkbox.setChecked(has_header)
        
        # Set separator in combo box
        sep_map = {',': "Comma (,)", '\t': "Tab", ';': "Semicolon (;)", ' ': "Space"}
        if separator in sep_map:
            self.sep_combo.setCurrentText(sep_map[separator])
        else:
            self.sep_combo.setCurrentText("Custom")
            self.custom_sep_edit.setText(separator)
            
        self.update_preview()
        
    def get_separator(self):
        """Get the current separator setting."""
        sep_text = self.sep_combo.currentText()
        if sep_text == "Auto-detect":
            return self.detect_separator()
        elif sep_text == "Comma (,)":
            return ','
        elif sep_text == "Tab":
            return '\t'
        elif sep_text == "Semicolon (;)":
            return ';'
        elif sep_text == "Space":
            return ' '
        elif sep_text == "Custom":
            return self.custom_sep_edit.text() or ','
        return ','
        
    def on_separator_changed(self):
        """Handle separator combo box change."""
        self.custom_sep_edit.setVisible(self.sep_combo.currentText() == "Custom")
        self.on_parameters_changed()
        
    def on_parameters_changed(self):
        """Handle parameter changes."""
        self.update_preview()
        
    def update_preview(self):
        """Update the preview table and column configuration."""
        try:
            skip_rows = self.skip_spinbox.value()
            separator = self.get_separator()
            header = 0 if self.header_checkbox.isChecked() else None
            
            # Read with pandas
            self.preview_df = pd.read_csv(
                self.filename,
                sep=separator,
                skiprows=skip_rows,
                header=header,
                nrows=20,  # Preview only first 20 rows
                dtype=str  # Keep as strings for preview
            )
            
            # Update preview table
            self.update_preview_table()
            
            # Update column configuration
            self.update_column_config()
            
        except Exception as e:
            QMessageBox.warning(self, "Preview Error", f"Could not preview with current settings: {str(e)}")
            
    def update_preview_table(self):
        """Update the preview table widget."""
        if self.preview_df is None:
            return
            
        df = self.preview_df
        self.preview_table.setRowCount(len(df))
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        
        for i in range(len(df)):
            for j in range(len(df.columns)):
                item = QTableWidgetItem(str(df.iloc[i, j]))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(i, j, item)
                
        self.preview_table.resizeColumnsToContents()
        
    def clear_layout(self, layout):
        """Recursively clear a layout and all its children."""
        if layout is not None:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())
    
    def update_column_config(self):
        """Update column configuration widgets."""
        # Clear existing widgets
        for i in reversed(range(self.config_widget.count())):
            child = self.config_widget.itemAt(i)
            if child:
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    self.clear_layout(child.layout())
                self.config_widget.removeItem(child)
                
        if self.preview_df is None:
            return
            
        # Column configurations
        self.column_configs = []
        
        for i, col_name in enumerate(self.preview_df.columns):
            config_layout = QHBoxLayout()
            
            # Column index
            config_layout.addWidget(QLabel(f"Col {i}:"))
            
            # Column name
            name_edit = QLineEdit(str(col_name))
            name_edit.setPlaceholderText(f"Column {i}")
            config_layout.addWidget(name_edit)
            
            # Use as index checkbox
            index_checkbox = QCheckBox("Is axis?")
            index_checkbox.setChecked(i == 0)  # Check if it's the first column
            config_layout.addWidget(index_checkbox)
            
            # Data type detection
            sample_data = self.preview_df[col_name].dropna().head(10)
            detected_type = self.detect_column_type(sample_data)
            type_label = QLabel(f"({detected_type})")
            type_label.setStyleSheet("color: gray; font-style: italic;")
            config_layout.addWidget(type_label)
            
            config_layout.addStretch()
            
            self.config_widget.addLayout(config_layout)
            self.column_configs.append({
                'name_edit': name_edit,
                'index_checkbox': index_checkbox,
                'original_name': col_name,
                'detected_type': detected_type
            })
            
    def detect_column_type(self, series):
        """Detect the likely data type of a column."""
        if len(series) == 0:
            return "unknown"
            
        # Try to convert to numeric
        numeric_count = 0
        for val in series:
            try:
                float(str(val))
                numeric_count += 1
            except (ValueError, TypeError):
                pass
                
        if numeric_count > len(series) * 0.8:
            # Check if integers
            int_count = 0
            for val in series:
                try:
                    float_val = float(str(val))
                    if float_val.is_integer():
                        int_count += 1
                except (ValueError, TypeError):
                    pass
                    
            if int_count > len(series) * 0.9:
                return "integer"
            else:
                return "float"
        else:
            return "text"
            
    def get_configured_dataframe(self):
        """Return the configured dataframe."""
        if self.preview_df is None:
            return None
            
        try:
            skip_rows = self.skip_spinbox.value()
            separator = self.get_separator()
            header = 0 if self.header_checkbox.isChecked() else None
            
            # Read full dataframe
            df = pd.read_csv(
                self.filename,
                sep=separator,
                skiprows=skip_rows,
                header=header
            )
            
            # Apply column renames
            if hasattr(self, 'column_configs'):
                rename_dict = {}
                index_cols = []
                
                for i, config in enumerate(self.column_configs):
                    new_name = config['name_edit'].text().strip()
                    if new_name and new_name != config['original_name']:
                        rename_dict[config['original_name']] = new_name
                        
                    if config['index_checkbox'].isChecked():
                        final_name = new_name if new_name else config['original_name']
                        index_cols.append(final_name)
                
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    
                if index_cols:
                    df = df.set_index(index_cols)
                    
            return df
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load configured dataframe: {str(e)}")
            return None
            
    def show_dialog(self):
        """Show dialog and return index columns if accepted."""
        if self.exec() == QDialog.Accepted:
            if hasattr(self, 'column_configs'):
                index_cols = []
                for config in self.column_configs:
                    if config['index_checkbox'].isChecked():
                        new_name = config['name_edit'].text().strip()
                        final_name = new_name if new_name else config['original_name']
                        index_cols.append(final_name)
                return index_cols
            return []
        return None


def show_csv_import_dialog(filename, parent=None):
    """
    Show the CSV import dialog and return the index columns if accepted.
    """
    dialog = CSVImportDialog(filename, parent)
    index_columns = dialog.show_dialog()
    
    if index_columns is None:
        return None  # User cancelled
    
    df = dialog.get_configured_dataframe()
    
    if df is None:
        raise ValueError("CSV import failed or was cancelled.")
    
    return df, index_columns


def load_csv_file(filename, parent=None):
    """
    Load a CSV file and return the dataframe and index columns,
    then convert to the dict as follows:
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
    df, index_columns = show_csv_import_dialog(filename, parent)
    
    if df is None:
        raise ValueError("CSV import failed or was cancelled.")
    

    data_dict = {}
    var_attr_names = []
    var_param_names = []
    # Handle index columns (parameters)
    if len(index_columns) > 0:
        if len(index_columns) == 1:
            # Single index case
            var_name = "var" + str(len(var_param_names) + 1)
            data_dict[var_name] = df.index.tolist()
            var_param_names.append(var_name)
        else:
            # Multi-index case
            for col in index_columns:
                var_name = "var" + str(len(var_param_names) + 1)
                data_dict[var_name] = df.index.get_level_values(col).unique().tolist()
                var_param_names.append(var_name)

    # Handle data columns (attributes)
    for col in df.columns:
        var_name = "var" + str(len(var_param_names) + len(var_attr_names) + 1)
        if len(index_columns) > 1:
            # If there are index columns, unstack the data
            data_dict[var_name] = df[col].unstack().to_numpy()
        else:
            # If no index columns, use data as is
            data_dict[var_name] = df[col].to_numpy()
        var_attr_names.append(var_name)

    # Convert DataFrame to dict
    data_dict["Lumerical_dataset"]= {
        'attributes': [{'variable': var_attr_names, 'name': list(df.columns)}],
        'parameters': [{'variable': var_param_names[i], 'name': n} for (i,n) in enumerate(list(index_columns))]
    }

    
        
    return {os.path.basename(filename):data_dict}
    




# Register the backend
# xr.backends.register_backend("mat", LumericalMATBackend)

if __name__ == "__main__":
    # Example usage
    # d:\Anaconda3\Test_tools\data\LI_Pulse_3.5um_1mm_2503131348.csv
    file_path = "d:\\Anaconda3\\Test_tools\\data\\LI_Pulse_3.5um_1mm_2503131348.csv"  # Replace with your .mat file path
    # save_file_path = "demo_waveguide_sim.nc"  # Replace with your desired output path
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
