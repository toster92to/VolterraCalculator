import sys
import os
import qdarkstyle
import numpy as np
import copy
from PyQt5.QtWidgets import (
	QListWidget, QProgressBar, QMessageBox, QComboBox, QHBoxLayout,
	QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
	QInputDialog, QGroupBox, QFormLayout, QLineEdit, QPlainTextEdit, QStyle
)
from PyQt5.QtGui import QIcon, QFontDatabase, QFont
from PyQt5.QtCore import QSize, QThread, pyqtSignal
import matplotlib.pyplot as plt

from Volterra_In_Out_6 import VolterraResponse
from Volterra_File_Operations import readKernels, writeKernels
import Volterra_PlotSystem as vps
import Volterra_File_Operations as vo5


class Worker(QThread):
	"""
	Worker thread for performing operations on Volterra systems.
	"""
	finished = pyqtSignal()
	error = pyqtSignal(str)

	def __init__(self, operation_func, system1, system2, output_system_name, parent=None):
		super().__init__(parent)
		self.operation_func = operation_func
		self.system1 = system1
		self.system2 = system2
		self.output_system_name = output_system_name
		self._is_running = True

	def run(self):
		print(int(QThread.currentThreadId()))
		try:
			if not self._is_running:
				return
			result_system = self.operation_func(self.system1, self.system2)
			if not self._is_running:
				return
			writeKernels(result_system, name=self.output_system_name)
			if not self._is_running:
				return
			self.parent(
			).volterra_systems[self.output_system_name] = result_system
			self.finished.emit()
			
		except Exception as e:
			self.error.emit(str(e))
			
	def cancel(self):
		self._is_running = False


class VolterraCalculator(QWidget):
	"""
	Main GUI class for Volterra System Calculator.
	"""

	def __init__(self):
		super().__init__()
		self.volterra_systems = {}
		self.initUI()

	def initUI(self):
		"""
		Initialize the user interface.
		"""
		QFontDatabase.addApplicationFont('Roboto-Regular.ttf')
		self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5() + """
			* {
				font-family: 'Roboto';
				font-size: 12pt;
			}
			QGroupBox {
				font-size: 14pt;
			}
		""")
		self.setWindowIcon(QIcon('Final/images/logo-no-background.png'))

		self.setGeometry(100, 100, 1000, 600)

		main_layout = QHBoxLayout()
		left_layout = QVBoxLayout()
		right_layout = QVBoxLayout()

		# Create system management layout
		manage_layout = self.create_system_management_layout()
		left_layout.addLayout(manage_layout)

		# Create system operations layout
		operations_layout = self.create_system_operations_layout()
		right_layout.addLayout(operations_layout)

		# Create operation buttons layout
		operation_button_layout = self.create_operation_buttons_layout()
		right_layout.addLayout(operation_button_layout)

		# Set left and right layouts
		main_layout.addLayout(left_layout, 1)
		main_layout.addLayout(right_layout, 2)

		# Set the main layout
		self.setLayout(main_layout)
		self.setWindowTitle('Volterra System Calculator')
		self.update_dropdowns()
		self.update_system_list()

	def create_system_management_layout(self):
		"""
		Create layout for managing Volterra systems.
		"""
		manage_layout = QVBoxLayout()
		self.system_list = QListWidget()
		self.system_list.setSelectionMode(QListWidget.SingleSelection)
		self.system_list.setFont(QFont('Roboto', 12))
		manage_layout.addWidget(self.system_list)
		return manage_layout

	def create_system_operations_layout(self):
		"""
		Create layout for Volterra system operations.
		"""
		operations_layout = QVBoxLayout()

		# Create New System Group
		create_group = QGroupBox("Create New Volterra System")
		create_group.setFont(QFont('Roboto', 14))
		form_layout = self.create_form_layout()
		create_group.setLayout(form_layout)
		create_group.setMinimumHeight(265)
		operations_layout.addWidget(create_group)

		# Create Action Buttons Layout
		action_button_layout = self.create_action_buttons_layout()
		operations_layout.addLayout(action_button_layout)

		# Visualization Group
		visualization_group = QGroupBox("System Visualization")
		visualization_group.setFont(QFont('Roboto', 14))
		visualization_layout = QVBoxLayout()
		visualization_layout.setContentsMargins(20, 20, 20, 20)

		plot_buttons_layout = QHBoxLayout()
		plot_overview_button = QPushButton('Plot Overview')
		plot_kernels_button = QPushButton('Plot Kernels')
		plot_overview_button.setFont(QFont('Roboto', 12))
		plot_kernels_button.setFont(QFont('Roboto', 12))

		plot_overview_button.clicked.connect(self.plotOverview)
		plot_kernels_button.clicked.connect(self.plot_kernels)

		plot_buttons_layout.addWidget(plot_overview_button)
		plot_buttons_layout.addWidget(plot_kernels_button)

		visualization_layout.addLayout(plot_buttons_layout)
		visualization_group.setLayout(visualization_layout)
		visualization_group.setMinimumHeight(130)

		operations_layout.addWidget(visualization_group)

		# Progress Indicator
		self.progress_bar = QProgressBar()
		self.progress_bar.setRange(0, 0)  # Indeterminate state
		self.progress_bar.setVisible(False)
		operations_layout.addWidget(self.progress_bar)

		# Cancel Button
		self.cancel_button = QPushButton('Cancel Operation')
		self.cancel_button.setFont(QFont('Roboto', 12))
		self.cancel_button.setVisible(False)
		self.cancel_button.clicked.connect(self.cancel_operation)
		operations_layout.addWidget(self.cancel_button)

		return operations_layout

	def create_form_layout(self):
		"""
		Create form layout for inputting new Volterra systems.
		"""
		form_layout = QFormLayout()
		self.system_name_input = QLineEdit()
		self.system_name_input.setFont(QFont('Roboto', 12))
		self.kernel_code_input = QPlainTextEdit()
		self.kernel_code_input.setFont(QFont('Roboto', 12))
		self.kernel_code_input.setPlaceholderText(
			"Example:\nh1[0] = 5\nh2[0, 0] = -1.5\nh3[0, 0, 0] = 0.3\nh4[0, 0, 0, 0] = -0.1\nh5[0, 0, 0, 0, 0] = 0.05"
		)
		create_button = QPushButton('Create System')
		create_button.setFont(QFont('Roboto', 12))
		create_button.clicked.connect(self.create_system)

		help_button = QPushButton()
		help_button.setIcon(self.style().standardIcon(
			QStyle.SP_MessageBoxQuestion))
		help_button.setIconSize(QSize(24, 24))
		help_button.setMaximumSize(QSize(30, 30))
		help_button.clicked.connect(self.show_help)

		form_layout.addRow(help_button)
		form_layout.addRow("System Name:", self.system_name_input)
		form_layout.addRow("Kernel Input:", self.kernel_code_input)
		form_layout.addRow(create_button)
		return form_layout

	def create_action_buttons_layout(self):
		"""
		Create layout for action buttons related to Volterra systems.
		"""
		action_button_layout = QHBoxLayout()
		save_button = QPushButton('Save System')
		load_button = QPushButton('Load System')
		delete_button = QPushButton('Delete System')
		zero_kernels_button = QPushButton('Zero Kernels')

		button_size = QSize(150, 30)

		for button in [save_button, load_button, delete_button, zero_kernels_button]:
			button.setFixedSize(button_size)
			button.setFont(QFont('Roboto', 12))

		save_button.clicked.connect(self.saveVolterraSystem)
		load_button.clicked.connect(self.loadVolterraSystem)
		delete_button.clicked.connect(self.delete_system)
		zero_kernels_button.clicked.connect(self.zero_kernels)

		action_button_layout.addWidget(save_button)
		action_button_layout.addWidget(load_button)
		action_button_layout.addWidget(delete_button)
		action_button_layout.addWidget(zero_kernels_button)
		return action_button_layout

	def create_operation_buttons_layout(self):
		"""
		Create layout for operation buttons.
		"""
		operation_button_layout = QVBoxLayout()
		operation_button_row = QHBoxLayout()

		self.system1_dropdown = QComboBox()
		self.system2_dropdown = QComboBox()
		self.system2_dropdown.addItem("None")

		self.system1_dropdown.setFixedWidth(200)
		self.system2_dropdown.setFixedWidth(200)

		self.system1_dropdown.setFont(QFont('Roboto', 12))
		self.system2_dropdown.setFont(QFont('Roboto', 12))

		self.output_system_name_input = QLineEdit()
		self.output_system_name_input.setFont(QFont('Roboto', 12))
		self.output_system_name_input.setPlaceholderText("Output System Name")

		operation_button_row.addWidget(self.system1_dropdown)
		operation_button_row.addWidget(self.system2_dropdown)
		operation_button_row.addWidget(self.output_system_name_input)

		operation_buttons = QHBoxLayout()
		add_button = QPushButton('Add')
		subtract_button = QPushButton('Subtract')
		chain_button = QPushButton('Concatenate')
		invert_button = QPushButton('Invert')
		multiply_button = QPushButton('Multiply')
		self.multiply_factor_input = QLineEdit()
		self.multiply_factor_input.setFixedSize(40, 30)
		self.multiply_factor_input.setFont(QFont('Roboto', 12))
		self.multiply_factor_input.setPlaceholderText("2")

		button_size = QSize(120, 30)

		for button in [add_button, subtract_button, chain_button, invert_button, multiply_button]:
			button.setFixedSize(button_size)
			button.setFont(QFont('Roboto', 12))

		add_button.clicked.connect(
			lambda: self.binaryOperation(vo5.addVolterraSystems))
		subtract_button.clicked.connect(
			lambda: self.binaryOperation(vo5.subtractVolterraSystems))
		multiply_button.clicked.connect(self.multiplyVolterraSystem)
		chain_button.clicked.connect(
			lambda: self.binaryOperation(vo5.chainVolterraSystems_opencl))
		invert_button.clicked.connect(self.invertKernels)

		operation_buttons.addWidget(add_button)
		operation_buttons.addWidget(subtract_button)
		operation_buttons.addWidget(chain_button)
		operation_buttons.addWidget(invert_button)
		operation_buttons.addWidget(multiply_button)
		operation_buttons.addWidget(self.multiply_factor_input)

		operation_button_layout.addLayout(operation_button_row)
		operation_button_layout.addLayout(operation_buttons)
		return operation_button_layout

	def update_ui(self):
		"""
		Update the UI elements based on current state.
		"""
		self.update_dropdowns()
		self.update_system_list()

	def update_dropdowns(self):
		"""
		Update the dropdowns with available Volterra systems.
		"""
		systems = list(self.volterra_systems.keys())
		self.system1_dropdown.clear()
		self.system2_dropdown.clear()
		self.system1_dropdown.addItems(systems)
		self.system2_dropdown.addItems(["None"] + systems)

	def update_system_list(self):
		"""
		Update the system list widget with available Volterra systems.
		"""
		self.system_list.clear()
		self.system_list.addItems(self.volterra_systems.keys())

	def create_system(self):
		"""
		Create a new Volterra system based on user input.
		"""
		system_name = self.system_name_input.text().strip()
		if not system_name:
			QMessageBox.critical(self, 'Input Error',
								 'System name cannot be empty.')
			return

		kernel_code = self.kernel_code_input.toPlainText().strip()
		if not kernel_code:
			QMessageBox.critical(self, 'Input Error',
								 'Kernel code cannot be empty.')
			return

		new_system = VolterraResponse(system_name)

		try:
			exec(kernel_code, {'system2': new_system, 'np': np, 'h0': new_system.h0,
							   'h1': new_system.h1, 'h2': new_system.h2, 'h3': new_system.h3, 'h4': new_system.h4, 'h5': new_system.h5})
			self.volterra_systems[system_name] = new_system
			writeKernels(new_system, name=system_name)
			self.update_ui()
			QMessageBox.information(
				self, 'System Created', f'{system_name} created successfully.')
		except Exception as e:
			QMessageBox.critical(
				self, 'Error', f'Failed to create system: {str(e)}')

	def binaryOperation(self, operation_func):
		"""
		Perform a binary operation on two Volterra systems.
		"""
		system1_name = self.system1_dropdown.currentText()
		system2_name = self.system2_dropdown.currentText()
		output_system_name = self.output_system_name_input.text().strip()

		if not output_system_name:
			output_system_name, ok = QInputDialog.getText(
				self, 'Output System Name Required', 'Enter the name for the output system:')
			if not ok or not output_system_name:
				QMessageBox.critical(
					self, 'Operation Error', 'Output system name is required.')
				return

		if system2_name == "None":
			QMessageBox.critical(self, 'Operation Error',
								 'This operation requires two systems.')
			return

		try:
			system1 = self.volterra_systems[system1_name]
			system2 = self.volterra_systems[system2_name]

			self.progress_bar.setVisible(True)
			self.cancel_button.setVisible(True)
			self.worker = Worker(operation_func, system1,
								 system2, output_system_name, self)
			self.worker.finished.connect(self.on_worker_finished)
			self.worker.error.connect(self.on_worker_error)
			self.worker.start()

		except Exception as e:
			QMessageBox.critical(self, 'Operation Error',
								 f'Error performing operation: {str(e)}')

	def on_worker_finished(self):
		"""
		Handle the completion of a worker thread operation.
		"""
		self.update_ui()
		self.progress_bar.setVisible(False)
		self.cancel_button.setVisible(False)
		QMessageBox.information(self, 'Operation Result',
								'Operation completed successfully.')

	def on_worker_error(self, error_message):
		"""
		Handle errors from a worker thread operation.
		"""
		self.progress_bar.setVisible(False)
		self.cancel_button.setVisible(False)
		QMessageBox.critical(self, 'Operation Error',
							 f'Error performing operation: {error_message}')

	def multiplyVolterraSystem(self):
		"""
		Multiply a Volterra system by a given factor.
		"""
		system_name = self.system1_dropdown.currentText()
		output_system_name = self.output_system_name_input.text().strip()

		if not output_system_name:
			output_system_name, ok = QInputDialog.getText(
				self, 'Output System Name Required', 'Enter the name for the output system:')
			if not ok or not output_system_name:
				QMessageBox.critical(
					self, 'Operation Error', 'Output system name is required.')
				return

		try:
			system = self.volterra_systems[system_name]
			factor = float(self.multiply_factor_input.text())
			result_system = vo5.multiplyVolterraSystem(system, factor)
			self.volterra_systems[output_system_name] = result_system
			writeKernels(result_system, name=output_system_name)
			self.update_ui()
			QMessageBox.information(
				self, 'Operation Result', f'Multiplication by {factor} completed successfully. Result saved as {output_system_name}.')
		except Exception as e:
			QMessageBox.critical(self, 'Operation Error',
								 f'Error performing multiplication: {str(e)}')

	def invertKernels(self):
		"""
		Invert the kernels of a Volterra system.
		"""
		system_name = self.system1_dropdown.currentText()
		output_system_name = self.output_system_name_input.text().strip()

		if not output_system_name:
			output_system_name, ok = QInputDialog.getText(
				self, 'Output System Name Required', 'Enter the name for the output system:')
			if not ok or not output_system_name:
				QMessageBox.critical(
					self, 'Operation Error', 'Output system name is required.')
				return

		try:
			system = self.volterra_systems[system_name]
			self.progress_bar.setVisible(True)
			self.cancel_button.setVisible(True)

			self.worker = Worker(vo5.invertKernels, system, None, output_system_name, self)
			self.worker.finished.connect(self.on_worker_finished)
			self.worker.error.connect(self.on_worker_error)
			self.worker.start()
			

		except Exception as e:
			QMessageBox.critical(self, 'Operation Error',
								 f'Error performing inversion: {str(e)}')

	def cancel_operation(self):
		"""
		Cancel the currently running operation.
		"""
		if hasattr(self, 'worker'):
			self.worker.cancel()
			self.progress_bar.setVisible(False)
			self.cancel_button.setVisible(False)
			QMessageBox.information(
				self, 'Operation Cancelled', 'The operation has been cancelled.')

	def saveVolterraSystem(self):
		"""
		Save a Volterra system to a file.
		"""
		system_name, ok = QInputDialog.getText(
			self, 'Save System', 'Enter a name for the system:')
		if ok and system_name:
			try:
				system = self.volterra_systems[system_name]
				writeKernels(system, name=system_name)
				QMessageBox.information(
					self, 'System Saved', f'System "{system_name}" has been saved successfully.')
			except Exception as e:
				QMessageBox.critical(self, 'Save Error',
									 f'Error saving system: {str(e)}')

	# def loadVolterraSystem(self):
	#	 """
	#	 Load a Volterra system from a file.
	#	 """
	#	 folder_name = QFileDialog.getExistingDirectory(
	#		 self, 'Select System Folder')

	#	 if folder_name:
	#		 system_name, ok = QInputDialog.getText(
	#			 self, 'Load System', 'Enter the name of the system to load (e.g., "DUT" for files named h0DUT.csv, h1DUT.csv, etc.):')

	#		 if ok and system_name:
	#			 try:
	#				 relative_folder_path = os.path.relpath(folder_name)
	#				 system = readKernels(
	#					 name=system_name, folderName=relative_folder_path)
	#				 self.volterra_systems[system_name] = system
	#				 self.update_ui()
	#				 QMessageBox.information(
	#					 self, 'System Loaded', f'System "{system_name}" has been loaded successfully.')
	#			 except Exception as e:
	#				 QMessageBox.critical(
	#					 self, 'Load Error', f'Error loading system: {str(e)}')

	def loadVolterraSystem(self):
		"""
		Load a Volterra system from a file.
		"""

		system_name, ok = QInputDialog.getText(
			self, 'Load System', 'Enter the name of the system to load (e.g., "DUT" for files named h0DUT.csv, h1DUT.csv, etc.):')

		if ok:
			try:
				system = readKernels(
					name=system_name)
				self.volterra_systems[system_name] = system
				self.update_ui()
				QMessageBox.information(
					self, 'System Loaded', f'System "{system_name}" has been loaded successfully.')
			except FileNotFoundError:
				QMessageBox.warning(
					self, 'Load Warning', f'The system "{system_name}" could not be found. Please ensure that the system files are located in the "Volterra_Kernels" folder under the current working directory.')
			except Exception as e:
				QMessageBox.critical(
					self, 'Load Error', f'Error loading system: {str(e)}')

	def delete_system(self):
		"""
		Delete a Volterra system.
		"""
		selected_item = self.system_list.currentItem()
		if selected_item:
			system_name = selected_item.text()
			del self.volterra_systems[system_name]  # Delete from memory

			folder_name = 'Volterra_Kernels'  # Assuming all systems are stored in this folder
			file_paths = [
				os.path.join(folder_name, f'h0{system_name}.csv'),
				os.path.join(folder_name, f'h1{system_name}.csv'),
				os.path.join(folder_name, f'h2{system_name}.csv'),
				os.path.join(folder_name, f'h3{system_name}.csv'),
				os.path.join(folder_name, f'h4{system_name}.csv'),
				os.path.join(folder_name, f'h5{system_name}.csv')
			]

			for file_path in file_paths:
				try:
					os.remove(file_path)
				except FileNotFoundError:
					pass  # Ignore if the file doesn't exist

			self.update_system_list()
			QMessageBox.information(
				self, 'System Deleted', f'{system_name} has been deleted successfully.')
		else:
			QMessageBox.warning(self, 'No Selection',
								'Please select a system to delete.')

	def zero_kernels(self):
		"""
		Set all kernels of a Volterra system to zero.
		"""
		selected_item = self.system_list.currentItem()
		if selected_item:
			system_name = selected_item.text()
			system = self.volterra_systems[system_name]
			system.set_kernels(0, np.zeros_like(system.h1), np.zeros_like(system.h2),
							   np.zeros_like(system.h3), np.zeros_like(system.h4), np.zeros_like(system.h5))
			writeKernels(system, name=system_name)
			QMessageBox.information(
				self, 'Kernels Zeroed', f'All kernels in {system_name} have been set to zero.')
		else:
			QMessageBox.warning(self, 'No Selection',
								'Please select a system to zero the kernels.')

	def plotOverview(self):
		"""
		Plot the overview of a selected Volterra system.

		"""
		order = 5
		inputBandwidth = 3.333333e9

		Amplitude = 0.5
		timestep = 1/(2*inputBandwidth*order)

		selected_item = self.system_list.currentItem()
		if selected_item:
			system_name = selected_item.text()
			system = self.volterra_systems[system_name]
			try:
				# Adjust amplitude and timestep as needed
				vps.plotOverview(system, Amplitude, order*timestep)
				plt.show()
			except Exception as e:
				QMessageBox.critical(self, 'Plotting Error',
									 f'Error plotting system overview: {str(e)}')
		else:
			QMessageBox.warning(self, 'No Selection',
								'Please select a system to plot.')

	def plot_kernels(self):
		"""
		Plot the kernels of a selected Volterra system.
		"""
		selected_item = self.system_list.currentItem()
		if selected_item:
			system_name = selected_item.text()
			system = self.volterra_systems[system_name]
			try:
				vps.plot1(system, 0, system.len1)
				vps.plot2(system, 0, system.len2)
				vps.plot3(system, 0, system.len3)
				vps.interactive_plot_h4(system)
				vps.interactive_plot_h5(system)
			except Exception as e:
				QMessageBox.critical(self, 'Plotting Error',
									 f'Error plotting kernels: {str(e)}')
		else:
			QMessageBox.warning(self, 'No Selection',
								'Please select a system to plot.')

	def show_help(self):
		"""
		Show help instructions for inputting kernels.
		"""
		instructions = """
		Instructions for Inputting Kernels:
		1. System Name: Enter a unique name for your Volterra system in the "System Name" field.

		2. Kernel Code:
		- Use the variable names h0, h1, h2, h3, h4, and h5 to refer to the different order kernels.
		- The dimensions of the kernels are:
			* h0: Scalar
			* h1: 1D array of length 10
			* h2: 2D array of size 15x15
			* h3: 3D array of size 15x15x15
			* h4: 4D array of size 15x15x15x15
			* h5: 5D array of size 15x15x15x15x15

		3. Example:
		h1[0] = 5
		h2[0, 0] = -1.5
		h3[0, 0, 0] = 0.3
		h4[0, 0, 0, 0] = -0.1
		h5[0, 0, 0, 0, 0] = 0.05

		4. Execution: Ensure the syntax is correct and all necessary values are specified.
		"""
		QMessageBox.information(self, 'Help', instructions)


def main():
	"""
	Main entry point of the application.
	"""
	app = QApplication(sys.argv)
	ex = VolterraCalculator()
	ex.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	main()
