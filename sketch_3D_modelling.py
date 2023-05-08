import sys
import faulthandler
import traceback
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize, QLine

import numpy as np
import os
import igl
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point

# The DrawingPanel class represents the panel where users can draw their shapes.
class DrawingPanel(QWidget):
    def __init__(self, parent=None):
        super(DrawingPanel, self).__init__(parent)
        self.points = []
        self.current_line = []
        self.start_point = None

        # Set a custom size for self.image
        custom_size = QSize(1200, 800)
        self.image = QImage(custom_size, QImage.Format_RGB32)
        self.image.fill(Qt.white)

    def resizeEvent(self, event):
        # Update the image size when the widget is resized
        if event.oldSize().width() > 0 and event.oldSize().height() > 0:
            new_image = QImage(event.size(), QImage.Format_RGB32)
            new_image.fill(Qt.white)
            painter = QPainter(new_image)
            painter.drawImage(0, 0, self.image)
            painter.end()
            self.image = new_image

    def mousePressEvent(self, event):
        # draw a line when the left mouse button is pressed
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            if not self.start_point:
                self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        # Update the line as the mouse is moved while the left button is pressed
        if event.buttons() & Qt.LeftButton and self.start_point and self.rect().contains(event.pos()):
            self.current_line = [self.start_point, event.pos()]
            self.update()

    def mouseReleaseEvent(self, event):
        # Draw the line on the image when the left mouse button is released
        if event.button() == Qt.LeftButton and self.start_point and self.rect().contains(event.pos()):
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 5))
            painter.drawLine(self.start_point, event.pos())
            painter.end()
            self.update()

            # Store the start and end points of the drawn line
            self.points.append(self.start_point)
            self.points.append(event.pos())
            self.start_point = event.pos()

    # Draw the image and the current line on the widget
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image, event.rect())

        if self.current_line:
            painter.setPen(QPen(Qt.black, 5, Qt.SolidLine))
            painter.drawLine(self.current_line[0], self.current_line[1])

    def clear_panel(self):
        # Clears the drawing panel
        self.image.fill(Qt.white)
        self.points = []
        self.start_point = None
        self.current_line = []
        self.update()

#The DrawingApp class represents the window of the drawing app.
class DrawingApp(QWidget):
    def __init__(self, parent=None):
        super(DrawingApp, self).__init__(parent)
        self.setWindowTitle("Draw Your 3D Model")
        self.setFixedSize(1200, 800)

        # Center the window on the screen
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # Set the background color of the window to black
        #self.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        self.drawing_panel = DrawingPanel()
        layout.addWidget(self.drawing_panel, stretch=1)

        buttons_layout = QHBoxLayout()

        # Define the CSS style for the buttons
        button_style = """
        QPushButton {
            font-size: 22px;
            background-color: #3a9de8;
            color: white;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #2c7db7;
        }
        QPushButton:pressed {
            background-color: #1a5c86;
        }
        """

        # build buttons
        generate_mesh_button = QPushButton("Generate Mesh")
        generate_mesh_button.setStyleSheet(button_style)
        generate_mesh_button.clicked.connect(self.generate_mesh)
        buttons_layout.addWidget(generate_mesh_button)

        clear_panel_button = QPushButton("Clear Panel")
        clear_panel_button.setStyleSheet(button_style)
        clear_panel_button.clicked.connect(self.drawing_panel.clear_panel)
        buttons_layout.addWidget(clear_panel_button)

        layout.addLayout(buttons_layout)
        self.setLayout(layout)
        self.main_window = None

    def on_main_window_closed(self):
        self.main_window.closed.disconnect(self.on_main_window_closed)
        self.main_window = None

    def generate_mesh(self):
        print("Generate Mesh button pressed")
        print(np.array(self.drawing_panel.points)) # format: [PyQt5.QtCore.QPoint(631, 248) PyQt5.QtCore.QPoint(343, 379)
                                                             # PyQt5.QtCore.QPoint(343, 379) PyQt5.QtCore.QPoint(473, 571)
                                                             # PyQt5.QtCore.QPoint(473, 571) PyQt5.QtCore.QPoint(668, 233)
                                                             # PyQt5.QtCore.QPoint(668, 233) PyQt5.QtCore.QPoint(630, 250)]

        # Extracting x and y values from QPoint objects
        xy_values = [(point.x(), point.y()) for point in self.drawing_panel.points]
        xy_values = np.array(xy_values)
        try:
            if check_watertight(xy_values):
                root_folder = os.getcwd()
                data_directory = os.path.join(os.getcwd(), "data")
                if not os.path.exists(data_directory):
                    os.makedirs(data_directory)

                #remove the last point from the array
                xy_values = xy_values[:-1]

                # add the first point to the end of array
                xy_values = np.vstack((xy_values, xy_values[0]))
                print(xy_values)

                # the shape was upside down
                xy_values[:, 1] = -xy_values[:, 1]

                # apply delaunay triangulation on shrink shape and the center
                cent = compute_center(xy_values)
                smaller_shape = shrink_shape(xy_values)
                smaller_shape_cent = np.vstack((smaller_shape, (cent)))
                smaller_shape_v, smaller_shape_f = delaunay_triangulation(smaller_shape_cent)
                igl.write_triangle_mesh(os.path.join(root_folder, "data", "shrink_shape.obj"), smaller_shape_v, smaller_shape_f)

                # extrude smaller_shape_v along z axis, and connect the points between the original shape and the shrink shape
                vertices, faces = mesh_inflation(xy_values, smaller_shape_f, smaller_shape_v, cent)
                igl.write_triangle_mesh(os.path.join(root_folder, "data", "inflation.obj"), vertices, faces)

                vertices, faces = mirror_mesh(vertices, faces)

                # mirror_mesh will occur duplicate vertices
                # Remove duplicate vertices and construct the new order of faces
                vertices, unique_indices = np.unique(vertices, axis=0, return_inverse=True)
                faces = unique_indices[faces]
                igl.write_triangle_mesh(os.path.join(root_folder, "data", "output.obj"), vertices, faces)


                # Create a MainWindow instance if it doesn't exist, or reuse the existing one
                if self.main_window is None:
                    self.main_window = MainWindow()
                    self.main_window.closed.connect(self.on_main_window_closed)

                self.main_window.show()
                print("done")
            else:
                # Display a message box if the mesh is not closed
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("Error")
                msg_box.setText("The drawn shape is not closed. Please draw again.")
                msg_box.setIcon(QMessageBox.Warning)
                msg_box.exec()
        except Exception as e:
            print("Exception occurred:", e)
            traceback.print_exc()


def compute_centroid(vertices): # compute the centroid 
    centroid = np.mean(vertices, axis=0)
    return centroid

def compute_center(vertices):
    # This function finds the minimum and maximum x and y coordinates (which define the bounding box of the shape)
    # then computes the center of this bounding box.
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (max_coords + min_coords) / 2
    return center

def shrink_shape(vertices, shrink_factor = 0.65):
    centroid = compute_centroid(vertices)
    radial_vectors = vertices - centroid
    new_vertices = centroid + (radial_vectors * shrink_factor)
    return new_vertices

def delaunay_triangulation(points):
    # Create Delaunay Triangulation
    delaunay = Delaunay(points)

    # Get vertices and faces from triangulation
    new_points = delaunay.points

    # Add z axis to new_points
    new_points = np.hstack((new_points, np.zeros((new_points.shape[0], 1))))

    faces = delaunay.simplices

    # Remove triangles outside the polygon
    polygon = Polygon(points)
    filtered_faces = []

    for face in faces:
        triangle = Polygon(new_points[face, :2])
        centroid = Point(triangle.centroid)

        if polygon.contains(centroid):
            filtered_faces.append(face)

    return new_points, np.array(filtered_faces)

def mesh_inflation(original_shape, faces, smaller_shape, z = 80):
    original_shape = np.hstack((original_shape, np.zeros((original_shape.shape[0], 1))))
    print(smaller_shape)
    smaller_shape[:, 2] = 80

    # Combine the old and new shape vertices
    combined_vertices = np.vstack((original_shape, smaller_shape))

    # Create the faces that connect the old and new shapes
    num_vertices = len(original_shape)
    connecting_faces = []
    for i in range(num_vertices - 1):
        connecting_faces.append([i, i + 1, i + num_vertices + 1])
        connecting_faces.append([i, i + num_vertices + 1, i + num_vertices])
    connecting_faces.append([num_vertices - 1, 0, num_vertices])
    connecting_faces.append([num_vertices - 1, num_vertices, num_vertices * 2 - 1])

    # Combine the original, connecting, and centroid faces
    combined_faces = np.vstack((faces + num_vertices, connecting_faces))

    combined_vertices, unique_indices = np.unique(combined_vertices, axis=0, return_inverse=True)
    combined_faces = unique_indices[combined_faces]

    return combined_vertices, combined_faces

def mirror_mesh(vertices, faces):
    # Create a copy of the vertices array and flip the z-axis values
    mirrored_vertices = np.copy(vertices)
    mirrored_vertices[:, 2] = -mirrored_vertices[:, 2]

    # Combine the original and mirrored vertices
    combined_vertices = np.vstack((vertices, mirrored_vertices))

    # Create faces for the mirrored mesh, offset by the number of original vertices
    mirrored_faces = np.copy(faces) + len(vertices)

    # Combine the original and mirrored faces
    combined_faces = np.vstack((faces, mirrored_faces))

    return combined_vertices, combined_faces

def check_watertight(points): # check if the user's drawing is a closed shape
    if len(points) < 3:
        return False

    first_point = points[0]
    last_point = points[-1]

    # Check if the first and last points are the same (or very close)
    distance = np.sqrt((first_point[0] - last_point[0]) ** 2 + (first_point[1] - last_point[1]) ** 2)
    return distance < 7


import vtk
from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QCheckBox, QMessageBox, QFileDialog
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5.QtCore import pyqtSignal

# The PointMoveCallback class is responsible to handle point movement
class PointMoveCallback:
    def __init__(self, polydata, actor, render_window, normals_filter):
        self.polydata = polydata
        self.actor = actor
        self.render_window = render_window
        self.normals_filter = normals_filter
        self.handle_widgets = {}

    def register_handle_widget(self, handle_widget, point_id):
        # Register handle_widget for each point with its corresponding point_id
        self.handle_widgets[handle_widget] = point_id

    # This method is called when the handle_widget is moved
    def __call__(self, caller, event):
        point_id = self.handle_widgets[caller]
        handle_representation = caller.GetRepresentation()
        new_position = handle_representation.GetWorldPosition()

        # Update the point position in the polydata
        self.polydata.GetPoints().SetPoint(point_id, new_position)

        # Update the handle widget's representation
        handle_representation.SetWorldPosition(new_position)

        self.polydata.Modified()
        self.normals_filter.Update()
        self.actor.GetProperty().Modified()
        self.actor.GetMapper().Update()
        self.render_window.Render()

# The MainWindow class is responsible for displaying and interacting with the 3D object
class MainWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("3D Object")
        self.setGeometry(100, 100, 1200, 800)
        self.center()

        self.frame = QWidget()
        self.vl = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtk_widget)

        self.init_vtk_scene()

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.save_button = QPushButton("Save", self.frame)
        button_style = """
        QPushButton {
            font-size: 22px;
            background-color: #3a9de8;
            color: white;
            padding: 8px;
            border-radius: 5px;
        }
        QPushButton:hover {
            background-color: #2c7db7;
        }
        QPushButton:pressed {
            background-color: #1a5c86;
        }
        """
        self.save_button.setStyleSheet(button_style)
        self.save_button.clicked.connect(self.save_obj)
        self.vl.addWidget(self.save_button)

        # Add checkbox
        self.hide_vertices_checkbox = QCheckBox('Hide Vertice Draggable Points', self.frame)
        self.hide_vertices_checkbox.setStyleSheet("QCheckBox { font-size: 20px}")
        self.hide_vertices_checkbox.stateChanged.connect(self.hide_vertices)
        self.vl.addWidget(self.hide_vertices_checkbox)

        # Add checkbox
        self.edges_visibility_checkbox = QCheckBox('Hide Edges', self.frame)
        self.edges_visibility_checkbox.setStyleSheet("QCheckBox { font-size: 20px}")
        self.edges_visibility_checkbox.setChecked(True)  # Set the checkbox as checked by default
        self.edges_visibility_checkbox.stateChanged.connect(self.toggle_edges_visibility)
        self.vl.addWidget(self.edges_visibility_checkbox)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)


    # Handle the closing of the main window
    def closeEvent(self, event):
        self.closed.emit()
        super(MainWindow, self).closeEvent(event)

    # Save the modified 3D object to an OBJ file
    def save_obj(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getSaveFileName(self, "Save 3D Object",
                                                   "", "OBJ Files (*.obj);;All Files (*)", options=options)

        if file_name:
            if not file_name.endswith('.obj'):
                file_name += '.obj'
            writer = vtk.vtkOBJWriter()
            writer.SetFileName(file_name)
            writer.SetInputData(self.point_move_callback.polydata)
            success = writer.Write()

            msg = QMessageBox()
            msg.setWindowTitle("Save 3D Object")

            if success:
                msg.setText(f"The 3D model '{file_name}' has been saved successfully.")
                msg.setIcon(QMessageBox.Information)
            else:
                msg.setText(f"An error occurred while saving the 3D model '{file_name}'.")
                msg.setIcon(QMessageBox.Critical)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        else:
            msg = QMessageBox()
            msg.setWindowTitle("Save 3D Object")
            msg.setText("No file name specified. The 3D model was not saved.")
            msg.setIcon(QMessageBox.Warning)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


    def hide_vertices(self, state):
        # Hide or show draggable points based on the state of the checkbox
        for handle_widget, _ in self.point_move_callback.handle_widgets.items():
            handle_widget.SetEnabled(not state)
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_edges_visibility(self, state):
        if state == Qt.Checked:
            self.actor.GetProperty().EdgeVisibilityOff()
        else:
            self.actor.GetProperty().EdgeVisibilityOn()
        self.vtk_widget.GetRenderWindow().Render()

    def center(self):   # Center the main window on the screen
        # get the screen resolution
        screen = QDesktopWidget().screenGeometry()
        # get the size of the window
        size = self.geometry()
        # set the position of the window in the center of the screen
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))

    def init_vtk_scene(self): #order: filters -> mapper -> actor -> renderer -> renderer window
        # Read the OBJ file
        reader = vtk.vtkOBJReader()
        reader.SetFileName('data/output.obj')
        reader.Update() # to process the data

        # convert the polygonal mesh into a triangulated mesh
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(reader.GetOutputPort())
        triangle_filter.Update()

        polydata = triangle_filter.GetOutput()

        # Create a normals filter to compute point normals for the 3D object.
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polydata)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.Update()

        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals_filter.GetOutputPort())

        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        self.actor.GetProperty().SetRepresentationToSurface()  # Use surface representation NEW!!!!!!!!!!!!!!!!!!!!!!!!
        self.actor.GetProperty().EdgeVisibilityOff()  # Display edges of the triangular meshes NEW!!!!!!!!!!!!!!!!!!!!!!!!
        self.actor.GetProperty().SetColor(1, 0.5, 0.5)  # RGB values for yellow are (1, 1, 0)

        # Set up the renderer
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1, 1, 1)  # Set the background color to white
        renderer.AddActor(self.actor)

        # Set up the renderer window
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(renderer)
        interactor = self.vtk_widget

        # Set up the interactor style
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interactor_style)

        # Set up handle widgets for point interaction
        self.point_move_callback = PointMoveCallback(polydata, self.actor, render_window, normals_filter)

        #  create a sphere handle representation and a handle widget for each point
        #  Register the handle widget and its corresponding point ID with the point move callback
        for i in range(polydata.GetNumberOfPoints()):
            handle_rep = vtk.vtkSphereHandleRepresentation()
            handle_rep.SetWorldPosition(polydata.GetPoints().GetPoint(i))
            handle_rep.GetProperty().SetColor(0, 0, 0)  # Set color to black
            handle_widget = vtk.vtkHandleWidget()
            handle_widget.SetInteractor(interactor)
            handle_widget.SetRepresentation(handle_rep)
            handle_widget.SetEnabled(True)
            handle_widget.SetPriority(1)

            self.point_move_callback.register_handle_widget(handle_widget, i)
            handle_widget.AddObserver("InteractionEvent", self.point_move_callback)

        # Initialize and start the interactor
        interactor.Initialize()

        # Reset the camera to show the entire 3D object
        renderer.ResetCamera()

        interactor.Start()

if __name__ == '__main__':
    faulthandler.enable()
    app = QApplication(sys.argv)
    mainWindow = DrawingApp()
    mainWindow.show()
    sys.exit(app.exec_())
