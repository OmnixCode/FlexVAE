import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QMainWindow
from PyQt5.QtGui import QPixmap, QPainter
import os
from PyQt5.QtCore import Qt



class ImageProcessingApp(QWidget):
    def __init__(self):
        super(ImageProcessingApp, self).__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Processing App')

        # Create dropdown menus for image selection
        self.image1_label = QLabel('Select Image 1:')
        self.image1_combobox = QComboBox()
        self.image1_combobox.currentIndexChanged.connect(self.update_image1)

        self.image2_label = QLabel('Select Image 2:')
        self.image2_combobox = QComboBox()
        self.image2_combobox.currentIndexChanged.connect(self.update_image2)

        self.populate_comboboxes()

        # Create a button to generate and display the processed image
        self.generate_button = QPushButton('Generate')
        self.generate_button.clicked.connect(self.generate_images)

        # Create layout
        layout = QVBoxLayout()
        layout.addWidget(self.image1_label)
        layout.addWidget(self.image1_combobox)
        layout.addWidget(self.image2_label)
        layout.addWidget(self.image2_combobox)
        layout.addWidget(self.generate_button)

        # Create a label to display the processed image
        self.result_label = QLabel()
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def populate_comboboxes(self):
        # Assume images are in the same directory as the script
        current_directory = os.getcwd()+'/results/VAE/'
        image_files = [f for f in os.listdir(current_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        self.image1_combobox.addItems(image_files)
        self.image2_combobox.addItems(image_files)

    def update_image1(self):
        # Update image1_label with the selected image
        selected_image = self.image1_combobox.currentText()
        self.display_image(selected_image, self.image1_label)

    def update_image2(self):
        # Update image2_label with the selected image
        selected_image = self.image2_combobox.currentText()
        self.display_image(selected_image, self.image2_label)

    def generate_images(self):
        # Get the selected image filenames
        image1_filename = self.image1_combobox.currentText()
        image2_filename = self.image2_combobox.currentText()

        # Process the images (you can replace this with your own image processing logic)
        processed_image = self.process_images(image1_filename, image2_filename)

        # Display the processed image
        self.result_label.setPixmap(processed_image)

    def process_images(self, image1_filename, image2_filename):
        # Replace this with your own image processing logic
        # For simplicity, this example just concatenates the two images side by side
        image1_path = os.path.join(os.getcwd()+'/results/VAE/', image1_filename)
        image2_path = os.path.join(os.getcwd()+'/results/VAE/', image2_filename)

        image1 = QPixmap(image1_path)
        image2 = QPixmap(image2_path)

        # Set the spacing between images
        spacing = 10

        # Create a blank canvas for the result image
        result_image = QPixmap(image1.width() + image2.width() + spacing, max(image1.height(), image2.height()))
        result_image.fill(Qt.white)

        # Use QPainter to draw images on the result image with spacing
        painter = QPainter(result_image)
        painter.drawPixmap(0, 0, image1)
        painter.drawPixmap(image1.width() + spacing, 0, image2)
        painter.end()

        return result_image

    def display_image(self, image_filename, label):
        # Display the selected image in the given label
        image_path = os.path.join(os.getcwd()+'/results/VAE/', image_filename)
        image = QPixmap(image_path)
        label.setPixmap(image)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())


    
    
    current_directory =os.getcwd()+'/results/VAE/'