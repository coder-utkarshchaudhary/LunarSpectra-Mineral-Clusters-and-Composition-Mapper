import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import csv

# Load the image
image_path = "C:/Users/kanis/Downloads/slice_test.png"
image = mpimg.imread(image_path)

# Read the coordinates from the CSV file
coordinates = []
with open("C:/Users/kanis/Downloads/bbox_results.csv", 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        class_id = int(row[0])
        x_center = float(row[1])
        y_center = float(row[2])
        width = float(row[3])
        height = float(row[4])
        coordinates.append((x_center, y_center, width, height))

# Plot the image
fig, ax = plt.subplots(1, figsize=(12, 12))  # Increase figure size for better visibility
ax.imshow(image)

# Plot the bounding boxes
for coord in coordinates:
    x, y, width, height = coord

    rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

# Show the plot
plt.show()
