import os
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split

# Step 1: Read particle locations
particle_locations = np.loadtxt('particle_locations.txt', dtype='str')

# Step 2: Read class mask
with mrcfile.open('class_mask.mrc', permissive=True) as mrc:
    class_mask = mrc.data.squeeze()

# Step 3: Define zoom factor
zoom_factor = 2  # Zoom factor around the particle

# Step 4: Define a mapping of particle types to numeric labels
particle_type_map = {
    'background': 0,
    '4V94': 1,
    '4CR2': 2,
    '1QVR': 3,
    '1BXN': 4,
    '3CF3': 5,
    '1U6G': 6,
    '3D2F': 7,
    '2CG9': 8,
    '3H84': 9,
    '3GL1': 10,
    '3QM1': 11,
    '1S3X': 12,
    '5MRC': 13,
    'vesicle': 14,
    'fiducial': 15
}

# Step 5: Split dataset into train and test
train_locations, test_locations = train_test_split(particle_locations, test_size=0.2, random_state=42)

# Step 6: Create directories for images and labels if not exist
for directory in ['images/train', 'images/test', 'labels/train', 'labels/test']:
    os.makedirs(directory, exist_ok=True)

# Step 7: Iterate through each particle and save images and labels
for idx, particle in enumerate(train_locations):
    class_id = particle[0]
    x, y, z = int(particle[1]), int(particle[2]), int(particle[3])

    # Calculate bounding box coordinates
    min_x = max(0, x - 10)
    max_x = min(class_mask.shape[1] - 1, x + 10)
    min_y = max(0, y - 10)
    max_y = min(class_mask.shape[0] - 1, y + 10)

    # Refine bounding box using particle segmentation
    refined_min_x = min_x
    refined_min_y = min_y
    refined_max_x = max_x
    refined_max_y = max_y

    # Start shrinking the box along each edge
    # Check upper border
    for upper_border in range(refined_min_y, refined_max_y):
        if np.any(class_mask[x, upper_border] != class_id) and np.any(class_mask[x, upper_border + 1] == class_id):
            refined_min_y = upper_border + 1
            break

    # Check lower border
    for lower_border in reversed(range(refined_min_y, refined_max_y)):
        if np.any(class_mask[x, lower_border] != class_id) and np.any(class_mask[x, lower_border - 1] == class_id):
            refined_max_y = lower_border
            break

    # Check left border (similar logic as upper border)
    for left_border in range(refined_min_x, refined_max_x):
        if np.any(class_mask[left_border, y] != class_id) and np.any(class_mask[left_border + 1, y] == class_id):
            refined_min_x = left_border + 1
            break

    # Check right border (similar logic as upper border)
    for right_border in reversed(range(refined_min_x, refined_max_x)):
        if np.any(class_mask[right_border, y] != class_id) and np.any(class_mask[right_border - 1, y] == class_id):
            refined_max_x = right_border
            break

    # Zoom into the refined region around the particle
    zoom_min_x = max(0, x - zoom_factor * (refined_max_x - refined_min_x))
    zoom_max_x = min(class_mask.shape[1] - 1, x + zoom_factor * (refined_max_x - refined_min_x))
    zoom_min_y = max(0, y - zoom_factor * (refined_max_y - refined_min_y))
    zoom_max_y = min(class_mask.shape[0] - 1, y + zoom_factor * (refined_max_y - refined_min_y))


    # Create a new figure and axis for each particle
    fig, ax = plt.subplots()

    # Plot the zoomed-in region
    ax.imshow(class_mask[z, zoom_min_y:zoom_max_y, zoom_min_x:zoom_max_x], cmap='jet')

    # Plot the bounding box
    rect = Rectangle((min_x - zoom_min_x, min_y - zoom_min_y), max_x - min_x, max_y - min_y,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Annotate with class id
    ax.text(min_x - zoom_min_x, min_y - zoom_min_y, class_id, color='r', fontsize=8, verticalalignment='top')

    # Set axis limits and turn off axis labels
    ax.set_xlim(0, zoom_max_x - zoom_min_x)
    ax.set_ylim(zoom_max_y - zoom_min_y, 0)
    ax.axis('off')

    # Save the resulting image with a unique filename
    filename = f'bounding_box_{idx}.png'
    plt.savefig(f'images/train/{filename}', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save label in YOLO format
    with open(f'labels/train/{os.path.splitext(filename)[0]}.txt', 'w') as label_file:
        label_file.write(f"{particle_type_map[class_id]} {(x - min_x) / (max_x - min_x)} {(y - min_y) / (max_y - min_y)} 1 1\n")

# Repeat the process for test data
for idx, particle in enumerate(test_locations):
    class_id = particle[0]
    x, y, z = int(particle[1]), int(particle[2]), int(particle[3])

    min_x = max(0, x - 10)
    max_x = min(class_mask.shape[1] - 1, x + 10)
    min_y = max(0, y - 10)
    max_y = min(class_mask.shape[0] - 1, y + 10)

     # Refine bounding box using particle segmentation
    refined_min_x = min_x
    refined_min_y = min_y
    refined_max_x = max_x
    refined_max_y = max_y

    # Start shrinking the box along each edge
    # Check upper border
    for upper_border in range(refined_min_y, refined_max_y):
        if np.any(class_mask[x, upper_border] != class_id) and np.any(class_mask[x, upper_border + 1] == class_id):
            refined_min_y = upper_border + 1
            break

    # Check lower border
    for lower_border in reversed(range(refined_min_y, refined_max_y)):
        if np.any(class_mask[x, lower_border] != class_id) and np.any(class_mask[x, lower_border - 1] == class_id):
            refined_max_y = lower_border
            break

    # Check left border (similar logic as upper border)
    for left_border in range(refined_min_x, refined_max_x):
        if np.any(class_mask[left_border, y] != class_id) and np.any(class_mask[left_border + 1, y] == class_id):
            refined_min_x = left_border + 1
            break

    # Check right border (similar logic as upper border)
    for right_border in reversed(range(refined_min_x, refined_max_x)):
        if np.any(class_mask[right_border, y] != class_id) and np.any(class_mask[right_border - 1, y] == class_id):
            refined_max_x = right_border
            break

    # Zoom into the refined region around the particle
    zoom_min_x = max(0, x - zoom_factor * (refined_max_x - refined_min_x))
    zoom_max_x = min(class_mask.shape[1] - 1, x + zoom_factor * (refined_max_x - refined_min_x))
    zoom_min_y = max(0, y - zoom_factor * (refined_max_y - refined_min_y))
    zoom_max_y = min(class_mask.shape[0] - 1, y + zoom_factor * (refined_max_y - refined_min_y))


    fig, ax = plt.subplots()

    ax.imshow(class_mask[z, zoom_min_y:zoom_max_y, zoom_min_x:zoom_max_x], cmap='jet')

    rect = Rectangle((min_x - zoom_min_x, min_y - zoom_min_y), max_x - min_x, max_y - min_y,
                     linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    ax.text(min_x - zoom_min_x, min_y - zoom_min_y, class_id, color='r', fontsize=8, verticalalignment='top')

    ax.set_xlim(0, zoom_max_x - zoom_min_x)
    ax.set_ylim(zoom_max_y - zoom_min_y, 0)
    ax.axis('off')

    filename = f'bounding_box_{idx}.png'
    plt.savefig(f'images/test/{filename}', bbox_inches='tight', pad_inches=0)
    plt.close()

    with open(f'labels/test/{os.path.splitext(filename)[0]}.txt', 'w') as label_file:
        label_file.write(f"{particle_type_map[class_id]} {(x - min_x) / (max_x - min_x)} {(y - min_y) / (max_y - min_y)} 1 1\n")