import numpy as np

def calculate_distance(image):
    non_black_pixels = np.argwhere(np.array(image) != 0)
    center = np.mean(non_black_pixels, axis=0)

    new_center_x = int(center[1])
    new_center_y = int(center[0])

    csv_x_and_y = image.shape
    csv_center_x_resize = csv_x_and_y[1] // 2
    csv_center_y_resize = csv_x_and_y[0] // 2
    
    distance = np.sqrt((new_center_x - csv_center_x_resize) ** 2 + (new_center_y - csv_center_y_resize) ** 2)
    return distance, new_center_x, new_center_y

def get_distance_centers(segmented_images, ids_segmented_images):
    distances = []
    
    for i, image in enumerate(segmented_images):
        distance, new_center_x, new_center_y = calculate_distance(image)
        
        if distance is not None:
            distances.append({
                'image_id': ids_segmented_images[i],
                'distance_to_nucleus': distance,
                'coords': [new_center_x, new_center_y]
            })

    return distances