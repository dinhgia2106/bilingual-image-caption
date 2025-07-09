import json
import os

def filter_captions():
    # Get list of image files in data directory
    data_dir = "data"
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    # Extract image IDs from filenames (remove leading zeros)
    image_ids = []
    for filename in image_files:
        # Extract number from filename like "000000001000.jpg" -> 1000
        id_str = filename.replace('.jpg', '')
        image_id = int(id_str)
        image_ids.append(image_id)
    
    print(f"Found {len(image_ids)} images in data directory")
    print(f"Image IDs: {sorted(image_ids)}")
    
    # Load original captions JSON
    with open('data/captions_val2017.json', 'r') as f:
        data = json.load(f)
    
    print(f"Original data contains {len(data['images'])} images and {len(data['annotations'])} annotations")
    
    # Filter images
    filtered_images = []
    for image in data['images']:
        if image['id'] in image_ids:
            filtered_images.append(image)
    
    # Filter annotations
    filtered_annotations = []
    for annotation in data['annotations']:
        if annotation['image_id'] in image_ids:
            filtered_annotations.append(annotation)
    
    # Create filtered data
    filtered_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': filtered_images,
        'annotations': filtered_annotations
    }
    
    print(f"Filtered data contains {len(filtered_data['images'])} images and {len(filtered_data['annotations'])} annotations")
    
    # Save filtered data
    output_file = 'data/captions_filtered.json'
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered data saved to {output_file}")
    
    # Print summary
    print("\nSummary:")
    for image in filtered_data['images']:
        image_annotations = [ann for ann in filtered_data['annotations'] if ann['image_id'] == image['id']]
        print(f"Image {image['file_name']}: {len(image_annotations)} captions")

if __name__ == "__main__":
    filter_captions() 